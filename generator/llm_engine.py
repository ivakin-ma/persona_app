import json
import re
from copy import deepcopy
from functools import lru_cache

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
REQUIRE_CUDA = True

_tokenizer = None
_model = None
_error = None
_device_info = ""
_last_generation_error = ""
_last_generation_text = ""
_JSON_PARSE_FAILED = object()


def get_model_error():
    return _error


def get_last_generation_error():
    return _last_generation_error


def get_last_generation_text():
    return _last_generation_text


def get_device_info():
    return _device_info


def is_model_loaded():
    return _model is not None and _tokenizer is not None


def get_model_name():
    return MODEL_NAME


def _model_devices(model):
    devices = {param.device.type for param in model.parameters()}
    devices.update(buffer.device.type for buffer in model.buffers())
    return devices


def load_llm():
    """Lazy CUDA LLM loader. It refuses silent CPU fallback when REQUIRE_CUDA is enabled."""
    global _tokenizer, _model, _error, _device_info
    if is_model_loaded():
        return _tokenizer, _model
    if _error is not None:
        return None, None

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if REQUIRE_CUDA and not torch.cuda.is_available():
            _error = (
                "CUDA недоступна для PyTorch. Установите CUDA-сборку torch и проверьте драйвер NVIDIA; "
                "CPU fallback отключён, чтобы модель не запускалась на процессоре."
            )
            return None, None

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            _device_info = f"cuda:0 ({gpu_name}, {total_vram:.1f} GB VRAM)"
        else:
            _device_info = "cpu"

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        quant_config = None
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as exc:
            if REQUIRE_CUDA:
                print(f"4-bit quantization is unavailable, loading fp16 on CUDA instead: {exc}")
            quant_config = None

        kwargs = {
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["device_map"] = "cpu"

        if quant_config is not None and torch.cuda.is_available():
            kwargs["quantization_config"] = quant_config
        else:
            kwargs["dtype"] = torch.float16

        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
        _model.eval()
        devices = _model_devices(_model)
        if REQUIRE_CUDA and devices != {"cuda"}:
            _error = f"Модель загружена не только на CUDA: {sorted(devices)}."
            _model = None
            _tokenizer = None
            return None, None
        _normalize_generation_config(_model.generation_config)
        return _tokenizer, _model
    except Exception as exc:
        _error = str(exc)
        _tokenizer = None
        _model = None
        return None, None


def _normalize_generation_config(config, do_sample=False, temperature=None):
    config.do_sample = do_sample
    if do_sample:
        config.temperature = temperature or 0.25
        config.top_p = 0.9
    else:
        config.temperature = None
        config.top_p = None
        config.top_k = None
    return config


def _repair_json_text(text):
    repaired = text.strip()
    repaired = re.sub(r"}\s*(?=\{)", "},", repaired)
    repaired = re.sub(r"]\s*(?=\{)", "],", repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _loads_json_candidate(candidate):
    try:
        return json.loads(candidate)
    except Exception:
        pass

    repaired = _repair_json_text(candidate)
    if repaired != candidate:
        try:
            return json.loads(repaired)
        except Exception:
            pass
    raise ValueError("Invalid JSON")


def extract_json(text, fallback=None):
    if fallback is None:
        fallback = {}
    if not text:
        return fallback
    text = text.strip()
    # Remove markdown fences if model returned them.
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return _loads_json_candidate(text)
    except Exception:
        pass

    # Try to parse first JSON object or array from the answer.
    start_candidates = [i for i in [text.find("{"), text.find("[")] if i != -1]
    if not start_candidates:
        return fallback
    start = min(start_candidates)
    for end in range(len(text), start, -1):
        candidate = text[start:end].strip()
        try:
            return _loads_json_candidate(candidate)
        except Exception:
            continue
    return fallback


def llm_generate_json(system_prompt, user_prompt, fallback=None, max_new_tokens=700, temperature=0.25, strict=False):
    global _error, _last_generation_error, _last_generation_text
    _last_generation_error = ""
    _last_generation_text = ""

    tokenizer, model = load_llm()
    if tokenizer is None or model is None:
        _last_generation_error = _error or "Модель не загружена."
        if strict:
            raise RuntimeError(_last_generation_error)
        return fallback if fallback is not None else {}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        import torch
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.05,
            "pad_token_id": tokenizer.eos_token_id,
        }
        generation_config = deepcopy(model.generation_config)

        # If sampling is disabled, do not pass temperature/top_p.
        # New versions of transformers warn that these flags are ignored when do_sample=False.
        if temperature and temperature > 0:
            _normalize_generation_config(generation_config, do_sample=True, temperature=temperature)
        else:
            _normalize_generation_config(generation_config, do_sample=False)

        with torch.no_grad():
            outputs = model.generate(**generation_kwargs, generation_config=generation_config)
        generated = outputs[0][inputs.input_ids.shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        _last_generation_text = text
        data = extract_json(text, fallback=_JSON_PARSE_FAILED)
        if data is _JSON_PARSE_FAILED:
            _last_generation_error = f"Модель вернула невалидный JSON: {text[:700]}"
            if strict:
                raise ValueError(_last_generation_error)
            return fallback if fallback is not None else {}
        return data
    except Exception as exc:
        _error = str(exc)
        _last_generation_error = str(exc)
        if strict:
            raise
        return fallback if fallback is not None else {}


def llm_generate_text(system_prompt, user_prompt, fallback="", max_new_tokens=700, temperature=0.25, strict=False):
    global _error, _last_generation_error, _last_generation_text
    _last_generation_error = ""
    _last_generation_text = ""

    tokenizer, model = load_llm()
    if tokenizer is None or model is None:
        _last_generation_error = _error or "Модель не загружена."
        if strict:
            raise RuntimeError(_last_generation_error)
        return fallback

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        import torch
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.05,
            "pad_token_id": tokenizer.eos_token_id,
        }
        generation_config = deepcopy(model.generation_config)
        if temperature and temperature > 0:
            _normalize_generation_config(generation_config, do_sample=True, temperature=temperature)
        else:
            _normalize_generation_config(generation_config, do_sample=False)

        with torch.no_grad():
            outputs = model.generate(**generation_kwargs, generation_config=generation_config)
        generated = outputs[0][inputs.input_ids.shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        _last_generation_text = text
        return text
    except Exception as exc:
        _error = str(exc)
        _last_generation_error = str(exc)
        if strict:
            raise
        return fallback


@lru_cache(maxsize=4096)
def classify_one(text, labels_tuple, task):
    labels = list(labels_tuple)
    fallback = {"label": labels[-1] if labels else "другое"}
    system = (
        "Ты аналитический модуль UX/UI. Верни только JSON без markdown. "
        "Нужно выбрать ровно одну метку из разрешенного списка."
    )
    user = json.dumps({
        "task": task,
        "text": text,
        "allowed_labels": labels,
        "output_format": {"label": "одна метка из allowed_labels"},
    }, ensure_ascii=False)
    data = llm_generate_json(system, user, fallback=fallback, max_new_tokens=80, temperature=0.0)
    label = str(data.get("label", "")).strip().lower()
    normalized = {x.lower(): x for x in labels}
    return normalized.get(label, fallback["label"])
