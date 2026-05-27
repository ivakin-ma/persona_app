import json
import random
import re
from faker import Faker
from llm_engine import llm_generate_text

fake = Faker("ru_RU")

GOAL_TYPES = ["life", "end", "experience", "social"]

PHONE_BRANDS = ["iPhone", "Samsung Galaxy", "Xiaomi Redmi", "Nokia", "Google Pixel", "Honor", "Realme", "OPPO", "Vivo"]
LAPTOP_BRANDS = ["MacBook Pro", "Lenovo ThinkPad", "ASUS ZenBook", "Huawei MateBook", "HP Pavilion", "Dell XPS", "Acer Aspire"]
TABLET_BRANDS = ["iPad", "Samsung Galaxy Tab", "Xiaomi Pad", "Huawei MatePad", "Lenovo Tab"]
DESKTOP_NAMES = ["iMac", "Mac mini", "Dell OptiPlex", "HP ProDesk", "Lenovo ThinkCentre", "сборный ПК"]


def fake_device():
    """Return realistic device model only; type is classified during analysis."""
    kind = random.choices(["phone", "laptop", "tablet", "desktop"], weights=[45, 30, 15, 10])[0]
    if kind == "phone":
        brand = random.choice(PHONE_BRANDS)
        if brand == "Nokia":
            model = random.choice(["Nokia 3310", "Nokia 105", "Nokia G22", "Nokia XR21"])
        elif brand == "iPhone":
            model = random.choice(["iPhone 12", "iPhone 13", "iPhone 14", "iPhone 15", "iPhone 15 Pro"])
        else:
            model = f"{brand} {random.choice(['A15', 'S24', 'Note 12', '12 Pro', '8 Pro', 'Magic 6', 'GT Neo'])}"
        return model
    if kind == "laptop":
        return f"{random.choice(LAPTOP_BRANDS)} {random.choice(['13', '14', '15', '16', 'Air', 'Pro'])}"
    if kind == "tablet":
        return f"{random.choice(TABLET_BRANDS)} {random.choice(['10', '11', '12.9', 'Air', 'Pro', 'SE'])}"
    return random.choice(DESKTOP_NAMES)


def fake_gender_and_name():
    gender = random.choice(["мужской", "женский"])
    if gender == "мужской":
        return {"gender": gender, "name": fake.first_name_male()}
    return {"gender": gender, "name": fake.first_name_female()}


def _fallback_semantics(base):
    occupation = base["occupation"]
    device = base["device"]
    return {
        "context": f"Пользователь работает с сервисом в повседневной ситуации, используя {device}, и хочет быстро решить задачу без лишних шагов.",
        "goal_type": random.choice(GOAL_TYPES),
        "goal": f"Выполнить основную задачу в интерфейсе быстро и без ошибок с учетом своей роли: {occupation}.",
        "behavior_pattern": "прагматичный пользователь",
        "behavior": "Сначала просматривает основные варианты, затем выбирает самый понятный путь и избегает лишних настроек.",
        "pain": "Сложность интерфейса и неочевидные действия мешают быстро завершить задачу.",
    }


def _make_base_persona():
    person = fake_gender_and_name()
    device = fake_device()
    return {
        "name": person["name"],
        "gender": person["gender"],
        "age": str(random.randint(18, 70)),
        "occupation": fake.job(),
        "device": device,
        "avatar_path": "",
    }


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(_as_text(item) for item in value if _as_text(item))
    if isinstance(value, dict):
        for key in ("description", "text", "value", "content"):
            if key in value:
                return _as_text(value[key])
        return "; ".join(_as_text(item) for item in value.values() if _as_text(item))
    return str(value).strip()


def _first_text(item, keys):
    for key in keys:
        text = _as_text(item.get(key))
        if text:
            return text
    return ""


def _extract_items(data, expected_count):
    if isinstance(data, list):
        return [item for item in data if isinstance(item, (dict, list, tuple))]

    if not isinstance(data, dict):
        return []

    for key in ("items", "data", "output", "result", "rows", "records", "personas", "users", "results", "descriptions", "archetypes"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, (dict, list, tuple))]
        if isinstance(value, dict):
            nested = _extract_items(value, expected_count)
            if nested:
                return nested

    numeric_items = []
    for key, value in sorted(data.items(), key=lambda pair: _parse_index(pair[0]) if _parse_index(pair[0]) is not None else 10**9):
        if _parse_index(key) is not None and isinstance(value, (dict, list, tuple)):
            numeric_items.append(value)
    if numeric_items:
        return numeric_items

    for value in data.values():
        if isinstance(value, list) and len(value) >= expected_count:
            candidates = [item for item in value if isinstance(item, (dict, list, tuple))]
            if candidates:
                return candidates

    if expected_count == 1 and any(key in data for key in ("context", "goal", "behavior", "pain")):
        return [data]

    return []


def _parse_index(value):
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def _items_by_index(items, expected_count):
    by_index = {}
    used = set()
    for pos, item in enumerate(items):
        if isinstance(item, dict):
            index = _parse_index(item.get("index", item.get("id", item.get("number"))))
        elif isinstance(item, (list, tuple)) and item:
            index = _parse_index(item[0])
        else:
            index = None
        if index is not None and 0 <= index < expected_count:
            by_index[index] = item
            used.add(pos)

    for index in range(expected_count):
        if index in by_index:
            continue
        for pos, item in enumerate(items):
            if pos not in used:
                by_index[index] = item
                used.add(pos)
                break
    return by_index


def _normalize_semantics(item):
    if isinstance(item, (list, tuple)):
        values = list(item)
        if len(values) >= 7 and _parse_index(values[0]) is not None:
            values = values[1:]
        values = values + [""] * 6
        return {
            "context": _as_text(values[0]),
            "goal_type": _as_text(values[1]),
            "goal": _as_text(values[2]),
            "behavior_pattern": _as_text(values[3]),
            "behavior": _as_text(values[4]),
            "pain": _as_text(values[5]),
        }

    return {
        "context": _first_text(item, ("context", "context_description", "usage_context", "situation", "scenario", "environment", "контекст")),
        "goal_type": _first_text(item, ("goal_type", "goalType", "type", "тип_цели")),
        "goal": _first_text(item, ("goal", "goal_description", "user_goal", "цель")),
        "behavior_pattern": _first_text(item, ("behavior_pattern", "pattern", "behavior_type", "паттерн")),
        "behavior": _first_text(item, ("behavior", "behavior_description", "user_behavior", "поведение")),
        "pain": _first_text(item, ("pain", "pain_point", "pain_points", "barrier", "problem", "боль", "барьер")),
    }


FIELD_ALIASES = {
    "CONTEXT": "context",
    "КОНТЕКСТ": "context",
    "GOAL_TYPE": "goal_type",
    "GOALTYPE": "goal_type",
    "ТИП_ЦЕЛИ": "goal_type",
    "ТИПЦЕЛИ": "goal_type",
    "GOAL": "goal",
    "ЦЕЛЬ": "goal",
    "BEHAVIOR_PATTERN": "behavior_pattern",
    "BEHAVIORPATTERN": "behavior_pattern",
    "ПАТТЕРН": "behavior_pattern",
    "BEHAVIOR": "behavior",
    "ПОВЕДЕНИЕ": "behavior",
    "PAIN": "pain",
    "БОЛЬ": "pain",
    "BARRIER": "pain",
    "БАРЬЕР": "pain",
}


def _field_name(raw):
    key = re.sub(r"[^A-Za-zА-Яа-я_]", "", raw or "").upper()
    return FIELD_ALIASES.get(key)


def _append_record(records, record):
    if record and any(_as_text(record.get(key)) for key in ("context", "goal", "behavior", "pain")):
        records.append(record)


def _parse_text_items(text, expected_count):
    records = []
    current = None
    last_field = None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line in {"```", "```text"}:
            continue

        user_match = re.match(r"^@{0,2}USER\s+(\d+)", line, flags=re.IGNORECASE)
        if user_match:
            _append_record(records, current)
            current = {"index": int(user_match.group(1))}
            last_field = None
            continue

        if line.upper().startswith("@@END"):
            _append_record(records, current)
            current = None
            last_field = None
            continue

        if ":" in line:
            raw_key, value = line.split(":", 1)
            field = _field_name(raw_key)
            if field:
                if current is None:
                    current = {}
                elif field == "context" and current.get("context"):
                    _append_record(records, current)
                    current = {}
                current[field] = value.strip()
                last_field = field
                continue

        if current is not None and last_field:
            current[last_field] = f"{current.get(last_field, '')} {line}".strip()

    _append_record(records, current)
    return records[:expected_count]


def enrich_batch_with_llm(base_personas):
    """Generate meaningful context, goals, pain points and behavior in one LLM call per batch."""
    system = (
        "Ты генератор UX-архетипов. Не пиши JSON и markdown. "
        "Для каждого пользователя верни текстовый блок строго в формате:\n"
        "@@USER <index>\n"
        "CONTEXT: ...\n"
        "GOAL_TYPE: life|end|experience|social\n"
        "GOAL: ...\n"
        "BEHAVIOR_PATTERN: ...\n"
        "BEHAVIOR: ...\n"
        "PAIN: ...\n"
        "@@END\n"
        "Не меняй исходные имя, пол, возраст, профессию и устройство. "
        "Не выдумывай браузер и ОС. Сценарии должны быть уникальными."
    )
    user = json.dumps({
        "rules": "не повторяй context/pain; не начинай context с 'Пользователь работает с сервисом'; GOAL_TYPE только life/end/experience/social",
        "users": [
            [i, p["name"], p["gender"], p["age"], p["occupation"], p["device"]]
            for i, p in enumerate(base_personas)
        ]
    }, ensure_ascii=False)
    text = llm_generate_text(system, user, fallback="", max_new_tokens=800, temperature=0.35, strict=True)
    items = _parse_text_items(text, len(base_personas))
    if not isinstance(items, list) or not items:
        raise ValueError(f"Qwen не вернул текстовые блоки для генерации описаний. Ответ: {text[:500]}")

    by_index = _items_by_index(items, len(base_personas))

    result = []
    for i, base in enumerate(base_personas):
        item = by_index.get(i)
        if not item:
            raise ValueError(f"Qwen не вернул описание для записи с index={i}.")

        semantic = _normalize_semantics(item)
        fallback = _fallback_semantics(base)
        for key in ("context", "goal_type", "goal", "behavior_pattern", "behavior", "pain"):
            if not str(semantic.get(key, "")).strip():
                semantic[key] = fallback[key]

        result.append({
            **base,
            "context": str(semantic.get("context", "")),
            "goals": [{
                "type": semantic.get("goal_type", "end") if semantic.get("goal_type") in GOAL_TYPES else "end",
                "description": str(semantic.get("goal", "")),
            }],
            "behaviors": [{
                "pattern": str(semantic.get("behavior_pattern", "")),
                "description": str(semantic.get("behavior", "")),
            }],
            "pain_points": [str(semantic.get("pain", ""))],
        })
    return result


def generate_persona(use_llm=True):
    return generate_many(1, use_llm=use_llm)[0]


def generate_many(count, use_llm=True, batch_size=4):
    bases = [_make_base_persona() for _ in range(count)]
    if not use_llm:
        return [{
            **base,
            "context": _fallback_semantics(base)["context"],
            "goals": [{"type": _fallback_semantics(base)["goal_type"], "description": _fallback_semantics(base)["goal"]}],
            "behaviors": [{"pattern": _fallback_semantics(base)["behavior_pattern"], "description": _fallback_semantics(base)["behavior"]}],
            "pain_points": [_fallback_semantics(base)["pain"]],
        } for base in bases]

    personas = []
    for start in range(0, count, batch_size):
        personas.extend(enrich_batch_with_llm(bases[start:start + batch_size]))
    return personas
