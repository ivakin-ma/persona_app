from collections import Counter
from llm_engine import (
    classify_one,
    get_device_info,
    get_model_name,
    is_model_loaded,
    get_model_error,
    load_llm,
    llm_generate_json,
)

MODEL_NAME = get_model_name()

DESIGN_ADVICE = [
    "Визуальная иерархия: выделите главное действие на каждом экране, а второстепенные элементы сделайте менее заметными.",
    "Адаптивность: проверьте ключевые сценарии на телефоне, планшете и десктопе, чтобы формы, таблицы и кнопки не теряли читаемость.",
    "Доступность: обеспечьте достаточный контраст, понятные подписи полей и состояния фокуса для клавиатурной навигации.",
    "Обратная связь: показывайте понятные статусы загрузки, подтверждения успешных действий и конкретные подсказки при ошибках.",
]


def _normalize_gender_text(text):
    low = (text or "").lower().strip()
    if not low:
        return ""
    if low in {"м", "муж", "мужской"}:
        return "мужской"
    if low in {"ж", "жен", "женский"}:
        return "женский"
    return ""


def infer_gender_ai(name="", entered_gender=""):
    normalized = _normalize_gender_text(entered_gender)
    if normalized:
        return normalized
    labels = ("мужской", "женский", "не определено")
    text = f"Имя пользователя: {name}. Определи пол по русскому имени. Если имя неоднозначное — не определено."
    predicted = classify_one(text, labels, "gender_by_name")
    if predicted in {"мужской", "женский"}:
        return predicted

    # Last deterministic fallback for common Russian names.
    low_name = (name or "").lower().strip()
    female_names = ("мария", "анна", "елена", "ольга", "наталья", "ирина", "светлана", "екатерина", "дарья", "алёна", "алена", "юлия", "татьяна")
    male_names = ("олег", "иван", "дмитрий", "алексей", "сергей", "андрей", "никита", "максим", "павел", "михаил", "артём", "артем", "александр")
    if low_name in female_names or low_name.endswith(("а", "я")):
        return "женский"
    if low_name in male_names or low_name.endswith(("й", "н", "р", "м", "в", "г", "д", "т")):
        return "мужской"
    return "не определено"


def infer_device_type_ai(device):
    labels = ("телефон", "ноутбук", "планшет", "стационарный компьютер", "другое")
    low = (device or "").lower()
    # Fast obvious fallback before LLM. This fixes Nokia 3310 and similar model-only strings.
    phone_hints = ("iphone", "android", "samsung", "galaxy", "nokia", "3310", "redmi", "xiaomi", "honor", "pixel", "realme", "oppo", "vivo", "телефон", "смартфон")
    tablet_hints = ("ipad", "планшет", "tablet", "galaxy tab", "matepad", "xiaomi pad")
    laptop_hints = ("macbook", "ноутбук", "laptop", "thinkpad", "zenbook", "matebook", "vivobook", "pavilion", "dell xps", "acer aspire")
    desktop_hints = ("пк", "pc", "desktop", "стационар", "imac", "mac mini", "optiplex", "prodesk", "thinkcentre", "сборный пк")
    if any(x in low for x in tablet_hints):
        return "планшет"
    if any(x in low for x in phone_hints):
        return "телефон"
    if any(x in low for x in laptop_hints):
        return "ноутбук"
    if any(x in low for x in desktop_hints):
        return "стационарный компьютер"
    return classify_one(f"Модель устройства: {device}. Определи только тип устройства.", labels, "device_type")


OCCUPATION_LABELS = (
    "финансы и учет", "продажи и маркетинг", "образование", "IT и администрирование",
    "дизайн и творчество", "транспорт и доставка", "медицина", "право",
    "студенты", "предпринимательство", "производство и ремонт", "сфера услуг", "другое"
)


def infer_occupation_group_ai(occupation):
    return classify_one(
        f"Профессия пользователя: {occupation}. Объедини похожие профессии в одну сферу деятельности.",
        OCCUPATION_LABELS,
        "occupation_group",
    )


def classify_goal_group(text):
    labels = ("покупка или заказ", "оплата или финансы", "поиск информации", "обучение", "общение", "развлечение", "работа с документами", "другое")
    return classify_one(f"Цель пользователя: {text}. Определи категорию цели.", labels, "goal_group")


def classify_pain_group(text):
    labels = ("сложность интерфейса", "нехватка времени", "недоверие", "технические проблемы", "отвлекающая среда", "недостаток информации", "другое")
    return classify_one(f"Боль пользователя: {text}. Определи категорию проблемы.", labels, "pain_group")


def classify_behavior_group(text):
    labels = ("осторожный пользователь", "быстрый пользователь", "сравнивает варианты", "социально ориентированный", "исследователь", "новичок", "другое")
    return classify_one(f"Поведение пользователя: {text}. Определи тип поведения.", labels, "behavior_group")


def percent(counter, total):
    if total == 0:
        return []
    return [(k or "не указано", v, round(v * 100 / total, 1)) for k, v in counter.most_common()]


def _top(stats):
    return stats[0] if stats else None


def make_data_advice(stats_payload):
    advice = []

    top_device = _top(stats_payload.get("devices", []))
    if top_device:
        name, _count, pct = top_device
        if name == "телефон":
            advice.append(f"Устройства: {pct}% пользователей работают с телефона. Проектируйте основной сценарий mobile-first: короткие формы, крупные зоны нажатия и минимум горизонтальных таблиц.")
        elif name == "ноутбук":
            advice.append(f"Устройства: {pct}% пользователей работают с ноутбука. Можно использовать более плотные рабочие экраны, фильтры и сравнение данных в несколько колонок.")
        elif name == "планшет":
            advice.append(f"Устройства: {pct}% пользователей работают с планшета. Держите интерактивные элементы крупными и проверяйте интерфейс в портретной и альбомной ориентации.")
        else:
            advice.append(f"Устройства: чаще всего встречается '{name}' ({pct}%). Проверьте ключевые сценарии именно под этот класс устройств.")

    top_occupation = _top(stats_payload.get("occupations", []))
    if top_occupation:
        name, _count, pct = top_occupation
        advice.append(f"Профессии: ведущая группа — '{name}' ({pct}%). Термины, порядок полей и быстрые действия стоит адаптировать под эту рабочую роль.")

    top_goal = _top(stats_payload.get("goals", []))
    if top_goal:
        name, _count, pct = top_goal
        advice.append(f"Цели: основная категория — '{name}' ({pct}%). Вынесите этот сценарий в самый короткий путь и сделайте его доступным с первого экрана.")

    top_pain = _top(stats_payload.get("pains", []))
    if top_pain:
        name, _count, pct = top_pain
        advice.append(f"Барьеры: чаще всего проявляется '{name}' ({pct}%). Добавьте подсказки, предотвращение ошибок и явное восстановление после неудачного действия.")

    top_behavior = _top(stats_payload.get("behaviors", []))
    if top_behavior:
        name, _count, pct = top_behavior
        advice.append(f"Поведение: доминирует тип '{name}' ({pct}%). Настройте навигацию и плотность информации под этот стиль принятия решений.")

    return advice


def make_llm_advice(stats_payload):
    fallback = ["Сократите количество шагов в основных сценариях и делайте ключевые действия заметными."]
    data = llm_generate_json(
        "Ты UX/UI аналитик. Верни только JSON без markdown.",
        {
            "task": "Сформулируй 5-8 конкретных UX/UI рекомендаций строго по переданной статистике. В каждой рекомендации опирайся на одну из категорий статистики и по возможности называй категорию.",
            "statistics": stats_payload,
            "output_format": {"advice": ["короткая рекомендация"]},
        }.__repr__(),
        fallback={"advice": fallback},
        max_new_tokens=700,
        temperature=0.35,
    )
    advice = data.get("advice", fallback) if isinstance(data, dict) else fallback
    dynamic_advice = [str(x) for x in advice if str(x).strip()] or fallback
    return make_data_advice(stats_payload) + dynamic_advice + DESIGN_ADVICE


def analyze_personas(rows, save_callback=None):
    # Load model once at the start, so status in GUI is honest.
    load_llm()

    enriched = []
    pain_groups = []
    behavior_groups = []
    goal_groups = []

    for row in rows:
        gender = infer_gender_ai(row["name"], row["gender"])
        device_type = infer_device_type_ai(row["device"])
        occupation_group = infer_occupation_group_ai(row["occupation"])
        goals_text = row["goals_text"] or ""
        pains_text = row["pain_points_text"] or ""
        behaviors_text = row["behaviors_text"] or ""

        if goals_text:
            goal_groups.append(classify_goal_group(goals_text))
        if pains_text:
            pain_groups.append(classify_pain_group(pains_text))
        if behaviors_text:
            behavior_groups.append(classify_behavior_group(behaviors_text))

        if save_callback:
            save_callback(row["id"], gender, device_type, occupation_group)

        enriched.append({
            "gender": gender,
            "device_type": device_type,
            "occupation_group": occupation_group,
        })

    total = len(enriched)
    gender_stats = percent(Counter(x["gender"] for x in enriched), total)
    device_stats = percent(Counter(x["device_type"] for x in enriched), total)
    occupation_stats = percent(Counter(x["occupation_group"] for x in enriched), total)
    goal_stats = percent(Counter(goal_groups), len(goal_groups))
    pain_stats = percent(Counter(pain_groups), len(pain_groups))
    behavior_stats = percent(Counter(behavior_groups), len(behavior_groups))

    stats_payload = {
        "total": total,
        "gender": gender_stats[:5],
        "devices": device_stats[:5],
        "occupations": occupation_stats[:8],
        "goals": goal_stats[:8],
        "pains": pain_stats[:8],
        "behaviors": behavior_stats[:8],
    }
    advice = make_llm_advice(stats_payload)

    return {
        "total": total,
        "gender_stats": gender_stats,
        "device_stats": device_stats,
        "occupation_stats": occupation_stats,
        "goal_stats": goal_stats,
        "pain_stats": pain_stats,
        "behavior_stats": behavior_stats,
        "advice": advice,
        "model": MODEL_NAME,
        "model_loaded": is_model_loaded(),
        "device": get_device_info(),
        "model_error": get_model_error(),
    }
