import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "personas.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS personas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    gender TEXT,
    age TEXT,
    occupation TEXT,
    device TEXT,
    context TEXT,
    avatar_path TEXT,
    device_type TEXT,
    inferred_gender TEXT,
    occupation_group TEXT
);
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona_id INTEGER,
    goal_type TEXT,
    description TEXT,
    FOREIGN KEY (persona_id) REFERENCES personas(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS behaviors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona_id INTEGER,
    pattern TEXT,
    description TEXT,
    FOREIGN KEY (persona_id) REFERENCES personas(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS contexts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona_id INTEGER,
    environment TEXT,
    device TEXT,
    time_of_day TEXT,
    FOREIGN KEY (persona_id) REFERENCES personas(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS pain_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona_id INTEGER,
    description TEXT,
    FOREIGN KEY (persona_id) REFERENCES personas(id) ON DELETE CASCADE
);
"""

REQUIRED_COLUMNS = {
    "personas": {
        "gender": "TEXT",
        "avatar_path": "TEXT",
        "device_type": "TEXT",
        "inferred_gender": "TEXT",
        "occupation_group": "TEXT",
    }
}


def _normalize_known_gender(text):
    low = (text or "").lower().strip()
    if low in {"м", "муж", "мужской"}:
        return "мужской"
    if low in {"ж", "жен", "женский"}:
        return "женский"
    return text or ""


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)
        migrate(conn)
        conn.commit()


def migrate(conn):
    for table, columns in REQUIRED_COLUMNS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, sql_type in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")


def _as_goal_dict(goal):
    if isinstance(goal, dict):
        return goal
    return {"type": "end", "description": str(goal)}


def _as_behavior_dict(behavior):
    if isinstance(behavior, dict):
        return behavior
    return {"pattern": str(behavior), "description": str(behavior)}


def add_persona(persona):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO personas
               (name, gender, age, occupation, device, context, avatar_path, device_type, inferred_gender, occupation_group)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                persona.get("name", ""),
                _normalize_known_gender(persona.get("gender", "")),
                persona.get("age", ""),
                persona.get("occupation", ""),
                persona.get("device", ""),
                persona.get("context", ""),
                persona.get("avatar_path", ""),
                persona.get("device_type", ""),
                "",
                persona.get("occupation_group", ""),
            ),
        )
        persona_id = cur.lastrowid
        for goal in persona.get("goals", []):
            goal = _as_goal_dict(goal)
            cur.execute(
                "INSERT INTO goals (persona_id, goal_type, description) VALUES (?, ?, ?)",
                (persona_id, goal.get("type", "end"), goal.get("description", "")),
            )
        for behavior in persona.get("behaviors", []):
            behavior = _as_behavior_dict(behavior)
            cur.execute(
                "INSERT INTO behaviors (persona_id, pattern, description) VALUES (?, ?, ?)",
                (persona_id, behavior.get("pattern", ""), behavior.get("description", "")),
            )
        for pain in persona.get("pain_points", []):
            cur.execute(
                "INSERT INTO pain_points (persona_id, description) VALUES (?, ?)",
                (persona_id, str(pain)),
            )
        cur.execute(
            "INSERT INTO contexts (persona_id, environment, device, time_of_day) VALUES (?, ?, ?, ?)",
            (persona_id, persona.get("context", ""), persona.get("device", ""), persona.get("time_of_day", "")),
        )
        conn.commit()
        return persona_id


def bulk_add(personas):
    with connect() as conn:
        cur = conn.cursor()
        for persona in personas:
            cur.execute(
                """INSERT INTO personas
                   (name, gender, age, occupation, device, context, avatar_path, device_type, inferred_gender, occupation_group)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    persona.get("name", ""), _normalize_known_gender(persona.get("gender", "")), persona.get("age", ""),
                    persona.get("occupation", ""), persona.get("device", ""), persona.get("context", ""),
                    persona.get("avatar_path", ""), persona.get("device_type", ""),
                    "", persona.get("occupation_group", ""),
                ),
            )
            persona_id = cur.lastrowid
            for goal in persona.get("goals", []):
                goal = _as_goal_dict(goal)
                cur.execute(
                    "INSERT INTO goals (persona_id, goal_type, description) VALUES (?, ?, ?)",
                    (persona_id, goal.get("type", "end"), goal.get("description", "")),
                )
            for behavior in persona.get("behaviors", []):
                behavior = _as_behavior_dict(behavior)
                cur.execute(
                    "INSERT INTO behaviors (persona_id, pattern, description) VALUES (?, ?, ?)",
                    (persona_id, behavior.get("pattern", ""), behavior.get("description", "")),
                )
            for pain in persona.get("pain_points", []):
                cur.execute("INSERT INTO pain_points (persona_id, description) VALUES (?, ?)", (persona_id, str(pain)))
            cur.execute(
                "INSERT INTO contexts (persona_id, environment, device, time_of_day) VALUES (?, ?, ?, ?)",
                (persona_id, persona.get("context", ""), persona.get("device", ""), persona.get("time_of_day", "")),
            )
        conn.commit()


def list_personas():
    with connect() as conn:
        return conn.execute("""
            SELECT
                p.*,
                COALESCE(GROUP_CONCAT(DISTINCT pp.description), '') AS pain_points_text
            FROM personas p
            LEFT JOIN pain_points pp ON pp.persona_id = p.id
            GROUP BY p.id
            ORDER BY p.id DESC
        """).fetchall()


def get_persona_details(persona_id):
    with connect() as conn:
        persona = conn.execute("SELECT * FROM personas WHERE id=?", (persona_id,)).fetchone()
        if persona is None:
            return None
        goals = conn.execute("SELECT goal_type, description FROM goals WHERE persona_id=?", (persona_id,)).fetchall()
        behaviors = conn.execute("SELECT pattern, description FROM behaviors WHERE persona_id=?", (persona_id,)).fetchall()
        pains = conn.execute("SELECT description FROM pain_points WHERE persona_id=?", (persona_id,)).fetchall()
        return {"persona": persona, "goals": goals, "behaviors": behaviors, "pain_points": pains}


def list_personas_for_analysis():
    with connect() as conn:
        rows = conn.execute("""
            SELECT
                p.*,
                COALESCE(GROUP_CONCAT(DISTINCT g.description), '') AS goals_text,
                COALESCE(GROUP_CONCAT(DISTINCT b.pattern || ': ' || b.description), '') AS behaviors_text,
                COALESCE(GROUP_CONCAT(DISTINCT pp.description), '') AS pain_points_text
            FROM personas p
            LEFT JOIN goals g ON g.persona_id = p.id
            LEFT JOIN behaviors b ON b.persona_id = p.id
            LEFT JOIN pain_points pp ON pp.persona_id = p.id
            GROUP BY p.id
            ORDER BY p.id DESC
        """).fetchall()
        return rows


def count_personas():
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM personas").fetchone()[0]


def clear_database():
    with connect() as conn:
        for table in ["goals", "behaviors", "contexts", "pain_points", "personas"]:
            conn.execute(f"DELETE FROM {table}")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('goals','behaviors','contexts','pain_points','personas')")
        conn.commit()


def update_analysis_fields(persona_id, gender=None, device_type=None, occupation_group=None):
    with connect() as conn:
        conn.execute(
            "UPDATE personas SET gender=?, inferred_gender='', device_type=?, occupation_group=? WHERE id=?",
            (gender or "", device_type or "", occupation_group or "", persona_id),
        )
        conn.commit()


def update_ai_fields(persona_id, inferred_gender=None, device_type=None, occupation_group=None):
    update_analysis_fields(persona_id, inferred_gender, device_type, occupation_group)
