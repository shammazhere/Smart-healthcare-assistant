# ============================================================
# db.py — SQLite storage for patient health records
#   This database is the PRIMARY source of truth for the AI.
#   Records arrive via pdf_ingest.py (PDF -> JSON -> here).
# ============================================================
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.getenv("HEALTH_DB_PATH", Path(__file__).parent / "data" / "health.db"))

# Tables that hold a list of rows per patient: table -> columns
LIST_TABLES: dict[str, list[str]] = {
    "allergies":         ["substance", "reaction", "severity"],
    "conditions":        ["name", "since", "status"],
    "surgeries":         ["procedure", "date", "hospital", "notes"],
    "medications":       ["name", "dose", "frequency", "purpose"],
    "lab_results":       ["date", "test", "value", "reference", "note"],
    "visits":            ["date", "doctor", "reason", "summary"],
    "appointments":      ["date", "time", "doctor", "reason", "location"],
    "care_instructions": ["instruction"],
    "doctor_notes":      ["date", "note"],
}


def _schema_sql() -> str:
    parts = ["""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        name TEXT, age INTEGER, sex TEXT, blood_group TEXT,
        height_cm TEXT, weight_kg TEXT, language TEXT, primary_doctor TEXT,
        emergency_name TEXT, emergency_relation TEXT, emergency_phone TEXT,
        vitals_hr TEXT, vitals_bp TEXT, vitals_spo2 TEXT, vitals_note TEXT,
        updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS source_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT, filename TEXT, sha256 TEXT, ingested_at TEXT
    );"""]
    for table, cols in LIST_TABLES.items():
        col_sql = ", ".join(f'"{c}" TEXT' for c in cols)
        parts.append(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
            {col_sql}
        );
        CREATE INDEX IF NOT EXISTS idx_{table}_pid ON {table}(patient_id);""")
    return "\n".join(parts)


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_schema_sql())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _s(v) -> str:
    return "" if v is None else str(v).strip()


def _age(v):
    try:
        return int(float(str(v).strip().split()[0]))
    except Exception:
        return None


def validate_record(rec: dict) -> dict:
    """Make sure a record has the shape we store. Raises ValueError if unusable."""
    if not isinstance(rec, dict):
        raise ValueError("Record must be a JSON object")
    pid = _s(rec.get("patient_id"))
    if not pid:
        raise ValueError("Record has no patient_id")
    profile = rec.get("profile") or {}
    if not _s(profile.get("name")):
        raise ValueError("Record has no patient name — is this the right PDF?")
    return rec


def save_patient_record(rec: dict, filename: str = "", sha256: str = "") -> None:
    """Insert or fully replace one patient's record (transactional)."""
    validate_record(rec)
    pid = _s(rec["patient_id"])
    p = rec.get("profile") or {}
    ec = p.get("emergency_contact") or {}
    vb = rec.get("vitals_baseline") or {}

    with connect() as conn:
        conn.execute("DELETE FROM patients WHERE patient_id = ?", (pid,))  # cascades to child rows
        conn.execute(
            """INSERT INTO patients (patient_id, name, age, sex, blood_group, height_cm, weight_kg, language,
                   primary_doctor, emergency_name, emergency_relation, emergency_phone,
                   vitals_hr, vitals_bp, vitals_spo2, vitals_note, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pid, _s(p.get("name")), _age(p.get("age")), _s(p.get("sex")), _s(p.get("blood_group")),
             _s(p.get("height_cm")), _s(p.get("weight_kg")), _s(p.get("language")), _s(p.get("primary_doctor")),
             _s(ec.get("name")), _s(ec.get("relation")), _s(ec.get("phone")),
             _s(vb.get("resting_heart_rate_bpm")), _s(vb.get("usual_blood_pressure")),
             _s(vb.get("usual_spo2_percent")), _s(vb.get("note")),
             datetime.now().isoformat(timespec="seconds")),
        )
        for table, cols in LIST_TABLES.items():
            items = rec.get(table) or []
            if not isinstance(items, list):
                continue
            for item in items:
                if table == "care_instructions":
                    row = {"instruction": item if isinstance(item, str) else _s(item.get("instruction"))}
                elif isinstance(item, dict):
                    row = item
                else:
                    continue
                values = [_s(row.get(c)) for c in cols]
                if not any(values):
                    continue
                col_sql = ", ".join(f'"{c}"' for c in cols)
                marks = ", ".join("?" for _ in cols)
                conn.execute(f"INSERT INTO {table} (patient_id, {col_sql}) VALUES (?, {marks})", (pid, *values))
        if filename:
            conn.execute(
                "INSERT INTO source_documents (patient_id, filename, sha256, ingested_at) VALUES (?,?,?,?)",
                (pid, filename, sha256, datetime.now().isoformat(timespec="seconds")))


def load_patient_record(pid: str) -> dict | None:
    """Rebuild a patient record (same shape as the JSON) from the database."""
    with connect() as conn:
        r = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (pid,)).fetchone()
        if r is None:
            return None
        rec: dict = {
            "patient_id": r["patient_id"],
            "profile": {
                "name": r["name"], "age": r["age"], "sex": r["sex"], "blood_group": r["blood_group"],
                "height_cm": r["height_cm"], "weight_kg": r["weight_kg"], "language": r["language"],
                "primary_doctor": r["primary_doctor"],
                "emergency_contact": {"name": r["emergency_name"], "relation": r["emergency_relation"],
                                      "phone": r["emergency_phone"]},
            },
            "vitals_baseline": {
                "resting_heart_rate_bpm": r["vitals_hr"], "usual_blood_pressure": r["vitals_bp"],
                "usual_spo2_percent": r["vitals_spo2"], "note": r["vitals_note"],
            },
        }
        for table, cols in LIST_TABLES.items():
            rows = conn.execute(f"SELECT * FROM {table} WHERE patient_id = ? ORDER BY id", (pid,)).fetchall()
            if table == "care_instructions":
                rec[table] = [x["instruction"] for x in rows]
            else:
                rec[table] = [{c: x[c] for c in cols} for x in rows]
        return rec


def list_patient_ids() -> list[str]:
    with connect() as conn:
        return [r["patient_id"] for r in conn.execute("SELECT patient_id FROM patients ORDER BY patient_id")]


def load_all_records() -> dict[str, dict]:
    return {pid: load_patient_record(pid) for pid in list_patient_ids()}


def document_already_ingested(sha256: str) -> str | None:
    with connect() as conn:
        r = conn.execute("SELECT patient_id FROM source_documents WHERE sha256 = ?", (sha256,)).fetchone()
        return r["patient_id"] if r else None


def db_mtime() -> float:
    try:
        return DB_PATH.stat().st_mtime
    except FileNotFoundError:
        return 0.0