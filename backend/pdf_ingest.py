# ============================================================
# pdf_ingest.py — PDF health record  ->  JSON  ->  SQLite
#
#   python pdf_ingest.py data/pdfs/patient_001.pdf              # ingest one PDF
#   python pdf_ingest.py data/pdfs                              # ingest every PDF in a folder
#   python pdf_ingest.py report.pdf --patient-id patient_002    # choose the patient id
#   python pdf_ingest.py report.pdf --dry-run                   # extract + show, don't save to DB
#   python pdf_ingest.py --json data/patients/patient_001.json  # import a ready-made JSON (no AI call)
#   python pdf_ingest.py --list                                 # patients stored in the database
#   python pdf_ingest.py --show patient_001                     # print one stored record
#
# Steps: 1) read text from the PDF  2) Groq turns it into structured JSON
#        3) JSON is saved to data/extracted/ so you can REVIEW it
#        4) the record is stored in SQLite (data/health.db), which the AI reads
# ============================================================
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

import db

load_dotenv(override=True)

EXTRACTED_DIR = Path(__file__).parent / "data" / "extracted"
MAX_CHARS = 90_000  # safety limit on text sent to the model

# The exact shape we want back. Empty values = "not in the document".
TEMPLATE = {
    "profile": {
        "name": "", "age": None, "sex": "", "blood_group": "", "height_cm": "", "weight_kg": "",
        "language": "", "primary_doctor": "",
        "emergency_contact": {"name": "", "relation": "", "phone": ""},
    },
    "allergies": [{"substance": "", "reaction": "", "severity": ""}],
    "conditions": [{"name": "", "since": "", "status": ""}],
    "surgeries": [{"procedure": "", "date": "", "hospital": "", "notes": ""}],
    "medications": [{"name": "", "dose": "", "frequency": "", "purpose": ""}],
    "lab_results": [{"date": "", "test": "", "value": "", "reference": "", "note": ""}],
    "vitals_baseline": {"resting_heart_rate_bpm": "", "usual_blood_pressure": "",
                        "usual_spo2_percent": "", "note": ""},
    "visits": [{"date": "", "doctor": "", "reason": "", "summary": ""}],
    "appointments": [{"date": "", "time": "", "doctor": "", "reason": "", "location": ""}],
    "care_instructions": [""],
    "doctor_notes": [{"date": "", "note": ""}],
}

EXTRACT_PROMPT = f"""You convert a patient's medical record text (extracted from a PDF) into JSON.

Rules:
- Output ONE JSON object only. No explanation, no markdown fences.
- Use exactly this structure (keys and nesting). Lists may hold any number of items, or be empty:
{json.dumps(TEMPLATE, indent=2)}
- Copy information ONLY from the document. NEVER guess or invent anything. If something is not stated, use "" (or null for age) or an empty list.
- Dates as YYYY-MM-DD when the full date is known; times as HH:MM (24 hour).
- Keep medicine names, doses, frequencies and lab values exactly as written.
- Put future-dated bookings in "appointments"; past consultations in "visits".
- Today's date is {date.today().isoformat()}.
"""


# ── 1. PDF -> text ────────────────────────────────────────────────────────────
def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError(f"{path.name} is password protected — remove the password first")
    pages = []
    for i, page in enumerate(reader.pages, 1):
        t = (page.extract_text() or "").strip()
        if t:
            pages.append(f"--- page {i} ---\n{t}")
    text = "\n\n".join(pages)
    if len(text) < 50:
        raise ValueError(
            f"No readable text in {path.name}. It is probably a scanned image PDF — "
            "it needs OCR first (e.g. 'ocrmypdf in.pdf out.pdf'), or export a text PDF.")
    return text


# ── 2. text -> JSON (Groq) ────────────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b > a:
            return json.loads(s[a:b + 1])
        raise


def extract_record_with_llm(text: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    client = OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    model = os.getenv("EXTRACT_MODEL", os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))

    if len(text) > MAX_CHARS:
        print(f"  ! Document is very long ({len(text)} chars); only the first {MAX_CHARS} are used")
        text = text[:MAX_CHARS]

    messages = [{"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": f"MEDICAL RECORD TEXT:\n\n{text}"}]
    last_err = None
    for attempt in (1, 2):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=8000,
                response_format={"type": "json_object"},
                extra_body={"reasoning_effort": "low"},
            )
            return _parse_json(resp.choices[0].message.content)
        except json.JSONDecodeError as e:
            last_err = e
            messages.append({"role": "user", "content": "That was not valid JSON. Reply with the JSON object only."})
    raise RuntimeError(f"Model did not return valid JSON: {last_err}")


# ── 3/4. save + store ─────────────────────────────────────────────────────────
def make_patient_id(path: Path, override: str | None) -> str:
    if override:
        return override
    return re.sub(r"[^a-z0-9_]+", "_", path.stem.lower()).strip("_") or "patient_001"


def summarize(rec: dict) -> str:
    p = rec.get("profile", {})
    counts = ", ".join(f"{k}={len(rec.get(k) or [])}" for k in db.LIST_TABLES)
    return f"{p.get('name')} (age {p.get('age')}) -> {counts}"


def ingest_pdf(path: Path, patient_id: str | None, dry_run: bool, force: bool) -> bool:
    print(f"\n▶ {path.name}")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if not dry_run and not force:
        existing = db.document_already_ingested(sha)
        if existing:
            print(f"  = Already ingested as '{existing}' (same file). Use --force to redo.")
            return True

    try:
        text = extract_pdf_text(path)
        print(f"  1) Read {len(text)} characters of text")
        rec = extract_record_with_llm(text)
        rec["patient_id"] = make_patient_id(path, patient_id)  # we decide the id, not the AI
        db.validate_record(rec)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    out = EXTRACTED_DIR / f"{rec['patient_id']}.json"
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  2) AI extracted: {summarize(rec)}")
    print(f"  3) JSON saved for review: {out}")

    if dry_run:
        print("  (dry run — not stored in the database)")
        return True
    db.save_patient_record(rec, filename=path.name, sha256=sha)
    print(f"  4) Stored in SQLite as patient '{rec['patient_id']}'  ✓")
    return True


def ingest_json(path: Path) -> bool:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        db.save_patient_record(rec, filename=path.name)
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    print(f"✓ Imported {path.name}: {summarize(rec)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="PDF health record -> JSON -> SQLite")
    ap.add_argument("paths", nargs="*", help="PDF file(s) or folder(s)")
    ap.add_argument("--patient-id", help="patient id to store under (default: PDF file name)")
    ap.add_argument("--dry-run", action="store_true", help="extract and save JSON only; do not write to the database")
    ap.add_argument("--force", action="store_true", help="re-ingest even if this exact PDF was already stored")
    ap.add_argument("--json", metavar="FILE", help="import a ready-made JSON record instead of a PDF")
    ap.add_argument("--list", action="store_true", help="list patients in the database")
    ap.add_argument("--show", metavar="PATIENT_ID", help="print a stored record as JSON")
    a = ap.parse_args()

    if a.list:
        ids = db.list_patient_ids()
        print("\n".join(ids) if ids else "(database is empty)")
        return 0
    if a.show:
        rec = db.load_patient_record(a.show)
        print(json.dumps(rec, indent=2, ensure_ascii=False) if rec else f"No patient '{a.show}'")
        return 0 if rec else 1
    if a.json:
        return 0 if ingest_json(Path(a.json)) else 1
    if not a.paths:
        ap.print_help()
        return 1

    pdfs: list[Path] = []
    for p in map(Path, a.paths):
        if p.is_dir():
            pdfs += sorted(p.glob("*.pdf"))
        elif p.exists():
            pdfs.append(p)
        else:
            print(f"✗ Not found: {p}")
    if not pdfs:
        print("No PDF files to process")
        return 1
    if a.patient_id and len(pdfs) > 1:
        print("--patient-id can only be used with a single PDF")
        return 1

    ok = [ingest_pdf(p, a.patient_id, a.dry_run, a.force) for p in pdfs]
    print(f"\nDone: {sum(ok)}/{len(ok)} succeeded")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())