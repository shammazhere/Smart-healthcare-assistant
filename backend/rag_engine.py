# ============================================================
# rag_engine.py — Retrieval over patient health records
#   * Loads patient records from the SQLite database (db.py)
#   * Turns each record into a small text "chunk"
#   * Retrieves the most relevant chunks for a question (BM25)
#   * No heavy dependencies: pure Python, works offline
# ============================================================
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date

import db

STOPWORDS = set("""
a an the is are was were be been am do does did have has had i me my mine you your it its of to in on at for
and or but with about what which who whom when where how can could should would will shall please tell give
any some there this that these those from as by if so than then them they he she his her we our us
""".split())

# Spoken questions rarely use the same words as the record, so we expand a few.
SYNONYMS = {
    "medicine": ["medication", "tablet", "dose", "drug"],
    "medicines": ["medication", "tablet", "dose", "drug"],
    "pill": ["medication", "tablet"],
    "pills": ["medication", "tablet"],
    "tablet": ["medication"],
    "tablets": ["medication"],
    "sugar": ["diabetes", "glucose", "hba1c"],
    "diabetes": ["glucose", "hba1c", "metformin"],
    "bp": ["hypertension", "blood", "pressure", "amlodipine"],
    "pressure": ["hypertension", "amlodipine"],
    "cholesterol": ["ldl", "atorvastatin", "lipid"],
    "kidney": ["creatinine", "ckd", "nephrology"],
    "allergic": ["allergy", "allergies"],
    "allergy": ["allergies", "reaction"],
    "operation": ["surgery", "surgeries", "procedure"],
    "surgery": ["procedure", "operation"],
    "knee": ["orthopedics", "replacement"],
    "doctor": ["appointment", "visit"],
    "appointment": ["doctor", "date", "visit"],
    "next": ["upcoming", "appointment"],
    "test": ["lab", "result"],
    "tests": ["lab", "result"],
    "report": ["lab", "result", "visit"],
    "reports": ["lab", "result", "visit"],
    "diet": ["food", "care", "instructions"],
    "eat": ["diet", "food", "care"],
    "food": ["diet", "care"],
    "family": ["emergency", "contact"],
    "wife": ["emergency", "contact"],
    "history": ["conditions", "surgeries", "visits"],
    "heart": ["pulse", "vitals", "aspirin"],
    "oxygen": ["spo2", "vitals"],
    "swelling": ["ankle", "visit"],
    "exercise": ["walk", "care"],
    "walk": ["exercise", "care"],
    "morning": ["breakfast"],
    "night": ["bedtime", "dinner"],
    "evening": ["dinner"],
    "afternoon": ["lunch"],
}


def _stem(w: str) -> str:
    for suf in ("ies", "es", "s", "ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


def expand_query(question: str) -> list[str]:
    tokens = []
    for w in re.findall(r"[a-z0-9]+", question.lower()):
        if w in STOPWORDS:
            continue
        tokens.append(_stem(w))
        for syn in SYNONYMS.get(w, []):
            tokens.append(_stem(syn))
    return tokens


@dataclass
class Chunk:
    patient_id: str
    section: str
    text: str
    date: str = ""


# ── Record → chunks ───────────────────────────────────────────────────────────
def _build_chunks(rec: dict) -> list[Chunk]:
    pid = rec["patient_id"]
    out: list[Chunk] = []

    p = rec.get("profile", {})
    ec = p.get("emergency_contact", {})
    out.append(Chunk(pid, "profile",
        f"Patient profile: {p.get('name')}, {p.get('age')} years old, {p.get('sex')}, blood group {p.get('blood_group')}, "
        f"weight {p.get('weight_kg')} kg. Primary doctor: {p.get('primary_doctor')}. "
        f"Emergency contact: {ec.get('name')} ({ec.get('relation')}), phone {ec.get('phone')}."))

    for a in rec.get("allergies", []):
        out.append(Chunk(pid, "allergies",
            f"Allergy: {a['substance']}. Reaction: {a['reaction']}. Severity: {a['severity']}."))

    for c in rec.get("conditions", []):
        out.append(Chunk(pid, "conditions",
            f"Medical condition: {c['name']}, since {c['since']}. Status: {c['status']}."))

    for s in rec.get("surgeries", []):
        out.append(Chunk(pid, "surgeries",
            f"Surgery: {s['procedure']} on {s['date']} at {s['hospital']}. {s['notes']}", s["date"]))

    for m in rec.get("medications", []):
        out.append(Chunk(pid, "medications",
            f"Medication: {m['name']} {m['dose']}, {m['frequency']}. Purpose: {m['purpose']}."))

    for l in rec.get("lab_results", []):
        out.append(Chunk(pid, "lab_results",
            f"Lab result on {l['date']}: {l['test']} = {l['value']} (reference {l['reference']}). {l['note']}.", l["date"]))

    vb = rec.get("vitals_baseline")
    if vb:
        out.append(Chunk(pid, "vitals_baseline",
            f"Usual vitals: resting heart rate {vb['resting_heart_rate_bpm']} bpm, blood pressure {vb['usual_blood_pressure']}, "
            f"SpO2 {vb['usual_spo2_percent']} percent. {vb['note']}"))

    for v in rec.get("visits", []):
        out.append(Chunk(pid, "visits",
            f"Doctor visit on {v['date']} with {v['doctor']} for {v['reason']}. Summary: {v['summary']}", v["date"]))

    for a in rec.get("appointments", []):
        out.append(Chunk(pid, "appointments",
            f"Upcoming appointment on {a['date']} at {a['time']} with {a['doctor']} for {a['reason']} at {a['location']}.", a["date"]))

    for i in rec.get("care_instructions", []):
        out.append(Chunk(pid, "care_instructions", f"Care instruction: {i}"))

    for n in rec.get("doctor_notes", []):
        out.append(Chunk(pid, "doctor_notes", f"Doctor note ({n['date']}): {n['note']}", n["date"]))

    return out


# ── BM25 index ────────────────────────────────────────────────────────────────
class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.N) if self.N else 0
        self.tf = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, query: list[str], i: int) -> float:
        s, dl = 0.0, len(self.docs[i])
        for t in set(query):
            f = self.tf[i].get(t, 0)
            if not f:
                continue
            s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


class PatientRecordStore:
    """Reads patient records from the SQLite database (db.py) — the primary data source."""

    def __init__(self):
        self.records: dict[str, dict] = {}
        self.chunks: dict[str, list[Chunk]] = {}
        self.index: dict[str, _BM25] = {}
        self.db_mtime = 0.0
        self.reload()

    def reload(self):
        self.records.clear(); self.chunks.clear(); self.index.clear()
        self.db_mtime = db.db_mtime()
        for pid, rec in db.load_all_records().items():
            self.records[pid] = rec
            self.chunks[pid] = _build_chunks(rec)
            self.index[pid] = _BM25([tokenize(c.text + " " + c.section) for c in self.chunks[pid]])
        if self.records:
            print(f"[RAG] Loaded {len(self.records)} patient record(s) from SQLite: {list(self.records)}")
        else:
            print("[RAG] WARNING: database has no patients. Run: python pdf_ingest.py <file.pdf>")

    def has_patient(self, pid: str) -> bool:
        return pid in self.records

    def patient_list(self) -> list[dict]:
        return [{"id": pid, "name": rec["profile"].get("name") or pid} for pid, rec in self.records.items()]

    def vocabulary_hint(self, pid: str) -> str:
        """Names the speech-to-text should expect (medicines, conditions, doctors)."""
        rec = self.records.get(pid)
        if not rec:
            return ""
        meds = [m["name"] for m in rec.get("medications", []) if m.get("name")]
        conds = [c["name"] for c in rec.get("conditions", []) if c.get("name")]
        docs: list[str] = []
        for section in ("visits", "appointments"):
            for x in rec.get(section, []):
                d = (x.get("doctor") or "").strip()
                if d and d not in docs:
                    docs.append(d)
        parts = []
        if meds:
            parts.append("Medicines: " + ", ".join(meds) + ".")
        if conds:
            parts.append("Conditions: " + ", ".join(conds) + ".")
        if docs:
            parts.append("Doctors: " + ", ".join(docs) + ".")
        return " ".join(parts)

    def overview(self, pid: str) -> dict | None:
        """Compact summary for the web page's patient card."""
        rec = self.records.get(pid)
        if not rec:
            return None
        p = rec["profile"]
        today = date.today().isoformat()
        upcoming = sorted((a for a in rec.get("appointments", []) if (a.get("date") or "") >= today),
                          key=lambda a: (a.get("date", ""), a.get("time", "")))
        return {
            "id": pid,
            "name": p.get("name"), "age": p.get("age"), "sex": p.get("sex"),
            "blood_group": p.get("blood_group"), "primary_doctor": p.get("primary_doctor"),
            "emergency_contact": p.get("emergency_contact"),
            "allergies": [{"substance": a["substance"], "severity": a["severity"]} for a in rec.get("allergies", [])],
            "conditions": [c["name"] for c in rec.get("conditions", [])],
            "medications": [{"name": m["name"], "dose": m["dose"], "frequency": m["frequency"]}
                            for m in rec.get("medications", [])],
            "next_appointment": upcoming[0] if upcoming else None,
        }

    def safety_summary(self, pid: str) -> str:
        """Always-on facts the AI must never miss: allergies, conditions, current meds."""
        rec = self.records[pid]
        allergies = "; ".join(f"{a['substance']} ({a['reaction']})" for a in rec.get("allergies", [])) or "none recorded"
        conds = "; ".join(c["name"] for c in rec.get("conditions", [])) or "none recorded"
        meds = "; ".join(f"{m['name']} {m['dose']} ({m['frequency']})" for m in rec.get("medications", [])) or "none recorded"
        p = rec["profile"]
        return (f"Name: {p['name']}, age {p['age']}, {p['sex']}. "
                f"ALLERGIES: {allergies}. CONDITIONS: {conds}. CURRENT MEDICINES: {meds}.")

    def retrieve(self, pid: str, question: str, k: int = 4) -> list[Chunk]:
        """Top matching record chunks: no weak matches, no duplicates,
        no past appointments, and only ONE appointment for 'next appointment' questions."""
        if pid not in self.index:
            return []
        q = expand_query(question)
        idx = self.index[pid]
        scored = sorted(((idx.score(q, i), i) for i in range(idx.N)), reverse=True)
        if not scored or scored[0][0] <= 0:
            return []

        best = scored[0][0]
        today = date.today().isoformat()
        hits: list[Chunk] = []
        seen: set[str] = set()
        for s, i in scored:
            if s <= 0 or s < best * 0.45 or len(hits) >= k:      # drop weak matches
                break
            c = self.chunks[pid][i]
            if c.section == "appointments" and c.date and c.date < today:
                continue                                          # past appointments aren't "upcoming"
            key = " ".join(c.text.lower().split())
            if key in seen:                                       # identical record text
                continue
            seen.add(key)
            hits.append(c)

        # "next appointment" should mean ONE appointment: the earliest upcoming
        qwords = set(re.findall(r"[a-z]+", question.lower()))
        appts = [c for c in hits if c.section == "appointments"]
        if len(appts) > 1 and qwords & {"next", "upcoming", "soon"}:
            keep = min(appts, key=lambda c: c.date)
            hits = [c for c in hits if c.section != "appointments" or c is keep]
        return hits


# Single shared store
_store: PatientRecordStore | None = None

def get_store() -> PatientRecordStore:
    """Shared store. Reloads automatically when the database file changes,
    so a newly ingested PDF is used without restarting the server."""
    global _store
    if _store is None:
        _store = PatientRecordStore()
    elif db.db_mtime() != _store.db_mtime:
        print("[RAG] Database changed — reloading records")
        _store.reload()
    return _store


def build_context_with_sources(pid: str, question: str, k: int = 4) -> tuple[str, list[Chunk]]:
    """Text block for the LLM prompt, plus the record chunks it was built from."""
    store = get_store()
    if not store.has_patient(pid):
        return "No health record found for this patient.", []
    hits = store.retrieve(pid, question, k)
    lines = [f"- [{c.section}] {c.text}" for c in hits]
    text = ("PATIENT SUMMARY:\n" + store.safety_summary(pid) +
            "\n\nRELEVANT RECORDS:\n" + ("\n".join(lines) if lines else "- (nothing specific found)"))
    return text, hits


def build_context(pid: str, question: str, k: int = 4) -> str:
    """Text block injected into the LLM prompt."""
    return build_context_with_sources(pid, question, k)[0]