# Generates realistic sample patient PDFs (for testing pdf_ingest.py).
#   python make_sample_pdf.py                  -> a PDF for EVERY data/patients/*.json
#   python make_sample_pdf.py patient_002      -> only data/pdfs/patient_002.pdf
#   python make_sample_pdf.py patient_002 patient_003
# Needs: pip install reportlab
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).parent
SRC_DIR = HERE / "data" / "patients"
OUT_DIR = HERE / "data" / "pdfs"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=18, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor("#1e3a8a"))
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
CELL = ParagraphStyle("C", parent=BODY, fontSize=8.5, leading=11)


def table(headers, rows, widths):
    data = [[Paragraph(f"<b>{h}</b>", CELL) for h in headers]] + \
           [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def build_pdf(rec: dict, out: Path) -> None:
    p, ec = rec["profile"], rec["profile"]["emergency_contact"]
    s = [
        Paragraph("City General Hospital — Patient Medical Record", H1),
        Paragraph("CONFIDENTIAL — sample / dummy data for testing", BODY),
        Paragraph("Patient Details", H2),
        Paragraph(f"<b>Name:</b> {p['name']} &nbsp; <b>Age:</b> {p['age']} &nbsp; <b>Sex:</b> {p['sex']} &nbsp; "
                  f"<b>Blood group:</b> {p['blood_group']}<br/>"
                  f"<b>Height:</b> {p['height_cm']} cm &nbsp; <b>Weight:</b> {p['weight_kg']} kg &nbsp; "
                  f"<b>Languages:</b> {p['language']}<br/>"
                  f"<b>Primary doctor:</b> {p['primary_doctor']}<br/>"
                  f"<b>Emergency contact:</b> {ec['name']} ({ec['relation']}), {ec['phone']}", BODY),
        Paragraph("Allergies", H2),
        table(["Substance", "Reaction", "Severity"],
              [[a["substance"], a["reaction"], a["severity"]] for a in rec["allergies"]], [50, 80, 40]),
        Paragraph("Medical Conditions", H2),
        table(["Condition", "Since", "Status"],
              [[c["name"], c["since"], c["status"]] for c in rec["conditions"]], [70, 25, 75]),
        Paragraph("Past Surgeries", H2),
        table(["Procedure", "Date", "Hospital", "Notes"],
              [[x["procedure"], x["date"], x["hospital"], x["notes"]] for x in rec["surgeries"]], [40, 22, 40, 68]),
        Paragraph("Current Medications", H2),
        table(["Medicine", "Dose", "How often", "Purpose"],
              [[m["name"], m["dose"], m["frequency"], m["purpose"]] for m in rec["medications"]], [35, 28, 60, 47]),
        Paragraph("Laboratory Results", H2),
        table(["Date", "Test", "Result", "Reference range", "Comment"],
              [[l["date"], l["test"], l["value"], l["reference"], l["note"]] for l in rec["lab_results"]],
              [22, 35, 25, 48, 40]),
        Paragraph("Usual Vitals", H2),
        Paragraph(f"Resting heart rate: {rec['vitals_baseline']['resting_heart_rate_bpm']} bpm. "
                  f"Usual blood pressure: {rec['vitals_baseline']['usual_blood_pressure']}. "
                  f"Usual SpO2: {rec['vitals_baseline']['usual_spo2_percent']}%. "
                  f"{rec['vitals_baseline']['note']}", BODY),
        Paragraph("Consultation History", H2),
        table(["Date", "Doctor", "Reason", "Summary"],
              [[v["date"], v["doctor"], v["reason"], v["summary"]] for v in rec["visits"]], [22, 38, 38, 72]),
        Paragraph("Upcoming Appointments", H2),
        table(["Date", "Time", "Doctor", "Reason", "Location"],
              [[a["date"], a["time"], a["doctor"], a["reason"], a["location"]] for a in rec["appointments"]],
              [22, 14, 38, 55, 41]),
        Paragraph("Care Instructions", H2),
        *[Paragraph(f"• {i}", BODY) for i in rec["care_instructions"]],
        Paragraph("Doctor's Notes", H2),
        *[Paragraph(f"<b>{n['date']}:</b> {n['note']}", BODY) for n in rec["doctor_notes"]],
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=16 * mm, title="Patient Medical Record").build(s)
    print(f"Created {out}")


def main() -> int:
    ids = sys.argv[1:] or sorted(f.stem for f in SRC_DIR.glob("*.json"))
    if not ids:
        print(f"No patient JSON files found in {SRC_DIR}")
        return 1
    status = 0
    for pid in ids:
        src = SRC_DIR / f"{pid}.json"
        if not src.exists():
            print(f"Not found: {src}")
            status = 1
            continue
        build_pdf(json.loads(src.read_text(encoding="utf-8")), OUT_DIR / f"{pid}.pdf")
    return status


if __name__ == "__main__":
    sys.exit(main())