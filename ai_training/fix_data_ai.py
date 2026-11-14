"""
Usage:
    python ai_training/fix_data_ai.py data/raw/data_ai.csv data/ai_training/data_ai_clean.csv data/ai_training/data_ai_issues.csv

Format-only fixes for data_ai.csv -> the 8 fields of doc section 3.2. No row is removed.
  - lesson_date  M/D/YYYY -> YYYY-MM-DD
  - start_time   '9:00'   -> '09:00'
  - subject_ids  sorted
  - every column after the 8 fields is dropped (config columns, empty columns)

Business-rule cases (duplicates, odd start times, overlaps, ...) are NOT data errors:
they are labelled later by build_training_set.py. The issues file only lists real
format problems.
"""
import csv
import datetime as dt
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

FIELDS = ["id", "tutor_id", "student_id", "subject_ids", "lesson_date", "start_time",
          "duration_minutes", "submitted_at"]
MAX_SUBJECTS = 5   # AE 2026-11-14: the app will accept up to 5 subjects


def main(src, out_csv, out_issues):
    rows = list(csv.reader(open(src, encoding="utf-8-sig", newline="")))
    if rows[0][:8] != FIELDS:
        sys.exit(f"Unexpected header: {rows[0][:8]}")
    clean, issues = [], []

    def flag(ln, rid, kind, detail):
        issues.append({"csv_line": ln, "id": rid, "issue": kind, "detail": detail})

    for ln, raw in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in raw):
            continue
        r = dict(zip(FIELDS, (c.strip() for c in raw[:8])))
        extra = [c for c in raw[10:] if c.strip()]          # cols 9-10 are config, anything after is stray
        if extra:
            flag(ln, r["id"], "catatan_di_kolom_kosong", f"isi kolom tambahan {extra} (ikut terhapus)")
        try:
            for k in ("id", "tutor_id", "student_id"):
                uuid.UUID(r[k])
            d = dt.datetime.strptime(r["lesson_date"], "%m/%d/%Y").date()   # verified: source is M/D/YYYY
            h, m = map(int, r["start_time"].split(":"))
            assert 0 <= h <= 23 and 0 <= m <= 59
            int(r["duration_minutes"])
            dt.datetime.strptime(r["submitted_at"], "%Y-%m-%dT%H:%M:%SZ")
            subs = json.loads(r["subject_ids"])
            assert isinstance(subs, list) and subs
            for s in subs:
                uuid.UUID(s)
        except Exception as e:   # noqa: BLE001 - report and keep going
            flag(ln, r["id"], "format_tidak_valid", f"{type(e).__name__}: {e}")
            continue
        r["lesson_date"] = d.isoformat()
        r["start_time"] = f"{h:02d}:{m:02d}"
        r["subject_ids"] = json.dumps(sorted(subs))
        if len(subs) > MAX_SUBJECTS:
            flag(ln, r["id"], "mapel_lebih_dari_5", f"{len(subs)} mapel; aplikasi menerima maksimal {MAX_SUBJECTS}")
        if len(set(subs)) != len(subs):
            flag(ln, r["id"], "mapel_berulang", r["subject_ids"])
        clean.append(r)

    ids = Counter(r["id"] for r in clean)
    for r in clean:
        if ids[r["id"]] > 1:
            flag("-", r["id"], "id_ganda", "id sesi muncul lebih dari sekali")

    for path in (out_csv, out_issues):
        Path(path).parent.mkdir(parents=True, exist_ok=True)   # create output folder if missing
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(clean)
    with open(out_issues, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["csv_line", "id", "issue", "detail"])
        w.writeheader()
        w.writerows(issues)
    print(f"{len(clean)} rows written; issues: {dict(Counter(i['issue'] for i in issues)) or 'none'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
