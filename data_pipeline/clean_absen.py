"""
Clean the raw bimbel attendance Excel into the 8-field session format
from docs/IMPLEMENTASI AI MODEL + COMPARISON.md, section 3.2:

    id, tutor_id, student_id, subject_ids, lesson_date,
    start_time, duration_minutes, submitted_at

Usage:
    python data_pipeline/clean_absen.py "data/raw/<raw.xlsx>" data/processed/absen

Outputs (in output_dir):
    sessions_clean.csv       8 fields, subject_ids as a JSON list string
    sessions_clean.json      same rows as a JSON array (the `target` shape)
    lookup_tutors.csv        tutor_id   <-> name          (PRIVATE: real names)
    lookup_students.csv      student_id <-> name/kelas/school (PRIVATE)
    lookup_subjects.csv      subject_id <-> subject name
    cleaning_issues.csv      every row that was fixed, flagged or dropped
"""
import csv
import datetime as dt
import hashlib
import json
import random
import re
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

# Fixed namespace so the same name always gets the same UUID on every run.
NS = uuid.UUID("6f0b6a8e-3c7e-4b8f-9a51-6d2f0c1e7a42")
WIB = dt.timezone(dt.timedelta(hours=7))
MAX_SUBJECTS = 3
# Sheets that hold the same child are merged via a PRIVATE config file
# (config.local.json, git-ignored) because it contains children's names.
# Format: {"sheet_aliases": {"<sheet name, lower-case>": "<sheet it belongs to>"}}
CONFIG_FILE = Path(__file__).with_name("config.local.json")
SHEET_ALIASES = {}
if CONFIG_FILE.exists():
    SHEET_ALIASES = {k.strip().lower(): v.strip().lower()
                     for k, v in json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("sheet_aliases", {}).items()}

# ---------------------------------------------------------------- subjects
# Order matters only for readability; every pattern is tested against the text.
SUBJECT_PATTERNS = [
    ("Matematika", r"matematika|\bmtk\b|\bmath\b|bilangan|penjumlahan|pengurangan|perkalian|pembagian|"
                   r"berhitung|menghitung|pecahan|spldv|pldv|phytagoras|pythagoras|bentuk ruang|"
                   r"menghitung lompat|nilai tempat|game matematika"),
    ("Bahasa Inggris", r"inggris|ingris|english|\bb\.?\s?ing\b|\bbing\b"),
    ("Bahasa Indonesia", r"indonesia|\bb?\.?\s?indo\b|\bbindo\b|huruf kapital|cerita pendek|cerpen|"
                         r"kalimat tanya"),
    ("Bahasa Arab", r"\barab\b|\bb\.\s?arab|\bbarab\b"),
    ("Mulok Bahasa Dayak", r"dayak|mulok|patih rumbih"),
    ("Mengaji", r"mengaji|\bngaji\b|iqro|iqra|hijaiyah|\bbtq\b|tajwid"),
    ("Tahfidz", r"tahfidz|hafalan|menghafal surah|muroja|muraja|tasmi|jus\s?amma|ayat kursi"),
    ("Pendidikan Agama Islam", r"\bpai\b|agama|fiqih|aqidah|akidah|diniyah|religi|sholat|shalat|wudhu|"
                               r"do[’']a|rukun iman|nama\" nabi|niat shalat"),
    ("Pendidikan Pancasila", r"pancasila|\bpkn\b|toleransi"),
    ("IPAS", r"\bipas\b|\bipa\b|\bips\b|pancaindera|peredaran darah|pertumbuhan dan perkembangan|"
             r"perkembangan dan pertumbuhan"),
    ("Seni", r"seni|mewarnai|menggunting|pianika|lagu daerah"),
    ("PJOK", r"pjok"),
    ("Informatika", r"tikom|komputer"),
    ("Calistung", r"calistung|membaca|menulis|mengeja|abjad|suku kata|huruf vocal|huruf vokal|konsonan"),
    ("Fisika", r"fisika"),
    ("Kimia", r"kimia"),
    ("Biologi", r"biologi"),
    ("Ekonomi", r"ekonomi"),
    ("Geografi", r"geografi"),
    ("Sejarah", r"sejarah"),
]
# Lesson-prep rows that name no subject at all.
GENERIC_SUBJECT = "Persiapan Ulangan (Umum)"
GENERIC_PATTERNS = r"persiapan ulangan|review chapter"


def subjects_from_text(text):
    """Return canonical subjects in order of first appearance in the text."""
    t = " ".join(str(text).split()).lower()
    hits = []
    for name, pat in SUBJECT_PATTERNS:
        m = re.search(pat, t)
        if m:
            hits.append((m.start(), name))
    # "Calistung" keywords (membaca/menulis) also appear inside Mengaji/Bahasa
    # lessons; only keep Calistung when it is the sole match or named explicitly.
    names = [n for _, n in sorted(hits)]
    if "Calistung" in names and len(names) > 1 and "calistung" not in t:
        names.remove("Calistung")
    if not names and re.search(GENERIC_PATTERNS, t):
        names = [GENERIC_SUBJECT]
    return names


# ---------------------------------------------------------------- helpers
def uid(kind, key):
    return str(uuid.uuid5(NS, f"{kind}:{key}"))


def norm_name(s):
    return " ".join(str(s).split()).title()


def parse_time_range(raw):
    """'14.30 - 15.30' / '14:57-15:58' / '11 00-12.00' -> (start, end) as minutes."""
    s = str(raw).replace('"', "").strip()
    parts = re.split(r"\s*-\s*", s)
    if len(parts) != 2:
        return None, None
    out = []
    for p in parts:
        m = re.fullmatch(r"(\d{1,2})[.:\s](\d{2})", p.strip())
        if not m:
            return None, None
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 23 or mi > 59:
            return None, None
        out.append(h * 60 + mi)
    return out[0], out[1]


def stated_minutes(raw):
    """Column I on a few sheets: '1 Jam 30 Menit' -> 90."""
    if raw in (None, ""):
        return None
    s = str(raw).lower()
    h = re.search(r"(\d+)\s*jam", s)
    m = re.search(r"(\d+)\s*menit", s)
    if not h and not m:
        return None
    return (int(h.group(1)) * 60 if h else 0) + (int(m.group(1)) if m else 0)


def date_candidates(v):
    """
    The workbook's locale is D/M but tutors typed M/D. So:
    - text cells ('9/14/2026') are M/D and unambiguous;
    - date cells were mis-read as D/M, so the real date is usually month<->day
      swapped. A few tutors' cells were already right, so return both options.
    """
    if isinstance(v, dt.datetime):
        cands = {"swapped": None, "as_is": v.date()}
        try:
            cands["swapped"] = dt.date(v.year, v.day, v.month)
        except ValueError:
            pass
        return cands
    m, d, y = (int(x) for x in str(v).strip().split("/"))
    return {"text": dt.date(y, m, d)}


def hm(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------- read raw
def read_rows(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    rows = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        header_seen = False
        for r_idx, r in enumerate(ws.iter_rows(values_only=True, max_col=9), start=1):
            r = list(r) + [None] * (9 - len(r))
            if not header_seen:
                header_seen = bool(r[1]) and str(r[1]).strip().lower() == "nama tutor"
                continue
            if all(c in (None, "") for c in r[:8]):
                continue
            rows.append({
                "sheet": sheet, "excel_row": r_idx, "no": r[0], "tutor": r[1], "murid": r[2],
                "kelas": r[3], "sekolah": r[4], "mapel": r[5], "tanggal": r[6],
                "durasi": r[7], "durasi_kolom_i": r[8],
            })
    return rows


# ---------------------------------------------------------------- main
def main(src, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = read_rows(src)
    issues = []

    def issue(row, kind, action, detail):
        issues.append({"sheet": row["sheet"], "excel_row": row["excel_row"], "issue": kind,
                       "action": action, "detail": detail})

    # 1. drop rows with no usable content (empty template sheet, stray text)
    usable = []
    for row in raw:
        if not row["tutor"] or not row["murid"] or not row["tanggal"]:
            issue(row, "baris_kosong_atau_tidak_lengkap", "dibuang",
                  f"tutor={row['tutor']!r} murid={row['murid']!r} tanggal={row['tanggal']!r}")
            continue
        usable.append(row)

    # 2. dates: resolve swapped/as-is per cell using text dates of the same
    #    sheet+tutor block as anchors
    blocks = defaultdict(list)
    for row in usable:
        blocks[(row["sheet"], norm_name(row["tutor"]))].append(row)
    for rows in blocks.values():
        anchors = [date_candidates(r["tanggal"])["text"] for r in rows
                   if not isinstance(r["tanggal"], dt.datetime)]
        for r in rows:
            c = date_candidates(r["tanggal"])
            if "text" in c:
                r["lesson_date"] = c["text"]
                continue
            opts = {k: v for k, v in c.items() if v is not None}
            if anchors:
                # Swapping is the normal fix; keep the cell as-is only when the
                # swapped date falls outside this block's date range (+-31 days).
                lo, hi = min(anchors) - dt.timedelta(31), max(anchors) + dt.timedelta(31)
                sw = opts.get("swapped")
                if sw is not None and lo <= sw <= hi:
                    pick = "swapped"
                else:
                    med = sorted(anchors)[len(anchors) // 2]
                    pick = min(opts, key=lambda k: abs((opts[k] - med).days))
            else:
                pick = "swapped" if "swapped" in opts else "as_is"
            r["lesson_date"] = opts[pick]
            if pick == "as_is" and opts.get("swapped") != opts["as_is"]:
                issue(r, "tanggal_tidak_ditukar", "diperbaiki",
                      f"sel tanggal {c['as_is']} dipakai apa adanya (cocok dengan tanggal lain di blok ini)")
        # out-of-sequence check (rows are typed in chronological order)
        seq = sorted(rows, key=lambda r: r["excel_row"])
        for i, r in enumerate(seq):
            nbrs = [x["lesson_date"] for x in seq[max(0, i - 2):i] + seq[i + 1:i + 3]]
            if nbrs and min(abs((r["lesson_date"] - d).days) for d in nbrs) > 20:
                issue(r, "tanggal_di_luar_urutan", "dipertahankan_perlu_cek",
                      f"{r['lesson_date']} jauh dari baris sekitarnya {sorted(map(str, nbrs))}")

    # 3. time range -> start_time + duration_minutes
    clean = []
    for r in usable:
        start, end = parse_time_range(r["durasi"])
        if start is None:
            issue(r, "jam_les_tidak_terbaca", "dibuang", f"durasi les={r['durasi']!r}")
            continue
        if end <= start:
            issue(r, "jam_selesai_sebelum_mulai", "diperbaiki",
                  f"{r['durasi']!r} ditukar menjadi {hm(end)}-{hm(start)}")
            start, end = end, start
        dur = end - start
        stated = stated_minutes(r["durasi_kolom_i"])
        if stated is not None and abs(stated - dur) > 10:
            issue(r, "durasi_tidak_cocok_kolom_i", "dipertahankan_perlu_cek",
                  f"dari jam={dur} menit, tertulis di kolom I={stated} menit")
        r["start_min"], r["end_min"], r["duration_minutes"] = start, end, dur
        clean.append(r)

    # 4. subjects
    for r in clean:
        subs = subjects_from_text(r["mapel"])
        if not subs:
            issue(r, "mapel_tidak_dikenali", "dibuang", repr(r["mapel"]))
            r["drop"] = True
            continue
        if len(subs) > MAX_SUBJECTS:
            issue(r, "mapel_lebih_dari_3", "dipotong_3_pertama",
                  f"{subs} -> {subs[:MAX_SUBJECTS]} | teks: {' '.join(str(r['mapel']).split())}")
            subs = subs[:MAX_SUBJECTS]
        r["subjects"] = subs
    clean = [r for r in clean if not r.get("drop")]

    # 5. identities
    tutors, students, subjects = {}, {}, {}
    for r in clean:
        t = norm_name(r["tutor"])
        r["tutor_id"] = tutors.setdefault(t, {"tutor_id": uid("tutor", t), "name": t})["tutor_id"]
        # The same first name can belong to different children on different sheets,
        # so the sheet name is part of the key (one sheet may also hold several children).
        sheet_key = r["sheet"].strip().lower()
        # Sheets listed in config.local.json are the same child (e.g. last year's sheet).
        sheet_key = SHEET_ALIASES.get(sheet_key, sheet_key)
        murid = norm_name(r["murid"])
        key = f"{sheet_key}|{murid.lower()}"
        s = students.setdefault(key, {"student_id": uid("student", key), "name": murid,
                                      "sheet": r["sheet"].strip(), "kelas": set(), "sekolah": set()})
        if r["kelas"]:
            s["kelas"].add(str(r["kelas"]).strip())
        if r["sekolah"]:
            s["sekolah"].add(str(r["sekolah"]).strip())
        r["student_id"] = s["student_id"]
        ids = []
        for name in r["subjects"]:
            ids.append(subjects.setdefault(name, {"subject_id": uid("subject", name), "name": name})["subject_id"])
        r["subject_ids"] = sorted(set(ids))

    # 6. session id + synthetic submitted_at
    #    The raw file has no submit time. Assume the tutor submits after the lesson
    #    ends: end time + 5..180 min, seeded per row so every run gives the same value.
    for r in clean:
        key = f"{r['sheet']}|{r['excel_row']}"
        r["id"] = uid("session", key)
        rnd = random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))
        local = dt.datetime.combine(r["lesson_date"], dt.time(0, 0), WIB) + dt.timedelta(
            minutes=r["end_min"] + rnd.randint(5, 180), seconds=rnd.randint(0, 59))
        r["submitted_at"] = local.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        r["start_time"] = hm(r["start_min"])

    # 7. exact duplicates (section 1.1 RED rule) - report only, keep rows
    dup = Counter((r["tutor_id"], r["student_id"], tuple(r["subject_ids"]), r["lesson_date"],
                   r["duration_minutes"]) for r in clean)
    for r in clean:
        k = (r["tutor_id"], r["student_id"], tuple(r["subject_ids"]), r["lesson_date"], r["duration_minutes"])
        if dup[k] > 1:
            issue(r, "duplikat_persis", "dipertahankan_perlu_cek", "5 atribut sama dengan baris lain")

    # ---------------------------------------------------------------- write
    clean.sort(key=lambda r: r["submitted_at"])
    fields = ["id", "tutor_id", "student_id", "subject_ids", "lesson_date", "start_time",
              "duration_minutes", "submitted_at"]
    records = [{
        "id": r["id"], "tutor_id": r["tutor_id"], "student_id": r["student_id"],
        "subject_ids": r["subject_ids"], "lesson_date": r["lesson_date"].isoformat(),
        "start_time": r["start_time"], "duration_minutes": r["duration_minutes"],
        "submitted_at": r["submitted_at"],
    } for r in clean]

    with open(out / "sessions_clean.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            w.writerow({**rec, "subject_ids": json.dumps(rec["subject_ids"])})
    (out / "sessions_clean.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))

    def dump(name, rows, cols):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    sess_t, sess_s = Counter(r["tutor_id"] for r in clean), Counter(r["student_id"] for r in clean)
    dump("lookup_tutors.csv", [{**t, "sessions": sess_t[t["tutor_id"]]} for t in sorted(tutors.values(), key=lambda x: x["name"])],
         ["tutor_id", "name", "sessions"])
    dump("lookup_students.csv", [{"student_id": s["student_id"], "name": s["name"], "sheet": s["sheet"],
                                  "kelas": "; ".join(sorted(s["kelas"])), "sekolah_raw": "; ".join(sorted(s["sekolah"])),
                                  "sessions": sess_s[s["student_id"]]}
                                 for s in sorted(students.values(), key=lambda x: x["name"])],
         ["student_id", "name", "sheet", "kelas", "sekolah_raw", "sessions"])
    sub_count = Counter(i for r in clean for i in r["subject_ids"])
    dump("lookup_subjects.csv", [{**s, "sessions": sub_count[s["subject_id"]]} for s in sorted(subjects.values(), key=lambda x: x["name"])],
         ["subject_id", "name", "sessions"])
    dump("cleaning_issues.csv", sorted(issues, key=lambda x: (x["issue"], x["sheet"], x["excel_row"])),
         ["sheet", "excel_row", "issue", "action", "detail"])

    # summary for the console
    print(f"raw rows: {len(raw)}  ->  clean sessions: {len(records)}")
    print(f"tutors: {len(tutors)}  students: {len(students)}  subjects: {len(subjects)}")
    print("issues:", dict(Counter((i['issue'], i['action']) for i in issues)))
    print("lesson months:", dict(sorted(Counter(r['lesson_date'][:7] for r in records).items())))
    print("duration min/max:", min(r['duration_minutes'] for r in records), max(r['duration_minutes'] for r in records))
    print("latest start:", max(r['start_time'] for r in records))
    print("subject_ids per session:", dict(Counter(len(r['subject_ids']) for r in records)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
