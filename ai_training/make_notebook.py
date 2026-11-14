"""
Regenerate pipeline_colab.ipynb from the .py scripts (code is copied, not rewritten).

Run this after changing fix_data_ai.py, inject_anomalies.py, build_training_set.py or features.py:

    pip install nbformat
    python ai_training/make_notebook.py

The .py scripts are the source of truth. Do not edit code inside the notebook by hand:
changes there are lost the next time this script runs.
"""
import ast
import re
import sys
from pathlib import Path

import nbformat as nbf

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent      # ai_training folder
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC / "pipeline_colab.ipynb"
files = {n: (SRC / f"{n}.py").read_text(encoding="utf-8") for n in
         ("features", "fix_data_ai", "inject_anomalies", "build_training_set")}
trees = {n: ast.parse(s) for n, s in files.items()}


def get(module, *names, rename=None):
    """Source of top-level defs/classes/assignments named `names`, in file order."""
    src, out = files[module], []
    for node in trees[module].body:
        node_names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node_names = [node.name]
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                elts = t.elts if isinstance(t, ast.Tuple) else [t]
                node_names += [e.id for e in elts if isinstance(e, ast.Name)]
        if set(node_names) & set(names):
            seg = ast.get_source_segment(src, node)
            # keep comment lines directly above the node
            lines = src.splitlines()
            i = node.lineno - 2
            comments = []
            while i >= 0 and lines[i].lstrip().startswith("#"):
                comments.insert(0, lines[i])
                i -= 1
            out.append("\n".join(comments + [seg]))
    code = "\n\n\n".join(out)
    # imports between the scripts are not needed: everything lives in one notebook
    code = re.sub(r"^\s*from (features|build_training_set) import .*\n", "", code, flags=re.M)
    for old, new in (rename or {}).items():
        code = re.sub(rf"\b{old}\b", new, code)
    return code


cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

# ------------------------------------------------------------------ intro
md("""
# AI Training — Data Model Validasi Absen (versi Colab)

Notebook ini menyiapkan **data training** untuk model validasi absen tutor Bimbel Mumtaz.
Isinya sama persis dengan skrip di folder `ai_training/` repo `data-science` (kode disalin dari file `.py`),
hanya dipecah per bagian dan diberi penjelasan.

Acuan: `docs/IMPLEMENTASI AI MODEL + COMPARISON.md` (repo `mumtaz-bimbel-dashboard`, branch `feat/ui-ux-polish`).

## Alur

```
data_ai.csv (dari tim)
   │  Langkah 1  fix_data_ai          → data_ai_clean.csv            format dirapikan, 8 field
   │  Langkah 2  inject_anomalies     → sessions_with_synthetic.csv  + baris anomali sintetis
   │  Langkah 3  build_training_set   → training_dataset.csv          19 fitur + label
   │                                  → sample_requests.json          contoh request FastAPI
   └─ features.py dipakai di Langkah 3 (dan nantinya di FastAPI)
```

## Cara pakai

1. Di Colab: **Runtime → Run all** (atau jalankan sel satu per satu dari atas).
2. Saat diminta, **upload `data_ai.csv`**.
3. Di bagian akhir, file hasil otomatis bisa di-*download*.

## ⚠️ Privasi data

Data berisi catatan les anak-anak yang asli. **Jangan simpan notebook ini ke GitHub dalam keadaan
ada output** (tabel pratinjau berisi data). Sebelum commit: **Edit → Clear all outputs**.
Simpan file data di Google Drive tim, bukan di repo.
""")

# ------------------------------------------------------------------ setup
md("""
## 0. Persiapan

Semua kode hanya memakai *library* bawaan Python. `pandas` dipakai **hanya untuk menampilkan
pratinjau tabel** di notebook ini (sudah tersedia di Colab).
""")
code("""
import csv
import datetime as dt
import hashlib
import json
import random
import sys
import uuid
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", 30)
""")
md("""
### Upload data

Sel di bawah meminta file `data_ai.csv`. Di luar Colab (misalnya Jupyter di laptop), sel ini
memakai file `data_ai.csv` di folder yang sama dengan notebook.

> Alternatif lewat Google Drive: `from google.colab import drive; drive.mount('/content/drive')`
> lalu ubah `RAW_PATH` menjadi lokasi file di Drive.
""")
code("""
WORK = Path("pipeline_output")
WORK.mkdir(exist_ok=True)

try:
    from google.colab import files  # hanya ada di Colab
    uploaded = files.upload()
    RAW_PATH = Path(next(iter(uploaded)))
except ImportError:
    RAW_PATH = Path("data_ai.csv")

print("File data:", RAW_PATH, "-", RAW_PATH.stat().st_size // 1024, "KB")
pd.read_csv(RAW_PATH, encoding="utf-8-sig").head()
""")

# ------------------------------------------------------------------ step 1
md("""
## Langkah 1 — Merapikan format (`fix_data_ai.py`)

Data dari tim sudah setengah bersih, tetapi formatnya belum sesuai **section 3.2** dokumen.
Langkah ini **hanya memperbaiki format, tidak ada baris yang dihapus**:

| Kolom | Sebelum | Sesudah |
|---|---|---|
| `lesson_date` | `6/16/2025` (bulan/hari/tahun) | `2025-06-16` |
| `start_time` | `9:00` | `09:00` |
| `subject_ids` | urutan acak | diurutkan |
| kolom tambahan | `durationOutlierMinutes`, `lateStartAfter`, 3 kolom kosong | dibuang |

Kasus seperti duplikat atau jam mulai aneh **bukan** kesalahan format — kasus itu diberi label
di Langkah 3. File *issues* hanya berisi masalah format yang sebenarnya.
""")
code(get("fix_data_ai", "FIELDS", "MAX_SUBJECTS", "main", rename={"main": "fix_data_ai"}))
code("""
CLEAN_PATH = WORK / "data_ai_clean.csv"
fix_data_ai(RAW_PATH, CLEAN_PATH, WORK / "data_ai_issues.csv")

display(pd.read_csv(WORK / "data_ai_issues.csv"))
pd.read_csv(CLEAN_PATH).head()
""")

# ------------------------------------------------------------------ features
md("""
## Aturan bisnis & fitur (`features.py`)

Bagian ini adalah **inti logika**: aturan label dan perhitungan fitur. File `features.py` dibuat
agar bisa dipakai bersama oleh skrip *training* **dan** FastAPI, sehingga fitur saat *training*
dan saat aplikasi berjalan dihitung dengan cara yang sama (section 4.5).

### Aturan label

| Label | Kondisi |
|---|---|
| 🔴 `rejected` | Duplikat persis: tutor, murid, mapel, tanggal, **jam mulai**, dan durasi sama dengan sesi yang di-*submit* lebih dulu |
| 🔵 `needs_revision` | Durasi `> 180` atau `< 30` menit · mulai `> 21:00` atau `< 07:00` · duplikat samar (jam mulai beda) · tutor bentrok dengan murid lain · murid bentrok dengan tutor lain · tutor & murid yang sama bentrok tetapi bukan duplikat persis |
| 🟢 `valid` | Tidak ada temuan (tepat 180/30 menit dan tepat 21:00/07:00 masih valid) |

Prioritas: `rejected` > `needs_revision` > `valid`. Semua temuan tetap dicatat di `findings`.

### Konfigurasi dan daftar fitur

`DEFAULT_CONFIG` berisi batas-batas aturan (bisa diubah tanpa mengubah kode).
`FEATURE_NAMES` adalah **urutan kolom input model** — 13 fitur dari section 3.3 ditambah 6 fitur
baru untuk aturan tambahan. Mengubah urutannya berarti model harus dilatih ulang.
""")
code(get("features", "DEFAULT_CONFIG", "FEATURE_NAMES"))
md("""
### Fungsi bantu

- `hhmm_to_minutes("14:00")` → `840` (menit sejak tengah malam), supaya jam bisa dihitung.
- `_subjects(session)` → daftar mapel yang sudah diurutkan, supaya `["B","A"]` dan `["A","B"]` dianggap sama.
- `_overlaps(...)` → `True` bila dua rentang jam **bertabrakan**. Sesi yang bersambung
  (14:00–15:00 lalu 15:00–16:00) **tidak** dianggap bertabrakan.
""")
code(get("features", "hhmm_to_minutes", "_subjects", "_overlaps"))
md("""
### `compute_features(target, comparisons, config)`

Menghitung **19 angka** untuk satu sesi (`target`) dengan membandingkannya ke sesi-sesi
sebelumnya (`comparisons`). Model ML hanya bisa membaca angka, bukan JSON atau UUID.

Cara kerjanya:
1. Untuk setiap sesi pembanding, cek apakah tutor/murid/tanggalnya sama.
2. Hanya sesi **di tanggal yang sama** yang bisa menjadi duplikat atau bentrok jadwal.
3. Hitung jumlah tiap jenis temuan (duplikat persis, duplikat samar, bentrok tutor/murid/sendiri).
4. Fitur "batas" dihitung dengan pengurangan: **nol atau minus = aman, positif = melanggar** (section 3.4).

Fungsi ini juga mengembalikan `related`: ID sesi pembanding yang menyebabkan temuan
(dipakai untuk `related_lesson_session_id` di respons FastAPI).
""")
code(get("features", "compute_features"))
md("""
### `label_from_features(features)`

Mengubah 19 angka menjadi **label** dan **daftar temuan** sesuai aturan bisnis.
Fungsi ini dipakai untuk memberi label data *training*, dan di FastAPI bisa dipakai sebagai
*guardrail* (pengaman) agar aturan mutlak tidak pernah meleset.
""")
code(get("features", "label_from_features"))
md("""
### Uji cepat

Memastikan fungsi di atas benar:
- Contoh di **section 4.2** dokumen harus menghasilkan angka yang sama persis.
- Kasus batas: 180 menit valid, 181 tidak; 21:00 valid, 21:01 tidak; dst.
- Contoh dari diskusi: jam mulai sama tetapi durasi beda → `needs_revision`.

Jika ada yang salah, sel ini akan berhenti dengan `AssertionError`.
""")
code("""
T = {"id": "sesi-baru-002", "tutor_id": "tutor-diva", "student_id": "murid-abqary",
     "subject_ids": ["mapel-mtk"], "lesson_date": "2026-09-02", "start_time": "14:00",
     "duration_minutes": 60, "submitted_at": "2026-09-02T08:20:00Z"}
C = [{**T, "id": "sesi-lama-001", "submitted_at": "2026-09-02T08:05:00Z"},
     {**T, "id": "sesi-minggu-lalu-000", "lesson_date": "2026-08-26", "submitted_at": "2026-08-26T09:00:00Z"}]
f, rel = compute_features(T, C)
doc_4_2 = {"duration_minutes": 60, "duration_over_limit": -120, "start_minute": 840, "start_over_limit": -420,
           "subject_count": 1, "comparisons_count": 2, "exact_dup_count": 1, "same_tutor_student_date_count": 1,
           "same_student_date_diff_tutor_count": 0, "time_overlap_count": 1, "tutor_sessions_30d": 2,
           "student_sessions_30d": 2, "duration_diff_from_tutor_mean_30d": 0.0}
assert all(f[k] == v for k, v in doc_4_2.items())
print("Contoh section 4.2 cocok:", label_from_features(f), rel)

def label_of(changes, comparisons=()):
    t = {**T, **changes}
    return label_from_features(compute_features(t, list(comparisons))[0])[0]

cases = [
    ({"duration_minutes": 180}, [], "valid"), ({"duration_minutes": 181}, [], "needs_revision"),
    ({"duration_minutes": 30}, [], "valid"), ({"duration_minutes": 29}, [], "needs_revision"),
    ({"start_time": "21:00"}, [], "valid"), ({"start_time": "21:01"}, [], "needs_revision"),
    ({"start_time": "07:00"}, [], "valid"), ({"start_time": "06:59"}, [], "needs_revision"),
    ({}, [{**T, "id": "x", "duration_minutes": 90}], "needs_revision"),              # jam sama, durasi beda
    ({}, [{**T, "id": "x", "student_id": "lain", "start_time": "14:30"}], "needs_revision"),  # tutor bentrok
    ({}, [{**T, "id": "x", "tutor_id": "lain", "start_time": "14:30"}], "needs_revision"),    # murid bentrok
    ({}, [{**T, "id": "x", "student_id": "lain", "start_time": "15:00"}], "valid"),           # bersambung
    ({}, [{**T, "id": "x"}], "rejected"),                                            # duplikat persis
]
for changes, comps, expected in cases:
    assert label_of(changes, comps) == expected, (changes, expected)
print(f"{len(cases)} kasus uji lolos")
""")

# ------------------------------------------------------------------ step 3 functions (needed by step 2)
md("""
## Fungsi penyusun `comparisons` (`build_training_set.py`)

Fungsi-fungsi ini didefinisikan **sebelum** Langkah 2 karena Langkah 2 memakainya untuk
mengetahui label data asli terlebih dahulu.

### Memuat data

`load_sessions` membaca CSV, `prepare_sessions` mengubah teks menjadi tipe yang tepat
(tanggal, angka, daftar mapel) dan mencatat apakah baris itu sintetis.
""")
code(get("build_training_set", "WINDOW_DAYS", "MAX_COMPARISONS", "SESSION_FIELDS",
         "load_sessions", "prepare_sessions", "public"))
md("""
### `build_comparisons(target, by_person)`

Menyusun daftar sesi pembanding untuk satu `target`, sama seperti query *backend* saat aplikasi
berjalan (section 3.1). Sebuah sesi masuk `comparisons` bila:

1. **Tutor atau murid sama** dengan `target`,
2. **Tanggal les** berjarak maksimal **±30 hari**,
3. **Di-*submit* lebih dulu** daripada `target` (`submitted_at` lebih kecil).

Maksimal **100 sesi**. Bila lebih dari 100, yang diambil adalah sesi dengan tanggal les
**paling dekat** — jadi sesi di tanggal yang sama (yang menentukan duplikat & bentrok) selalu ikut.

`by_person` adalah indeks sesi per tutor dan per murid yang sudah diurutkan per tanggal, sehingga
pencarian cepat (`bisect`) walau datanya ribuan baris.

**Kenapa `submitted_at` penting?** Absen yang masuk **duluan** dianggap asli (`valid`), yang masuk
**belakangan** dianggap duplikat (`rejected`), karena saat absen pertama masuk, kembarannya belum ada.
""")
code(get("build_training_set", "build_comparisons", "_neg"))
md("""
### `label_sessions(sessions)`

Untuk **setiap sesi**: susun `comparisons` → hitung 19 fitur → beri label. Hasilnya satu baris per
sesi berisi fitur, label (`validity`), temuan (`findings`), dan ID sesi pembanding yang terkait.

Fungsi ini juga menyimpan **satu contoh request per jenis temuan** (`sample_requests.json`) dalam
format `{target, comparisons, config}` — format yang akan dikirim *backend* ke FastAPI.
""")
code(get("build_training_set", "label_sessions"))
md("""
### Label data asli (sebelum ditambah data sintetis)

Kita lihat dulu komposisi label data asli. Biasanya sebagian besar `valid`, dan beberapa aturan
hampir tidak punya contoh — itulah alasan Langkah 2 diperlukan.
""")
code("""
real_rows, _, _ = label_sessions(load_sessions(CLEAN_PATH))
real_df = pd.DataFrame(real_rows)
display(real_df["validity"].value_counts().to_frame("jumlah").assign(
    persen=lambda d: (d["jumlah"] / d["jumlah"].sum() * 100).round(1)))
real_df["findings"].str.split(";").explode().replace("", pd.NA).dropna().value_counts().to_frame("jumlah temuan")
""")

# ------------------------------------------------------------------ step 2
md("""
## Langkah 2 — Menambah anomali sintetis (`inject_anomalies.py`)

Model tidak bisa belajar aturan yang hampir tidak pernah ia lihat. Karena itu ditambahkan
**baris anomali buatan** (section 5.5) sampai komposisi akhir **60% valid / 20% needs_revision /
20% rejected**.

### Pengaturan

- `TARGET_SHARE`: komposisi label akhir yang diinginkan.
- `FIXED_COUNTS`: skenario dengan jumlah tetap (contoh batas yang valid, dan pasangan "kombinasi").
- `NEEDS_REVISION_SCENARIOS`: skenario `needs_revision`, jumlahnya dibagi rata.
- `plan_counts()`: **menghitung otomatis** berapa baris sintetis per skenario berdasarkan label data
  asli, sehingga komposisi tetap tepat walaupun data asli berubah.

| Skenario | Isi |
|---|---|
| `durasi_panjang` / `durasi_pendek` | durasi 181–300 / 10–29 menit |
| `mulai_malam` / `mulai_pagi` | mulai 21:01–22:30 / 05:00–06:59 |
| `duplikat_samar` | salinan sesi asli dengan jam mulai berbeda |
| `tumpang_tindih` / `_murid` / `_sendiri` | bentrok jadwal tutor / murid / tutor+murid yang sama |
| `duplikat` | salinan persis sesi asli, di-*submit* belakangan → `rejected` |
| `kombinasi_dup_*` | sesi tidak wajar + salinan persisnya (dua temuan sekaligus) |
| `batas_valid` | tepat 180/30 menit, tepat 21:00/07:00 → `valid` (mengajari model letak batas) |
""")
code(get("inject_anomalies", "SEED", "FIELDS", "DAY_START", "DAY_END", "LAST_MINUTE",
         "TARGET_SHARE", "FIXED_COUNTS", "NEEDS_REVISION_SCENARIOS", "plan_counts", "hm", "ts", "ts_str"))
md("""
### Kelas `Injector`

Membuat baris sintetis dengan **aman**:

- Setiap baris sintetis dibuat dari **sesi asli yang valid** (tutor, murid, mapel asli), lalu diubah
  sesuai skenario.
- Baris ditempatkan di **hari atau slot jam yang kosong**, dan waktu *submit* diatur, sehingga
  **label baris asli tidak ikut berubah**.
- Setiap baris diberi ID baru, `is_synthetic=1`, dan nama `scenario`.
- Memakai *seed* tetap (`SEED`), jadi hasilnya **selalu sama** setiap dijalankan.

Metode penting:
- `standalone()` — satu sesi tidak wajar (durasi/jam) di hari yang kosong.
- `exact_copy()` / `fuzzy_copy()` — duplikat persis / duplikat samar.
- `overlap_other_student()` / `overlap_other_tutor()` / `self_overlap()` — tiga jenis bentrok jadwal.
- `run()` — menjalankan semua skenario sesuai jumlah dari `plan_counts()`.
""")
code(get("inject_anomalies", "Injector"))
md("""
### Menjalankan Langkah 2

`inject_anomalies()` memberi label data asli, menghitung rencana jumlah per skenario, membuat baris
sintetis, lalu menyimpan gabungan data asli + sintetis.
""")
code(get("inject_anomalies", "main", rename={"main": "inject_anomalies"}).replace(
    "    from import label_sessions, prepare_sessions\n", ""))
code("""
SYN_PATH = WORK / "sessions_with_synthetic.csv"
inject_anomalies(CLEAN_PATH, SYN_PATH)
pd.read_csv(SYN_PATH)["scenario"].value_counts().to_frame("jumlah baris")
""")

# ------------------------------------------------------------------ step 3 run
md("""
## Langkah 3 — Menyusun data *training* (`build_training_set.py`)

Menjalankan `label_sessions` pada data gabungan (asli + sintetis), lalu menyimpan:

- `training/training_dataset.csv` — satu baris per sesi: 19 fitur, `validity`, `findings`,
  `related_session_id`, `scenario`, `is_synthetic`. **Siap untuk `model.fit(X, y)`**.
- `training/sample_requests.json` — contoh request untuk menguji FastAPI.
""")
code(get("build_training_set", "main", rename={"main": "build_training_set"}))
code("""
TRAIN_DIR = WORK / "training"
build_training_set(SYN_PATH, TRAIN_DIR)
train = pd.read_csv(TRAIN_DIR / "training_dataset.csv")
train.head()
""")
md("""
### Hasil: komposisi label

Tabel pertama: komposisi akhir (harus 60 / 20 / 20).
Tabel kedua: label per sumber data — terlihat berapa yang asli dan berapa yang sintetis.
""")
code("""
display(train["validity"].value_counts().to_frame("jumlah").assign(
    persen=lambda d: (d["jumlah"] / len(train) * 100).round(1)))
pd.crosstab(train["validity"], train["is_synthetic"].map({0: "asli", 1: "sintetis"}), margins=True)
""")
md("""
### Hasil: jenis temuan

Setiap aturan sekarang punya cukup contoh untuk dipelajari model.
""")
code("""
train["findings"].str.split(";").explode().replace("", pd.NA).dropna().value_counts().to_frame("jumlah")
""")

# ------------------------------------------------------------------ verify
md("""
## Pemeriksaan

Dua pemeriksaan penting:

1. **Label asli tidak berubah** — label setiap baris asli harus sama dengan label sebelum data sintetis
   ditambahkan (Langkah 2 tidak boleh "mengotori" data asli).
2. **Cek ulang aturan secara terpisah** — label dihitung ulang dengan kode sederhana yang **tidak memakai**
   `features.py`, lalu dibandingkan. Hasil harus 0 perbedaan.
""")
code("""
# 1. label baris asli sebelum vs sesudah data sintetis
before = real_df.set_index("session_id")[["validity", "findings"]]
after = train[train["is_synthetic"] == 0].set_index("session_id")[["validity", "findings"]].fillna("")
changed = (before.fillna("") != after.loc[before.index]).any(axis=1).sum()
print("Baris asli yang labelnya berubah:", changed)

# 2. cek ulang aturan dengan kode terpisah (brute force per tanggal)
sessions = list(csv.DictReader(open(SYN_PATH, encoding="utf-8")))
mins = lambda t: int(t[:2]) * 60 + int(t[3:])
for s in sessions:
    s["st"], s["du"], s["sub"] = mins(s["start_time"]), int(s["duration_minutes"]), tuple(sorted(json.loads(s["subject_ids"])))
by_date = defaultdict(list)
for s in sessions:
    by_date[s["lesson_date"]].append(s)
expected_label = {}
for t in sessions:
    earlier = [c for c in by_date[t["lesson_date"]] if c is not t and c["submitted_at"] < t["submitted_at"]]
    overlap = lambda c: t["st"] < c["st"] + c["du"] and c["st"] < t["st"] + t["du"]
    same5 = lambda c: c["tutor_id"] == t["tutor_id"] and c["student_id"] == t["student_id"] and c["sub"] == t["sub"] and c["du"] == t["du"]
    exact = any(same5(c) and c["st"] == t["st"] for c in earlier)
    blue = (t["du"] > 180 or t["du"] < 30 or t["st"] > 21 * 60 or t["st"] < 7 * 60
            or any(same5(c) and c["st"] != t["st"] for c in earlier)
            or any(c["tutor_id"] == t["tutor_id"] and c["student_id"] != t["student_id"] and overlap(c) for c in earlier)
            or any(c["student_id"] == t["student_id"] and c["tutor_id"] != t["tutor_id"] and overlap(c) for c in earlier)
            or any(c["tutor_id"] == t["tutor_id"] and c["student_id"] == t["student_id"] and overlap(c)
                   and not (same5(c) and c["st"] == t["st"]) for c in earlier))
    expected_label[t["id"]] = "rejected" if exact else "needs_revision" if blue else "valid"
mismatch = (train.set_index("session_id")["validity"] != pd.Series(expected_label)).sum()
print("Label yang berbeda dari cek ulang:", mismatch, "dari", len(train))
assert changed == 0 and mismatch == 0
""")

# ------------------------------------------------------------------ download
md("""
## Download hasil

Mengunduh file hasil ke komputer. Simpan di **Google Drive tim**, jangan di GitHub.
""")
code("""
outputs = [CLEAN_PATH, SYN_PATH, TRAIN_DIR / "training_dataset.csv", TRAIN_DIR / "sample_requests.json"]
try:
    from google.colab import files
    for p in outputs:
        files.download(str(p))
except ImportError:
    for p in outputs:
        print("Tersimpan:", p)
""")
md("""
## Langkah berikutnya (AI Engineer)

- Kolom fitur untuk model: `FEATURE_NAMES` (19 kolom), target: `validity`.
- Sisihkan sebagian **baris asli** (`is_synthetic == 0`) sebagai data uji agar akurasi model
  diukur pada data nyata, bukan data buatan.
- Gunakan `features.py` di FastAPI agar fitur saat *live* dihitung sama persis dengan saat *training*.

⚠️ Sebelum menyimpan notebook ini ke GitHub: **Edit → Clear all outputs**.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, OUT)
print("cells:", len(cells), "->", OUT)
