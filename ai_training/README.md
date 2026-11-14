# AI Training — Data Model Validasi Absen

Skrip Data Scientist (DS) untuk menyiapkan data *training* model validasi absen.
Acuan: `docs/IMPLEMENTASI AI MODEL + COMPARISON.md` di repo `mumtaz-bimbel-dashboard` (branch `feat/ui-ux-polish`).

Semua data sesi memakai 8 *field* sesi pada **section 3.2**:
`id, tutor_id, student_id, subject_ids, lesson_date, start_time, duration_minutes, submitted_at`

## ⚠️ Data tidak di-commit

Semua repo KEBAB-SHIIII bersifat **publik**. Folder `data/` sudah di-*ignore* oleh Git,
jadi file CSV/Excel tidak akan ikut ter-*push*. Bagikan data lewat Google Drive.

| Yang di-commit ✅ | Yang TIDAK di-commit ❌ |
|---|---|
| Skrip `.py`, README, `config.example.json` | Isi folder `data/` (CSV, Excel) |
| `pipeline_colab.ipynb` — **tanpa output** (Edit → Clear all outputs) | Notebook yang masih berisi output (tabel pratinjau berisi data asli) |
| | `config.local.json` (berisi nama anak) |
| | File `lookup_*.csv` (berisi nama asli tutor & murid) |

## Setup

```bash
pip install -r ai_training/requirements.txt
```

Letakkan file data mentah di `data/raw/` (ambil dari Google Drive tim). Hasil disimpan di `data/ai_training/`.

## Alur lengkap (jalankan dari root repo)

```bash
# 1. Rapikan format data mentah -> 8 field section 3.2
python ai_training/fix_data_ai.py data/raw/data_ai.csv data/ai_training/data_ai_clean.csv data/ai_training/data_ai_issues.csv
# 2. Tambahkan baris anomali sintetis
python ai_training/inject_anomalies.py data/ai_training/data_ai_clean.csv data/ai_training/sessions_with_synthetic.csv
# 3. Susun comparisons, hitung 19 fitur, beri label
python ai_training/build_training_set.py data/ai_training/sessions_with_synthetic.csv data/ai_training/training
```

Hasil akhir: `data/ai_training/training/training_dataset.csv` (siap untuk `model.fit(X, y)`)
dan `sample_requests.json` (contoh payload `{target, comparisons, config}` untuk menguji FastAPI).

### Versi Google Colab

`pipeline_colab.ipynb` berisi alur yang sama (kode disalin dari skrip `.py`), dipecah per bagian
dan diberi penjelasan. Buka di Colab (**File → Upload notebook**), lalu **Runtime → Run all** dan
upload `data_ai.csv` saat diminta. Notebook juga menjalankan pemeriksaan label di akhir.
⚠️ Sebelum commit notebook: **Edit → Clear all outputs** (output berisi data asli).

**Sumber utama adalah skrip `.py`, bukan notebook.** Notebook hanya salinan untuk dibaca dan
dijalankan di Colab. `features.py` juga dipakai FastAPI, jadi perubahan yang hanya dibuat di
notebook **tidak** sampai ke skrip maupun aplikasi. Cara mengubah kode:

1. Ubah file `.py` yang bersangkutan.
2. Jalankan ulang ketiga skrip (atau notebook) dan pastikan hasilnya benar.
3. Buat ulang notebook agar ikut terbarui:

```bash
pip install nbformat
python ai_training/make_notebook.py
```

Jangan mengedit kode di dalam notebook secara manual — perubahan itu akan tertimpa saat
`make_notebook.py` dijalankan.

## Aturan bisnis (doc section 1.1 + koreksi AE 14 Nov 2026 + aturan bentrok jadwal)

| Label | Kondisi |
|---|---|
| 🔴 `rejected` | Duplikat persis: `tutor_id`, `student_id`, `subject_ids`, `lesson_date`, **`start_time`**, `duration_minutes` sama dengan sesi yang di-*submit* lebih dulu |
| 🔵 `needs_revision` | Durasi `> 180` **atau `< 30`** menit · mulai `> 21:00` **atau `< 07:00`** · **duplikat samar** (5 atribut sama, `start_time` beda) · **tutor punya sesi tumpang tindih dengan murid lain** · **murid punya sesi tumpang tindih dengan tutor lain** · **tutor dan murid yang sama punya sesi tumpang tindih yang bukan duplikat persis** (mis. jam mulai sama tapi durasi beda, jam sama tapi mapel beda) |
| 🟢 `valid` | Tidak ada temuan. Tepat 180/30 menit dan tepat 21:00/07:00 masih `valid` |

Prioritas: `rejected` > `needs_revision` > `valid`; semua temuan tetap ditulis di kolom `findings`.
`subject_ids` boleh berisi **1–5** mapel dan tidak ada yang dibuang.

## Skrip

### `features.py` — dipakai bersama oleh training **dan FastAPI**

`compute_features(target, comparisons, config)` dan `label_from_features(features)`.
Python murni tanpa dependensi, sehingga AI Engineer bisa menyalin file ini ke repo `AI-MODEL-MUMTAZ` dan meng-*import*-nya di `app/`
agar fitur saat *training* dan *live* identik (section 4.5).

**19 fitur** = 13 fitur di section 3.3 (no. 1–13) + 6 fitur baru (no. 14–19, ditandai 🆕).
Urutan kolom di bawah adalah urutan input model (`FEATURE_NAMES` di `features.py`);
mengubah urutan berarti model harus dilatih ulang.

Fitur "batas" (no. 2, 4, 14, 15) memakai aturan section 3.4: **nol atau minus = aman**,
**positif = melanggar batas**. Semua fitur "jumlah" dihitung dari sesi di `comparisons`
(sesi yang di-*submit* lebih dulu, ±30 hari, tutor atau murid sama).

#### A. Dari `target` & `config`

| No | Fitur | Rumus | Arti |
|---|---|---|---|
| 1 | `duration_minutes` | `target.duration_minutes` | Lama les dalam menit |
| 2 | `duration_over_limit` | `duration_minutes - 180` | Positif = durasi > 180 menit |
| 3 | `start_minute` | `jam × 60 + menit` dari `target.start_time` | Jam mulai dalam menit sejak 00:00 (14:00 = 840) |
| 4 | `start_over_limit` | `start_minute - 1260` (21:00 = 1260) | Positif = mulai setelah 21:00 |
| 5 | `subject_count` | jumlah item `target.subject_ids` | Banyaknya mapel (1–5) |
| 14 🆕 | `duration_under_limit` | `30 - duration_minutes` | Positif = durasi < 30 menit |
| 15 🆕 | `start_under_limit` | `420 - start_minute` (07:00 = 420) | Positif = mulai sebelum 07:00 |

#### B. Dari perbandingan `target` dengan `comparisons`

| No | Fitur | Cara menghitung | Arti |
|---|---|---|---|
| 6 | `comparisons_count` | jumlah sesi di `comparisons` | Banyaknya riwayat 30 hari |
| 7 | `exact_dup_count` | sesi dengan tutor, murid, mapel, tanggal, **jam mulai**, dan durasi sama persis | **≥ 1 → `rejected`** (duplikat persis) |
| 8 | `same_tutor_student_date_count` | sesi dengan tutor, murid, dan tanggal sama (mapel/durasi boleh beda) | Konteks: input ganda di hari yang sama |
| 9 | `same_student_date_diff_tutor_count` | sesi dengan murid dan tanggal sama, mapel beririsan, tutor beda | Konteks: satu murid diinput dua tutor |
| 10 | `time_overlap_count` | sesi di tanggal sama yang jamnya bertabrakan, tutor **atau** murid sama | Konteks: jadwal bertabrakan (semua jenis) |
| 11 | `tutor_sessions_30d` | sesi milik tutor yang sama | Keaktifan tutor sebulan |
| 12 | `student_sessions_30d` | sesi milik murid yang sama | Frekuensi les murid sebulan |
| 13 | `duration_diff_from_tutor_mean_30d` | `duration_minutes` − rata-rata durasi tutor tsb. (0 bila tidak ada riwayat) | Seberapa menyimpang dari kebiasaan tutor |
| 16 🆕 | `fuzzy_dup_count` | sesi dengan 5 atribut sama, **jam mulai beda** | ≥ 1 → `needs_revision` (duplikat samar) |
| 17 🆕 | `tutor_overlap_diff_student_count` | sesi tutor yang sama, **murid lain**, jam bertabrakan | ≥ 1 → `needs_revision` (jadwal tutor bentrok) |
| 18 🆕 | `student_overlap_diff_tutor_count` | sesi murid yang sama, **tutor lain**, jam bertabrakan | ≥ 1 → `needs_revision` (jadwal murid bentrok) |
| 19 🆕 | `self_overlap_count` | sesi tutor **dan** murid yang sama, jam bertabrakan, bukan duplikat persis | ≥ 1 → `needs_revision` (duplikat samar jenis lain) |

Fitur no. 8–13 tidak langsung menentukan label, tetapi memberi konteks tambahan bagi model.

Perubahan dari doc: `exact_dup_count` (no. 7) sekarang juga membandingkan `start_time`,
dan `subject_count` (no. 5) bisa bernilai sampai 5.
`config` mendapat 2 kunci baru: `durationMinMinutes` (30) dan `earlyStartBefore` ("07:00").

### `fix_data_ai.py` — merapikan `data_ai.csv`

Perbaikan format saja, **tidak ada baris yang dihapus**: `lesson_date` → `YYYY-MM-DD`,
`start_time` → `HH:MM`, `subject_ids` diurutkan, kolom di luar 8 field dibuang.
`data_ai_issues.csv` hanya berisi masalah format (kasus aturan bisnis diberi label di langkah 3).

### `inject_anomalies.py` — baris anomali sintetis (section 5.5)

Data asli ~94% `valid`, jadi baris sintetis ditambahkan sampai komposisi akhir sesuai
`TARGET_SHARE` (bawaan **60% valid / 20% needs_revision / 20% rejected**). Jumlah per skenario
**dihitung otomatis** dari label data asli, jadi komposisi tetap tepat walau data asli berubah.

| `scenario` | Jumlah (data saat ini) | Label |
|---|---|---|
| `durasi_panjang` (181–300 mnt) | 128 | needs_revision |
| `durasi_pendek` (10–29 mnt) | 128 | needs_revision |
| `mulai_malam` (21:01–22:30) | 128 | needs_revision |
| `mulai_pagi` (05:00–06:59) | 128 | needs_revision |
| `duplikat_samar` | 128 | needs_revision |
| `tumpang_tindih` (tutor bentrok) | 128 | needs_revision |
| `tumpang_tindih_murid` (murid bentrok) | 128 | needs_revision |
| `tumpang_tindih_sendiri` (tutor + murid sama bentrok: durasi beda / mapel beda / jam bergeser) | 128 | needs_revision |
| `duplikat` | 1.273 | rejected |
| `kombinasi_dup_panjang` / `_pendek` / `_malam` / `_pagi` (pasangan) | 4 × (50 + 50) | needs_revision + rejected |
| `batas_valid` (tepat 180/30 mnt, tepat 21:00/07:00) | 100 | valid |

Hasil saat ini: 4.760 asli + 2.797 sintetis = 7.557 baris → 4.534 valid / 1.511 needs_revision / 1.512 rejected.
Ubah `TARGET_SHARE` atau `FIXED_COUNTS` di bagian atas skrip untuk komposisi lain.

Setiap baris sintetis dibuat dari sesi asli yang `valid`, ditandai `is_synthetic=1`, dan ditempatkan
sehingga **tidak mengubah label baris asli mana pun**. Seed tetap → hasil selalu sama.

### `build_training_set.py` — comparisons + fitur + label

`comparisons` disusun seperti query *backend* (section 3.1): tutor **atau** murid sama, ±30 hari,
`submitted_at` lebih awal, maks 100 sesi. Bila kandidat > 100, yang diambil adalah sesi dengan
`lesson_date` terdekat (sesi di tanggal yang sama selalu ikut). **Query backend sebaiknya memakai
urutan yang sama.**

### `clean_absen.py` — dari Excel absen mentah (format per-sheet murid)

```bash
python ai_training/clean_absen.py "data/raw/10. Absen Bimbel Oktober 2026.xlsx" data/ai_training/absen
```

Memperbaiki tanggal yang tertukar oleh Excel, memecah `Durasi Les` menjadi `start_time` +
`duration_minutes`, memetakan teks mapel ke mapel standar, membuat UUID tetap, dan membuat
`submitted_at` sintetis. Untuk murid yang tercatat di dua *sheet*, salin `config.example.json`
menjadi `config.local.json` lalu isi `sheet_aliases`.
