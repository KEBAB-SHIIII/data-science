# data-science — Bimbel Mumtaz

Repo tim Data Science untuk sistem absensi Bimbel Mumtaz Sampit (capstone KEBAB-SHIIII).

## Isi repo

| Folder | Isi | Dipakai oleh |
|---|---|---|
| [`ai_training/`](ai_training/) | Menyiapkan data *training* model validasi absen (`valid` / `needs_revision` / `rejected`): skrip, notebook Colab, dokumentasi lengkap ([README](ai_training/README.md)) | AI Engineer — repo [`AI-MODEL-MUMTAZ`](https://github.com/KEBAB-SHIIII/AI-MODEL-MUMTAZ) |
| [`dashboard_data/`](dashboard_data/) | Membersihkan data mentah untuk diisi ke dashboard ([README](dashboard_data/README.md)) | Full Stack — repo [`mumtaz-bimbel-dashboard`](https://github.com/KEBAB-SHIIII/mumtaz-bimbel-dashboard) |
| `data/` | Tempat data lokal. **Tidak pernah ikut ter-*push*** | — |

Struktur folder `data/` di laptop masing-masing:

```
data/
├── raw/              ← data mentah (ambil dari Google Drive tim)
├── ai_training/      ← hasil ai_training/
└── dashboard_data/   ← hasil dashboard_data/
```

## Mulai cepat — `ai_training`

```bash
pip install -r ai_training/requirements.txt
# letakkan data_ai.csv di data/raw/, lalu:
python ai_training/fix_data_ai.py data/raw/data_ai.csv data/ai_training/data_ai_clean.csv data/ai_training/data_ai_issues.csv
python ai_training/inject_anomalies.py data/ai_training/data_ai_clean.csv data/ai_training/sessions_with_synthetic.csv
python ai_training/build_training_set.py data/ai_training/sessions_with_synthetic.csv data/ai_training/training
```

Atau buka `ai_training/pipeline_colab.ipynb` di Google Colab dan pilih **Runtime → Run all**.

## Aturan kerja bersama

- Setiap orang bekerja di **folder masing-masing**. File di root (`README.md`, `.gitignore`)
  dipakai bersama — `git pull` dulu sebelum mengubahnya.
- Selalu `git pull` sebelum mulai bekerja, lalu commit & push sedikit-sedikit.
- **ID harus konsisten:** dashboard dan model AI harus memakai `tutor_id`, `student_id`, dan
  `subject_ids` yang sama untuk orang/mapel yang sama, karena model membandingkan sesi berdasarkan ID.

## ⚠️ Data

Data absen berisi catatan les anak-anak yang asli. Jangan commit file data, file `lookup_*.csv`,
`config.local.json`, atau notebook yang masih berisi output. Bagikan data lewat Google Drive tim.
