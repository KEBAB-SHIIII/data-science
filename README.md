# data-science — Bimbel Mumtaz

Repo tim Data Science untuk sistem absensi Bimbel Mumtaz Sampit (capstone KEBAB-SHIIII).
Berisi skrip untuk membersihkan data absen dan menyiapkan data *training* model validasi absen
(`valid` / `needs_revision` / `rejected`) yang dipakai layanan AI di repo
[`AI-MODEL-MUMTAZ`](https://github.com/KEBAB-SHIIII/AI-MODEL-MUMTAZ).

## Isi repo

| Folder / file | Isi |
|---|---|
| [`data_pipeline/`](data_pipeline/) | Skrip pipeline, notebook Colab, dan **dokumentasi lengkap** ([README](data_pipeline/README.md)) |
| `data/` | Tempat data lokal (`raw/`, `processed/`). **Tidak pernah ikut ter-*push*** |

## Mulai cepat

```bash
pip install -r data_pipeline/requirements.txt
# letakkan data_ai.csv di data/raw/ (ambil dari Google Drive tim), lalu:
python data_pipeline/fix_data_ai.py data/raw/data_ai.csv data/processed/data_ai_clean.csv data/processed/data_ai_issues.csv
python data_pipeline/inject_anomalies.py data/processed/data_ai_clean.csv data/processed/sessions_with_synthetic.csv
python data_pipeline/build_training_set.py data/processed/sessions_with_synthetic.csv data/processed/training
```

Atau buka `data_pipeline/pipeline_colab.ipynb` di Google Colab dan pilih **Runtime → Run all**.

Hasil: `data/processed/training/training_dataset.csv` (siap untuk *training*) dan
`sample_requests.json` (contoh request untuk menguji FastAPI).

## ⚠️ Data

Data absen berisi catatan les anak-anak yang asli. Jangan commit file data, file `lookup_*.csv`,
`config.local.json`, atau notebook yang masih berisi output. Bagikan data lewat Google Drive tim.
