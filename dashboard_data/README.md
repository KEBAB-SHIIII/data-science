# Dashboard Data — Pembersihan Data untuk Dashboard

> Folder ini disiapkan untuk pekerjaan pembersihan data mentah yang akan diisi ke dashboard
> (repo `mumtaz-bimbel-dashboard`). Silakan ganti isi README ini.

## Yang perlu dijelaskan di sini

- **Sumber data:** file mentah apa yang dipakai (letakkan di `data/raw/`).
- **Cara menjalankan:** perintah untuk setiap skrip/notebook.
- **Hasil:** file apa yang dihasilkan (simpan di `data/dashboard_data/`) dan formatnya.
- **Aturan pembersihan:** apa saja yang diperbaiki, dibuang, atau ditandai.

## Catatan

- `ai_training/clean_absen.py` sudah bisa membersihkan Excel absen per-*sheet* murid (format
  "Absen Bimbel Oktober 2026") dan membuat UUID tetap untuk tutor, murid, dan mapel.
  Bisa dipakai ulang atau dipindah ke folder ini — diskusikan dengan pemilik `ai_training/`.
- `tutor_id`, `student_id`, dan `subject_ids` harus sama dengan yang dipakai `ai_training/`.
- Jangan commit data asli; folder `data/` sudah di-*ignore*.
