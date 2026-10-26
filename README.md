# Deteksi dan Pengenalan Pelat Nomor Kendaraan

Proyek kelompok **Pengolahan Citra Digital 2026** · kelas `uji` · topik **T1**

[![Cek proyek](https://github.com/Prodi-TI-IBIKesatuan/pcd2026-uji-t1/actions/workflows/cek-proyek.yml/badge.svg)](https://github.com/Prodi-TI-IBIKesatuan/pcd2026-uji-t1/actions/workflows/cek-proyek.yml)

**Lingkup kelompok kami:** [isi: satu kalimat, mis. "pelat motor"]

Menemukan letak pelat nomor pada foto kendaraan di lingkungan nyata, lalu membaca karakternya. Anggota dan perannya ada di [ANGGOTA.yml](ANGGOTA.yml); cara kerja kelompok ada di [PANDUAN.md](PANDUAN.md).

## Notebook

| Notebook | Tahap | Isi | Buka |
| --- | --- | --- | --- |
| `01_dataset_preprocessing.ipynb` | 1 | Statistik dataset, cek anotasi, preprocessing, gambar sebelum–sesudah | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prodi-TI-IBIKesatuan/pcd2026-uji-t1/blob/main/notebooks/01_dataset_preprocessing.ipynb) |
| `02_metode_klasik.ipynb` | 2 | 2 metode klasik dan metriknya | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prodi-TI-IBIKesatuan/pcd2026-uji-t1/blob/main/notebooks/02_metode_klasik.ipynb) |
| `03_metode_modern.ipynb` | 2 | Inferensi model pralatih dan metriknya | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prodi-TI-IBIKesatuan/pcd2026-uji-t1/blob/main/notebooks/03_metode_modern.ipynb) |
| `04_evaluasi.ipynb` | 2 | Tabel perbandingan, uji ablasi, kasus gagal | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prodi-TI-IBIKesatuan/pcd2026-uji-t1/blob/main/notebooks/04_evaluasi.ipynb) |

## Dataset (Tahap 1)

- Jumlah citra: [isi, minimal 150]
- Foto sendiri: [isi, minimal 45, 15 per anggota]
- Sumber data tambahan dan lisensinya: [isi, atau tulis "tidak ada"]
- Langkah preprocessing: [isi, ringkas dan berurutan]

Rincian dataset, format anotasi, dan tautan Drive: [data/README.md](data/README.md). Contoh sebelum–sesudah ada di [outputs/figures/](outputs/figures/).

## Hasil (Tahap 2)

| Metode | Jenis | IoU | CER |
| --- | --- | --- | --- |
| [isi: nama metode klasik 1] | klasik | [isi] | [isi] |
| [isi: nama metode klasik 2] | klasik | [isi] | [isi] |
| [isi: nama model] | modern | [isi] | [isi] |

- **Uji ablasi:** [isi: parameter atau langkah yang diubah, dan pengaruhnya pada metrik]
- **Kasus gagal:** [isi: minimal 3 contoh dan penyebabnya, gambar di outputs/figures/]
- **Kesimpulan:** [isi: metode mana yang terbaik untuk lingkup kami dan mengapa]

## Demo

Aplikasi: [isi: tautan https://huggingface.co/spaces/...]

Unggah foto kendaraan, tampilkan kotak pelat dan teks yang terbaca. Kode aplikasi ada di [app/](app/).

## Sumber dan atribusi

Tuliskan setiap kode, model, dataset, atau bantuan AI yang dipakai beserta sumbernya. Pemakaian tanpa sumber dihitung plagiarisme.

| Yang dipakai | Sumber | Dipakai untuk |
| --- | --- | --- |
| [isi] | [isi] | [isi] |
