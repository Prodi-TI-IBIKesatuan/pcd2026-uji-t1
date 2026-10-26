# Deteksi dan Pengenalan Pelat Nomor Kendaraan

Proyek kelompok **Pengolahan Citra Digital 2026** · kelas `uji` · topik **T1**

[![Cek proyek](https://github.com/Prodi-TI-IBIKesatuan/pcd2026-uji-t1/actions/workflows/cek-proyek.yml/badge.svg)](https://github.com/Prodi-TI-IBIKesatuan/pcd2026-uji-t1/actions/workflows/cek-proyek.yml)

**Lingkup kelompok kami:** UJI COBA — repo ini hanya simulasi alur dengan citra sintetis dan kode dummy.

Menemukan letak pelat nomor pada foto kendaraan di lingkungan nyata, lalu membaca karakternya. Anggota dan perannya ada di [ANGGOTA.yml](ANGGOTA.yml); cara kerja kelompok ada di [PANDUAN.md](PANDUAN.md).

## Notebook

| Notebook | Tahap | Isi | Buka |
| --- | --- | --- | --- |
| `01_dataset_preprocessing.ipynb` | 1 | Statistik dataset, cek anotasi, preprocessing, gambar sebelum–sesudah | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prodi-TI-IBIKesatuan/pcd2026-uji-t1/blob/main/notebooks/01_dataset_preprocessing.ipynb) |
| `02_metode_klasik.ipynb` | 2 | 2 metode klasik dan metriknya | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prodi-TI-IBIKesatuan/pcd2026-uji-t1/blob/main/notebooks/02_metode_klasik.ipynb) |
| `03_metode_modern.ipynb` | 2 | Inferensi model pralatih dan metriknya | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prodi-TI-IBIKesatuan/pcd2026-uji-t1/blob/main/notebooks/03_metode_modern.ipynb) |
| `04_evaluasi.ipynb` | 2 | Tabel perbandingan, uji ablasi, kasus gagal | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prodi-TI-IBIKesatuan/pcd2026-uji-t1/blob/main/notebooks/04_evaluasi.ipynb) |

## Dataset (Tahap 1)

- Jumlah citra: 6 citra sintetis (uji coba; kelompok sungguhan minimal 150)
- Foto sendiri: tidak ada (uji coba)
- Sumber data tambahan dan lisensinya: tidak ada; citra dibangkitkan acak dengan NumPy
- Langkah preprocessing: DUMMY — hanya memperkecil ukuran (fungsi bawaan template)

Rincian dataset, format anotasi, dan tautan Drive: [data/README.md](data/README.md). Contoh sebelum–sesudah ada di [outputs/figures/](outputs/figures/).

## Hasil (Tahap 2)

| Metode | Jenis | IoU | CER |
| --- | --- | --- | --- |
| Dummy klasik 1 | klasik | 0.000 | 1.000 |
| Dummy klasik 2 | klasik | 0.000 | 1.000 |
| Dummy modern | modern | 0.000 | 1.000 |

- **Uji ablasi:** UJI COBA — parameter diubah 4 kali; metode dummy memberi metrik yang sama.
- **Kasus gagal:** UJI COBA — semua citra gagal karena metode dummy tidak mendeteksi pelat.
- **Kesimpulan:** UJI COBA — tidak ada kesimpulan; repo ini hanya menguji alur pengumpulan.

## Demo

Aplikasi: https://huggingface.co/spaces/contoh/pcd2026-uji-t1 (tautan fiktif untuk uji coba)

Unggah foto kendaraan, tampilkan kotak pelat dan teks yang terbaca. Kode aplikasi ada di [app/](app/).

## Sumber dan atribusi

Tuliskan setiap kode, model, dataset, atau bantuan AI yang dipakai beserta sumbernya. Pemakaian tanpa sumber dihitung plagiarisme.

| Yang dipakai | Sumber | Dipakai untuk |
| --- | --- | --- |
| Kerangka notebook dan pcd_proyek | Template kelas-pcd-2026 | Seluruh notebook (uji coba) |
