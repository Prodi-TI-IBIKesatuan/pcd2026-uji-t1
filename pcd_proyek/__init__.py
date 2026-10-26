"""Alat bantu bersama proyek kelompok PCD 2026: metrik evaluasi dan utilitas data.

Notebook di folder notebooks/ mengimpor paket ini setelah menyalin repo ke Colab:

    from pcd_proyek import evaluasi, metrik, util

Kode di sini sudah diuji (folder tests/). Kelompok boleh menambah fungsi sendiri
di modul baru, tetapi sebaiknya tidak mengubah fungsi metrik dan evaluasi agar
hasil antarkelompok tetap dapat dibandingkan.
"""
from . import evaluasi, metrik, util

__all__ = ["evaluasi", "metrik", "util"]
