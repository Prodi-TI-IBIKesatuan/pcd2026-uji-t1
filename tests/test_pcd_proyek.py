"""Tes alat bantu pcd_proyek. Dijalankan otomatis oleh pemeriksaan cek di setiap push.

Kelompok boleh menambah tes sendiri di folder ini, asal hanya memakai numpy,
opencv-python-headless, dan pytest (pustaka lain tidak dipasang di pemeriksaan cek).
"""
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pcd_proyek import metrik, util  # noqa: E402


def test_iou_kasus_dasar():
    assert metrik.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert metrik.iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert metrik.iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert metrik.iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


def test_tingkat_deteksi_menghitung_none_sebagai_gagal():
    pred = [(0, 0, 10, 10), None, (100, 100, 5, 5)]
    acuan = [(1, 1, 10, 10), (0, 0, 10, 10), (0, 0, 10, 10)]
    assert metrik.tingkat_deteksi(pred, acuan) == pytest.approx(1 / 3)


def test_cer_pelat_tanpa_spasi_dan_huruf_kecil():
    assert metrik.cer("f 1234 ab", "F1234AB", spasi="hapus") == 0.0
    assert metrik.cer("F1234AC", "F1234AB", spasi="hapus") == pytest.approx(1 / 7)
    assert metrik.cer("", "") == 0.0 and metrik.cer("X", "") == 1.0
    assert metrik.jarak_levenshtein("kitten", "sitting") == 3


def test_cer_merapikan_spasi_teks_struk():
    assert metrik.cer("TOTAL   45.500\n", "total 45.500") == 0.0
    assert metrik.cer_rerata(["AB", "XY"], ["AB", "AB"]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        metrik.normalisasi_teks("a", spasi="aneh")


def test_akurasi_isian_angka_dibandingkan_digitnya():
    acuan = {"tanggal": "2026-10-21", "total": "45500"}
    assert metrik.akurasi_isian({"tanggal": "2026-10-21", "total": "Rp45.500"}, acuan) == 1.0
    assert metrik.akurasi_isian({"tanggal": "21/10/2026", "total": "45.500"}, acuan) == 0.5
    assert metrik.akurasi_isian({}, {"tanggal": "", "total": ""}) == 0.0


def test_psnr_dan_ssim():
    rng = np.random.default_rng(0)
    bersih = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    assert math.isinf(metrik.psnr(bersih, bersih))
    assert metrik.ssim(bersih, bersih) == pytest.approx(1.0)
    sedikit = np.clip(bersih + rng.normal(0, 5, bersih.shape), 0, 255).astype(np.uint8)
    banyak = np.clip(bersih + rng.normal(0, 40, bersih.shape), 0, 255).astype(np.uint8)
    assert metrik.psnr(sedikit, bersih) > metrik.psnr(banyak, bersih) > 0
    assert 1.0 > metrik.ssim(sedikit, bersih) > metrik.ssim(banyak, bersih)
    warna = np.dstack([bersih] * 3)
    assert metrik.ssim(warna, warna) == pytest.approx(1.0)
    assert metrik.psnr(bersih / 255.0, sedikit / 255.0) == pytest.approx(metrik.psnr(bersih, sedikit))
    with pytest.raises(ValueError):
        metrik.psnr(bersih, bersih[:10])


def test_presisi_dan_map():
    assert metrik.presisi_at_k([1, 0, 1, 0, 0], 5) == pytest.approx(0.4)
    assert metrik.presisi_at_k([1, 1], 5) == pytest.approx(0.4)
    assert metrik.average_precision([1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)
    assert metrik.average_precision([1, 0, 1], jumlah_relevan=4) == pytest.approx((1 + 2 / 3) / 4)
    assert metrik.average_precision([0, 0]) == 0.0
    assert metrik.mean_average_precision([[1, 0], [0, 1]]) == pytest.approx((1 + 0.5) / 2)
    with pytest.raises(ValueError):
        metrik.presisi_at_k([1], 0)


def test_util_baca_simpan_dan_berdampingan(tmp_path):
    citra = np.zeros((40, 80, 3), np.uint8)
    citra[:, 40:] = (0, 0, 255)
    path = util.simpan_citra(tmp_path / "sub" / "contoh ü.png", citra)
    assert np.array_equal(util.baca_citra(path), citra)
    besar = np.zeros((2000, 1000), np.uint8)
    assert util.ubah_ukuran_maks(besar, 1024).shape == (1024, 512)
    assert util.ubah_ukuran_maks(citra, 1024).shape == citra.shape
    gabung = util.berdampingan([citra, citra[:, :, 0]], ["a", "b"], tinggi=100)
    assert gabung.shape[0] == 136 and gabung.shape[2] == 3
    keluaran = util.simpan_sebelum_sesudah(citra, citra / 255.0, tmp_path / "fig.png")
    assert util.baca_citra(keluaran).shape[0] == 396
    (tmp_path / "rusak.jpg").write_bytes(b"bukan citra")
    with pytest.raises(ValueError):
        util.baca_citra(tmp_path / "rusak.jpg")


def test_util_statistik_dan_anotasi(tmp_path):
    folder = tmp_path / "citra"
    util.simpan_citra(folder / "a.png", np.zeros((10, 20, 3), np.uint8))
    util.simpan_citra(folder / "sub" / "b.png", np.zeros((30, 40, 3), np.uint8))
    (folder / "c.jpg").write_bytes(b"")
    berkas = util.daftar_citra(folder)
    assert [p.name for p in berkas] == ["a.png", "c.jpg", "b.png"]
    stat = util.statistik_dataset(berkas)
    assert stat["terbaca"] == 2 and stat["lebar"]["maks"] == 40 and len(stat["gagal"]) == 1
    csv = tmp_path / "anotasi.csv"
    csv.write_text("berkas,kategori\na.png,x\nhilang.png,y\n", encoding="utf-8")
    baris = util.baca_anotasi(csv, ["berkas", "kategori"])
    hasil = util.cek_anotasi(baris, folder, ["berkas"])
    assert hasil == {"hilang": ["hilang.png"], "tanpa_anotasi": ["c.jpg", "sub/b.png"]}
    with pytest.raises(ValueError, match="kolom anotasi kurang"):
        util.baca_anotasi(csv, ["berkas", "teks_pelat"])
    with pytest.raises(FileNotFoundError):
        util.daftar_citra(tmp_path / "tidak-ada")
