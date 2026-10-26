"""Tes evaluator pcd_proyek.evaluasi dengan data sintetis kecil."""
import math
import pathlib
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pcd_proyek import evaluasi, util  # noqa: E402


def _simpan(folder, nama, citra):
    util.simpan_citra(folder / nama, citra)
    return nama


def test_evaluasi_pelat(tmp_path):
    _simpan(tmp_path, "a.png", np.zeros((50, 80, 3), np.uint8))
    _simpan(tmp_path, "b.png", np.zeros((50, 80, 3), np.uint8))
    anotasi = [{"berkas": "a.png", "x": "10", "y": "10", "w": "20", "h": "10", "teks_pelat": "F1234AB"},
               {"berkas": "b.png", "x": "0", "y": "0", "w": "10", "h": "10", "teks_pelat": "B1AB"}]
    jawaban = iter([((10, 10, 20, 10), "f 1234 ab"), (None, "")])
    hasil = evaluasi.evaluasi_pelat(lambda c: next(jawaban), anotasi, tmp_path)
    m = hasil["metrik"]
    assert m["IoU"] == pytest.approx(0.5) and m["Deteksi@0.5"] == pytest.approx(0.5)
    assert m["CER"] == pytest.approx(0.5) and m["jumlah"] == 2


def test_evaluasi_berreferensi_resize_dan_lewati(tmp_path):
    rng = np.random.default_rng(1)
    acuan = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    _simpan(tmp_path, "ref.png", acuan)
    _simpan(tmp_path, "gelap.png", (acuan // 4).astype(np.uint8))
    anotasi = [{"berkas_gelap": "gelap.png", "berkas_referensi": "ref.png"},
               {"berkas_gelap": "gelap.png", "berkas_referensi": ""}]
    sempurna = evaluasi.evaluasi_berreferensi(lambda c: cv2.resize(acuan, (20, 20)), anotasi, tmp_path,
                                              "berkas_gelap", "berkas_referensi")
    assert sempurna["metrik"]["jumlah"] == 1 and sempurna["metrik"]["tanpa_acuan"] == 1
    identik = evaluasi.evaluasi_berreferensi(lambda c: acuan / 255.0, anotasi[:1], tmp_path, "berkas_gelap", "berkas_referensi")
    assert math.isinf(identik["rincian"][0]["PSNR"]) and identik["metrik"]["SSIM"] == pytest.approx(1.0)
    abu = evaluasi.evaluasi_berreferensi(lambda c: cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), anotasi[:1], tmp_path,
                                         "berkas_gelap", "berkas_referensi")
    assert 0 < abu["metrik"]["PSNR"] < 30


def test_evaluasi_ocr(tmp_path):
    _simpan(tmp_path, "s.png", np.zeros((10, 10, 3), np.uint8))
    anotasi = [{"berkas": "s.png", "teks_lengkap": "TOTAL 45.500", "tanggal": "2026-10-21", "total": "45500"}]
    hasil = evaluasi.evaluasi_ocr(lambda c: ("total 45.500", {"tanggal": "2026-10-21", "total": "Rp 45.500"}), anotasi, tmp_path)
    assert hasil["metrik"]["CER"] == 0.0 and hasil["metrik"]["Akurasi isian"] == 1.0


def test_evaluasi_temu_kembali_dan_tabel(tmp_path):
    anotasi = []
    for i, (warna, kategori) in enumerate([((0, 0, 255), "merah"), ((10, 0, 240), "merah"),
                                           ((255, 0, 0), "biru"), ((240, 10, 0), "biru")]):
        citra = np.zeros((16, 16, 3), np.uint8)
        citra[:] = warna
        anotasi.append({"berkas": _simpan(tmp_path, f"c{i}.png", citra), "kategori": kategori})

    def rerata_warna(c):
        return c.reshape(-1, 3).mean(axis=0)

    hasil = evaluasi.evaluasi_temu_kembali(rerata_warna, anotasi, tmp_path, k=1)
    assert hasil["metrik"]["Presisi@1"] == 1.0 and hasil["metrik"]["mAP"] == 1.0
    assert hasil["rincian"][0]["teratas"] == ["c1.png"]
    with pytest.raises(ValueError):
        evaluasi.evaluasi_temu_kembali(rerata_warna, anotasi[:1], tmp_path)

    path = tmp_path / "hasil.json"
    evaluasi.simpan_hasil(path, "Histogram", "klasik", {"metrik": {"Presisi@5": 0.5, "mAP": float("nan")}})
    evaluasi.simpan_hasil(path, "CLIP", "modern", {"metrik": {"Presisi@5": np.float64(0.9), "mAP": 0.8}})
    evaluasi.simpan_hasil(path, "Akhir", "klasik", {"metrik": {"Presisi@5": 1, "mAP": float("inf")}})
    tabel = evaluasi.tabel_markdown(evaluasi.baca_hasil(path), ["Presisi@5", "mAP"]).splitlines()
    assert tabel[0] == "| Metode | Jenis | Presisi@5 | mAP |"
    assert tabel[2:] == ["| Akhir | klasik | 1 | ∞ |", "| Histogram | klasik | 0.500 | – |", "| CLIP | modern | 0.900 | 0.800 |"]
    with pytest.raises(ValueError):
        evaluasi.simpan_hasil(path, "X", "lain", {"metrik": {}})
    assert evaluasi.baca_hasil(tmp_path / "tidak-ada.json") == {}


def test_terburuk():
    hasil = {"rincian": [{"berkas": "a", "CER": 0.1}, {"berkas": "b", "CER": 0.9}, {"berkas": "c", "CER": 0.5}]}
    assert [r["berkas"] for r in evaluasi.terburuk(hasil, "CER", 2, makin_kecil_makin_baik=True)] == ["b", "c"]
    assert [r["berkas"] for r in evaluasi.terburuk(hasil, "CER", 1)] == ["a"]
