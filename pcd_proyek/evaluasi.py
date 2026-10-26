"""Evaluasi metode secara seragam untuk setiap topik.

Setiap fungsi `evaluasi_*` menjalankan sebuah metode pada data beranotasi dan
mengembalikan dict:

    {"metrik": {"IoU": 0.71, "CER": 0.18, ...},   # rata-rata, nama sama dengan README
     "rincian": [{"berkas": "a.jpg", ...}, ...]}  # skor per citra, untuk analisis kasus gagal

`anotasi` adalah list baris dari `util.baca_anotasi()`, `folder` adalah folder citra.
Argumen `batas` membatasi jumlah baris agar uji coba cepat (None = semua).
"""
from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Callable, Mapping, Sequence

import cv2
import numpy as np

from . import metrik, util


def _baris(anotasi: Sequence[Mapping[str, str]], batas: int | None):
    return list(anotasi)[:batas] if batas else list(anotasi)


def _rerata(nilai) -> float:
    nilai = [v for v in nilai if v is not None and not math.isinf(v)]
    return float(np.mean(nilai)) if nilai else float("nan")


def evaluasi_pelat(metode: Callable, anotasi, folder, batas: int | None = None, ambang: float = 0.5) -> dict:
    """T1. metode(citra_bgr) -> (kotak (x, y, w, h) atau None, teks_pelat)."""
    folder = pathlib.Path(folder)
    rincian = []
    for b in _baris(anotasi, batas):
        kotak, teks = metode(util.baca_citra(folder / b["berkas"]))
        acuan = tuple(float(b[k]) for k in ("x", "y", "w", "h"))
        nilai_iou = 0.0 if kotak is None else metrik.iou(kotak, acuan)
        rincian.append({"berkas": b["berkas"], "IoU": nilai_iou,
                        "CER": metrik.cer(teks or "", b["teks_pelat"], spasi="hapus"), "teks": teks or ""})
    return {"metrik": {"IoU": _rerata(r["IoU"] for r in rincian),
                       "CER": _rerata(r["CER"] for r in rincian),
                       f"Deteksi@{ambang:g}": _rerata(float(r["IoU"] >= ambang) for r in rincian),
                       "jumlah": len(rincian)},
            "rincian": rincian}


def evaluasi_berreferensi(metode: Callable, anotasi, folder, kolom_masukan: str, kolom_acuan: str,
                          batas: int | None = None) -> dict:
    """T2 dan T3. metode(citra_bgr) -> citra_hasil_bgr; dibandingkan dengan citra acuan.

    T2: kolom_masukan="berkas_gelap", kolom_acuan="berkas_referensi" (baris tanpa referensi dilewati).
    T3: kolom_masukan="berkas_rusak", kolom_acuan="berkas_bersih".
    Hasil di-resize ke ukuran acuan bila berbeda.
    """
    folder = pathlib.Path(folder)
    rincian, dilewati = [], 0
    for b in _baris(anotasi, batas):
        if not b.get(kolom_acuan):
            dilewati += 1
            continue
        acuan = util.baca_citra(folder / b[kolom_acuan])
        hasil = np.asarray(metode(util.baca_citra(folder / b[kolom_masukan])))
        if hasil.dtype != np.uint8:
            hasil = np.clip(hasil * (255.0 if hasil.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        if hasil.ndim == 2:
            hasil = cv2.cvtColor(hasil, cv2.COLOR_GRAY2BGR)
        if hasil.shape[:2] != acuan.shape[:2]:
            hasil = cv2.resize(hasil, (acuan.shape[1], acuan.shape[0]), interpolation=cv2.INTER_AREA)
        rincian.append({"berkas": b[kolom_masukan], "PSNR": metrik.psnr(hasil, acuan), "SSIM": metrik.ssim(hasil, acuan)})
    return {"metrik": {"PSNR": _rerata(r["PSNR"] for r in rincian), "SSIM": _rerata(r["SSIM"] for r in rincian),
                       "jumlah": len(rincian), "tanpa_acuan": dilewati},
            "rincian": rincian}


def evaluasi_ocr(metode: Callable, anotasi, folder, batas: int | None = None) -> dict:
    """T4. metode(citra_bgr) -> (teks_lengkap, {"tanggal": ..., "total": ...})."""
    folder = pathlib.Path(folder)
    rincian = []
    for b in _baris(anotasi, batas):
        teks, isian = metode(util.baca_citra(folder / b["berkas"]))
        rincian.append({"berkas": b["berkas"], "CER": metrik.cer(teks or "", b["teks_lengkap"]),
                        "Akurasi isian": metrik.akurasi_isian(isian or {}, b), "isian": dict(isian or {})})
    return {"metrik": {"CER": _rerata(r["CER"] for r in rincian),
                       "Akurasi isian": _rerata(r["Akurasi isian"] for r in rincian), "jumlah": len(rincian)},
            "rincian": rincian}


def evaluasi_temu_kembali(fitur: Callable, anotasi, folder, k: int = 5, batas: int | None = None) -> dict:
    """T5. fitur(citra_bgr) -> vektor 1D. Setiap citra menjadi query terhadap semua citra lain.

    Relevan = kategori sama. Kemiripan = kosinus.
    """
    folder = pathlib.Path(folder)
    baris = _baris(anotasi, batas)
    if len(baris) < 2:
        raise ValueError("butuh minimal 2 citra beranotasi")
    vektor = np.vstack([np.ravel(fitur(util.baca_citra(folder / b["berkas"]))).astype(np.float64) for b in baris])
    vektor /= np.linalg.norm(vektor, axis=1, keepdims=True) + 1e-12
    skor = vektor @ vektor.T
    kategori = [b["kategori"] for b in baris]
    rincian, semua_relevan = [], []
    for i, b in enumerate(baris):
        urut = [j for j in np.lexsort((np.arange(len(baris)), -skor[i])) if j != i]
        relevan = [kategori[j] == kategori[i] for j in urut]
        total = sum(relevan)
        semua_relevan.append(relevan)
        rincian.append({"berkas": b["berkas"], f"Presisi@{k}": metrik.presisi_at_k(relevan, k),
                        "AP": metrik.average_precision(relevan, total), "teratas": [baris[j]["berkas"] for j in urut[:k]]})
    return {"metrik": {f"Presisi@{k}": _rerata(r[f"Presisi@{k}"] for r in rincian),
                       "mAP": metrik.mean_average_precision(semua_relevan), "jumlah": len(rincian)},
            "rincian": rincian}


# ---------------------------------------------------------------- pelaporan


def simpan_hasil(path: str | pathlib.Path, nama: str, jenis: str, hasil: dict) -> pathlib.Path:
    """Tambahkan/ganti hasil satu metode di berkas JSON (dipakai notebook 04 untuk tabel)."""
    if jenis not in {"klasik", "modern"}:
        raise ValueError("jenis harus 'klasik' atau 'modern'")
    path = pathlib.Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[nama] = {"jenis": jenis, "metrik": hasil["metrik"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return path


def baca_hasil(path: str | pathlib.Path) -> dict:
    path = pathlib.Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _format(nilai) -> str:
    if isinstance(nilai, (int, np.integer)):
        return str(int(nilai))
    if nilai is None or (isinstance(nilai, float) and math.isnan(nilai)):
        return "–"
    if isinstance(nilai, float) and math.isinf(nilai):
        return "∞"
    return f"{float(nilai):.3f}"


def tabel_markdown(semua: Mapping[str, Mapping], kolom_metrik: Sequence[str]) -> str:
    """Baris tabel README bagian Hasil: | Metode | Jenis | metrik… |, klasik dulu lalu modern."""
    urut = sorted(semua.items(), key=lambda kv: (kv[1]["jenis"] != "klasik", kv[0]))
    baris = [f"| Metode | Jenis | {' | '.join(kolom_metrik)} |", "| --- | --- |" + " --- |" * len(kolom_metrik)]
    for nama, isi in urut:
        nilai = " | ".join(_format(isi["metrik"].get(m)) for m in kolom_metrik)
        baris.append(f"| {nama} | {isi['jenis']} | {nilai} |")
    return "\n".join(baris)


def terburuk(hasil: dict, kunci: str, n: int = 3, makin_kecil_makin_baik: bool = False) -> list[dict]:
    """n baris rincian dengan skor terburuk untuk analisis kasus gagal."""
    rincian = [r for r in hasil["rincian"] if r.get(kunci) is not None]
    return sorted(rincian, key=lambda r: r[kunci], reverse=makin_kecil_makin_baik)[:n]
