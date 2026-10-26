"""Metrik evaluasi untuk kelima topik proyek.

| Topik | Metrik |
| --- | --- |
| T1 Pelat nomor | iou, cer (spasi="hapus") |
| T2 Citra malam | psnr, ssim |
| T3 Restorasi | psnr, ssim |
| T4 OCR struk | cer, akurasi_isian |
| T5 Citra serupa | presisi_at_k, mean_average_precision |
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------- deteksi


def iou(kotak_a: Sequence[float], kotak_b: Sequence[float]) -> float:
    """Intersection over Union dua kotak (x, y, lebar, tinggi). Hasil 0–1."""
    ax, ay, aw, ah = (float(v) for v in kotak_a)
    bx, by, bw, bh = (float(v) for v in kotak_b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    lebar = min(ax + aw, bx + bw) - max(ax, bx)
    tinggi = min(ay + ah, by + bh) - max(ay, by)
    irisan = max(0.0, lebar) * max(0.0, tinggi)
    gabungan = aw * ah + bw * bh - irisan
    return irisan / gabungan if gabungan > 0 else 0.0


def tingkat_deteksi(prediksi: Iterable[Sequence[float] | None], acuan: Iterable[Sequence[float]], ambang: float = 0.5) -> float:
    """Fraksi citra yang kotak prediksinya memiliki IoU >= ambang terhadap acuan.

    `prediksi` boleh berisi None untuk citra yang tidak terdeteksi apa pun.
    """
    pasangan = list(zip(prediksi, acuan, strict=True))
    if not pasangan:
        return 0.0
    benar = sum(1 for p, a in pasangan if p is not None and iou(p, a) >= ambang)
    return benar / len(pasangan)


# ---------------------------------------------------------------- teks


def normalisasi_teks(teks: str, spasi: str = "rapikan", huruf_besar: bool = True) -> str:
    """Seragamkan teks sebelum dibandingkan.

    spasi: "hapus" (untuk pelat nomor), "rapikan" (spasi ganda jadi satu), atau "biarkan".
    """
    if spasi not in {"hapus", "rapikan", "biarkan"}:
        raise ValueError("spasi harus 'hapus', 'rapikan', atau 'biarkan'")
    teks = "" if teks is None else str(teks)
    if huruf_besar:
        teks = teks.upper()
    if spasi == "hapus":
        teks = re.sub(r"\s+", "", teks)
    elif spasi == "rapikan":
        teks = re.sub(r"\s+", " ", teks).strip()
    return teks


def jarak_levenshtein(a: str, b: str) -> int:
    """Jumlah minimum sisip, hapus, atau ganti karakter untuk mengubah a menjadi b."""
    if len(a) < len(b):
        a, b = b, a
    sebelumnya = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        sekarang = [i]
        for j, cb in enumerate(b, 1):
            sekarang.append(min(sebelumnya[j] + 1, sekarang[j - 1] + 1, sebelumnya[j - 1] + (ca != cb)))
        sebelumnya = sekarang
    return sebelumnya[-1]


def cer(prediksi: str, acuan: str, spasi: str = "rapikan", huruf_besar: bool = True) -> float:
    """Character error rate = jarak Levenshtein / panjang acuan. 0 = sempurna, bisa > 1."""
    p = normalisasi_teks(prediksi, spasi, huruf_besar)
    a = normalisasi_teks(acuan, spasi, huruf_besar)
    if not a:
        return 0.0 if not p else 1.0
    return jarak_levenshtein(p, a) / len(a)


def cer_rerata(prediksi: Iterable[str], acuan: Iterable[str], **opsi) -> float:
    """Rata-rata CER pada banyak pasangan teks."""
    nilai = [cer(p, a, **opsi) for p, a in zip(prediksi, acuan, strict=True)]
    return float(np.mean(nilai)) if nilai else 0.0


def _seragamkan_isian(nilai, angka: bool) -> str:
    teks = "" if nilai is None else str(nilai)
    if angka:
        return re.sub(r"\D", "", teks)
    return re.sub(r"\s+", "", teks).lower()


def akurasi_isian(prediksi: Mapping[str, object], acuan: Mapping[str, object],
                  kunci: Sequence[str] = ("tanggal", "total"), angka: Sequence[str] = ("total",)) -> float:
    """Fraksi isian (mis. tanggal dan total) yang terbaca sama dengan acuan.

    Isian di `angka` dibandingkan hanya pada digitnya, sehingga "Rp45.500" sama dengan "45500".
    """
    if not kunci:
        return 0.0
    benar = 0
    for k in kunci:
        sama_angka = k in angka
        p = _seragamkan_isian(prediksi.get(k), sama_angka)
        a = _seragamkan_isian(acuan.get(k), sama_angka)
        benar += int(bool(a) and p == a)
    return benar / len(kunci)


# ---------------------------------------------------------------- kualitas citra


def _ke_float(citra: np.ndarray, maks: float | None) -> tuple[np.ndarray, float]:
    citra = np.asarray(citra)
    if maks is None:
        maks = 255.0 if citra.dtype == np.uint8 else 1.0
    return citra.astype(np.float64), float(maks)


def psnr(citra: np.ndarray, acuan: np.ndarray, maks: float | None = None) -> float:
    """Peak signal-to-noise ratio dalam dB. Citra identik menghasilkan inf.

    maks: nilai piksel tertinggi; bawaannya 255 untuk uint8 dan 1.0 untuk float.
    """
    if np.shape(citra) != np.shape(acuan):
        raise ValueError(f"ukuran berbeda: {np.shape(citra)} vs {np.shape(acuan)}")
    a, maks = _ke_float(citra, maks)
    b, _ = _ke_float(acuan, maks)
    mse = float(np.mean((a - b) ** 2))
    return math.inf if mse == 0 else 10.0 * math.log10(maks**2 / mse)


def ssim(citra: np.ndarray, acuan: np.ndarray, maks: float | None = None) -> float:
    """Structural similarity (Wang dkk., 2004) dengan jendela Gauss 11x11, sigma 1,5.

    Citra berwarna dihitung per kanal lalu dirata-rata. Hasil 1 = identik.
    """
    if np.shape(citra) != np.shape(acuan):
        raise ValueError(f"ukuran berbeda: {np.shape(citra)} vs {np.shape(acuan)}")
    x, maks = _ke_float(citra, maks)
    y, _ = _ke_float(acuan, maks)
    if x.ndim == 3:
        return float(np.mean([ssim(x[..., k], y[..., k], maks) for k in range(x.shape[2])]))
    c1, c2 = (0.01 * maks) ** 2, (0.03 * maks) ** 2

    def blur(m):
        return cv2.GaussianBlur(m, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)

    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cxy = blur(x * y) - mx * my
    peta = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(np.mean(peta))


# ---------------------------------------------------------------- temu kembali


def presisi_at_k(relevan_terurut: Sequence[bool], k: int = 5) -> float:
    """Fraksi k hasil teratas yang relevan. `relevan_terurut[i]` = hasil peringkat ke-i relevan?"""
    if k <= 0:
        raise ValueError("k harus > 0")
    return sum(bool(r) for r in list(relevan_terurut)[:k]) / k


def average_precision(relevan_terurut: Sequence[bool], jumlah_relevan: int | None = None) -> float:
    """Average precision satu query.

    jumlah_relevan: total citra relevan di basis data (bawaannya: yang muncul di daftar).
    """
    relevan = [bool(r) for r in relevan_terurut]
    total = sum(relevan) if jumlah_relevan is None else jumlah_relevan
    if total == 0:
        return 0.0
    jumlah, ditemukan = 0.0, 0
    for i, r in enumerate(relevan, 1):
        if r:
            ditemukan += 1
            jumlah += ditemukan / i
    return jumlah / total


def mean_average_precision(semua_query: Iterable[Sequence[bool]], jumlah_relevan: Iterable[int] | None = None) -> float:
    """Rata-rata average precision dari banyak query."""
    daftar = list(semua_query)
    if not daftar:
        return 0.0
    totals = [None] * len(daftar) if jumlah_relevan is None else list(jumlah_relevan)
    return float(np.mean([average_precision(q, t) for q, t in zip(daftar, totals, strict=True)]))
