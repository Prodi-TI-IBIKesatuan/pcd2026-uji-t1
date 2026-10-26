"""Utilitas data: membaca citra dan anotasi, statistik dataset, dan gambar sebelum–sesudah."""
from __future__ import annotations

import csv
import pathlib
from collections.abc import Sequence

import cv2
import numpy as np

EKSTENSI = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def daftar_citra(folder: str | pathlib.Path) -> list[pathlib.Path]:
    """Semua berkas citra di folder (rekursif), terurut."""
    folder = pathlib.Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"folder {folder} tidak ada; periksa pintasan Drive Anda")
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EKSTENSI and not p.name.startswith("."))


def baca_citra(path: str | pathlib.Path) -> np.ndarray:
    """Baca citra sebagai array BGR uint8 (aman untuk nama berkas berhuruf non-ASCII)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    citra = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if citra is None:
        raise ValueError(f"{path} bukan citra yang bisa dibaca")
    return citra


def simpan_citra(path: str | pathlib.Path, citra: np.ndarray) -> pathlib.Path:
    """Simpan citra (BGR atau keabuan) ke path; folder dibuat bila belum ada."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix or ".png", citra)
    if not ok:
        raise ValueError(f"gagal menyandikan citra ke {path.suffix}")
    data.tofile(str(path))
    return path


def ubah_ukuran_maks(citra: np.ndarray, sisi: int = 1024) -> np.ndarray:
    """Perkecil citra agar sisi terpanjangnya <= sisi (citra kecil tidak diperbesar)."""
    tinggi, lebar = citra.shape[:2]
    skala = sisi / max(tinggi, lebar)
    if skala >= 1:
        return citra.copy()
    return cv2.resize(citra, (round(lebar * skala), round(tinggi * skala)), interpolation=cv2.INTER_AREA)


def _ke_bgr(citra: np.ndarray) -> np.ndarray:
    citra = np.asarray(citra)
    if citra.dtype != np.uint8:
        citra = np.clip(citra * (255.0 if citra.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if citra.ndim == 2:
        citra = cv2.cvtColor(citra, cv2.COLOR_GRAY2BGR)
    return citra


def berdampingan(daftar: Sequence[np.ndarray], judul: Sequence[str] | None = None, tinggi: int = 360) -> np.ndarray:
    """Gabungkan beberapa citra ke satu baris dengan tinggi sama dan label di atasnya."""
    if not daftar:
        raise ValueError("daftar citra kosong")
    judul = list(judul) if judul else [""] * len(daftar)
    if len(judul) != len(daftar):
        raise ValueError("jumlah judul harus sama dengan jumlah citra")
    panel = []
    for citra, teks in zip(daftar, judul):
        citra = _ke_bgr(citra)
        h, w = citra.shape[:2]
        citra = cv2.resize(citra, (max(1, round(w * tinggi / h)), tinggi), interpolation=cv2.INTER_AREA)
        pita = np.full((36, citra.shape[1], 3), 245, np.uint8)
        cv2.putText(pita, teks, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        panel.append(np.vstack([pita, citra]))
        panel.append(np.full((tinggi + 36, 8, 3), 255, np.uint8))
    return np.hstack(panel[:-1])


def simpan_sebelum_sesudah(sebelum: np.ndarray, sesudah: np.ndarray, path: str | pathlib.Path,
                           judul: Sequence[str] = ("sebelum", "sesudah")) -> pathlib.Path:
    """Simpan gambar sebelum–sesudah berdampingan (untuk outputs/figures/)."""
    return simpan_citra(path, berdampingan([sebelum, sesudah], judul))


def tampil(*gambar: np.ndarray, judul: Sequence[str] | None = None, kolom: int | None = None, ukuran: float = 4.0):
    """Tampilkan citra di notebook. Citra 3 kanal dianggap BGR (format OpenCV)."""
    import matplotlib.pyplot as plt

    judul = list(judul) if judul else [""] * len(gambar)
    kolom = kolom or len(gambar)
    baris = -(-len(gambar) // kolom)
    fig, ax = plt.subplots(baris, kolom, figsize=(ukuran * kolom, ukuran * baris), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for a, g, j in zip(ax.ravel(), gambar, judul):
        g = np.asarray(g)
        a.imshow(g[:, :, ::-1] if g.ndim == 3 else g, cmap=None if g.ndim == 3 else "gray")
        a.set_title(j)
    plt.tight_layout()
    plt.show()


def statistik_dataset(berkas: Sequence[pathlib.Path]) -> dict:
    """Jumlah citra serta ringkasan ukuran (lebar, tinggi) dan citra yang gagal dibaca."""
    ukuran, gagal = [], []
    for p in berkas:
        try:
            h, w = baca_citra(p).shape[:2]
            ukuran.append((w, h))
        except ValueError:
            gagal.append(p)
    hasil = {"jumlah": len(berkas), "terbaca": len(ukuran), "gagal": gagal}
    if ukuran:
        arr = np.array(ukuran)
        for i, nama in enumerate(("lebar", "tinggi")):
            hasil[nama] = {"min": int(arr[:, i].min()), "median": int(np.median(arr[:, i])), "maks": int(arr[:, i].max())}
    return hasil


def baca_anotasi(path: str | pathlib.Path, kolom_wajib: Sequence[str]) -> list[dict[str, str]]:
    """Baca anotasi.csv dan pastikan kolomnya persis sesuai format topik."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        pembaca = csv.DictReader(f)
        kolom = [k.strip() for k in (pembaca.fieldnames or [])]
        kurang = [k for k in kolom_wajib if k not in kolom]
        if kurang:
            raise ValueError(f"kolom anotasi kurang: {', '.join(kurang)} (ada: {', '.join(kolom)})")
        return [{k.strip(): (v or "").strip() for k, v in baris.items() if k} for baris in pembaca]


def cek_anotasi(baris: Sequence[dict[str, str]], folder_citra: str | pathlib.Path,
                kolom_berkas: Sequence[str]) -> dict[str, list[str]]:
    """Cocokkan anotasi dengan berkas citra.

    Hasil: berkas yang disebut anotasi tetapi tidak ada (`hilang`) dan citra yang
    belum beranotasi sama sekali (`tanpa_anotasi`). Nama berkas relatif terhadap folder_citra.
    """
    folder = pathlib.Path(folder_citra)
    ada = {p.relative_to(folder).as_posix() for p in daftar_citra(folder)}
    disebut = {b[k] for b in baris for k in kolom_berkas if b.get(k)}
    return {"hilang": sorted(disebut - ada), "tanpa_anotasi": sorted(ada - disebut)}
