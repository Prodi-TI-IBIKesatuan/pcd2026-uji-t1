"""Demo Gradio untuk Deteksi dan Pengenalan Pelat Nomor Kendaraan.

Jalankan lokal: pip install -r requirements.txt && python app.py  (buka http://127.0.0.1:7860)
Deploy: unggah isi folder app/ ke Hugging Face Spaces (SDK Gradio). Lihat README.md di folder ini.

Yang perlu diubah kelompok: isi fungsi `proses()`. Antarmuka sudah disiapkan untuk topik ini.
"""
from __future__ import annotations

import pathlib

import cv2
import gradio as gr
import numpy as np

MODE = "gambar_teks"  # gambar_teks | gambar_gambar | galeri
JUDUL = "Deteksi dan Pengenalan Pelat Nomor Kendaraan"
KETERANGAN = "Unggah foto kendaraan, tampilkan kotak pelat dan teks yang terbaca."
FOLDER = pathlib.Path(__file__).parent
CONTOH = sorted(str(p) for p in (FOLDER / "contoh").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def proses(citra_bgr: np.ndarray):
    """GANTI dengan metode terbaik kelompok. Masukan: citra BGR uint8 (format OpenCV).

    Keluaran menurut MODE:
      gambar_teks   -> (citra_hasil_bgr, teks)          mis. kotak pelat + teks terbaca
      gambar_gambar -> citra_hasil_bgr                  mis. hasil perbaikan citra malam
      galeri        -> list[(citra_bgr, keterangan)]    mis. citra paling mirip + skornya
    """
    # DUMMY: citra dikembalikan apa adanya. Ganti dengan metode terbaik kelompok.
    if MODE == "gambar_teks":
        return citra_bgr, "DUMMY — ganti fungsi proses() di app.py dengan metode kelompok."
    if MODE == "galeri":
        return [(citra_bgr, "DUMMY")]
    return citra_bgr


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def jalankan(citra_rgb):
    if citra_rgb is None:
        raise gr.Error("Unggah citra terlebih dahulu.")
    hasil = proses(cv2.cvtColor(citra_rgb, cv2.COLOR_RGB2BGR))
    if MODE == "gambar_teks":
        citra, teks = hasil
        return _rgb(citra), teks
    if MODE == "galeri":
        return [(_rgb(c), k) for c, k in hasil]
    return _rgb(hasil)


with gr.Blocks(title=JUDUL) as demo:
    gr.Markdown(f"# {JUDUL}\n{KETERANGAN}")
    with gr.Row():
        masukan = gr.Image(label="Citra masukan", type="numpy")
        if MODE == "gambar_teks":
            with gr.Column():
                keluaran = [gr.Image(label="Hasil"), gr.Textbox(label="Teks terbaca")]
        elif MODE == "galeri":
            keluaran = [gr.Gallery(label="Citra paling mirip", columns=5)]
        else:
            keluaran = [gr.Image(label="Hasil")]
    tombol = gr.Button("Proses", variant="primary")
    tombol.click(jalankan, masukan, keluaran if len(keluaran) > 1 else keluaran[0])
    if CONTOH:
        gr.Examples(CONTOH, masukan)

if __name__ == "__main__":
    demo.launch()
