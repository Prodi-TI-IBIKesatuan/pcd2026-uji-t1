"""Pemeriksaan otomatis repo proyek kelompok PCD 2026 (pemeriksaan "cek").

Dijalankan oleh .github/workflows/cek-proyek.yml di setiap push, pull request, dan Release.

    python .github/scripts/cek_proyek.py                 # struktur (tahap 0)
    python .github/scripts/cek_proyek.py --tahap 1       # syarat Release tahap-1
    python .github/scripts/cek_proyek.py --tahap 2       # syarat Release tahap-2 (termasuk tahap 1)

Kode keluar 1 bila ada GALAT. Pada tahap 0, kesiapan tahap 1 dan 2 ikut dilaporkan
sebagai informasi agar kelompok tahu kapan boleh membuat Release.
Berkas ini dikelola dosen; perubahan oleh kelompok akan ditimpa.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass

AKAR_BAWAAN = pathlib.Path(__file__).resolve().parents[2]

WAJIB = [
    "README.md", "PANDUAN.md", "ANGGOTA.yml", "requirements.txt",
    "data/README.md", "data/format_anotasi.csv", "app/app.py",
    "notebooks/01_dataset_preprocessing.ipynb", "pcd_proyek/metrik.py",
]
EKSTENSI_CITRA = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".gif"}
BATAS_MB = 10
BATAS_CITRA = {"data/sample": 20, "outputs/figures": 60, "app/contoh": 10}
MIN_SAMPEL = 5
MIN_FIGUR = 10
PERAN = ("data", "klasik", "modern", "evaluasi")
ANGGOTA_MIN, ANGGOTA_MAKS = 3, 5
POLA_USERNAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
POLA_ISI = re.compile(r"\[isi[^\]]*\]", re.IGNORECASE)
POLA_DRIVE = re.compile(r"https://drive\.google\.com/\S+")
POLA_SPACE = re.compile(r"https://huggingface\.co/spaces/[\w.-]+/[\w.-]+")
NOTEBOOK_TAHAP2 = ["02_metode_klasik.ipynb", "03_metode_modern.ipynb", "04_evaluasi.ipynb"]
ABAIKAN_DIR = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


@dataclass(frozen=True)
class Temuan:
    tingkat: str  # "galat" atau "peringatan"
    pesan: str
    berkas: str = ""


def _galat(pesan, berkas=""):
    return Temuan("galat", pesan, berkas)


def _peringatan(pesan, berkas=""):
    return Temuan("peringatan", pesan, berkas)


def semua_berkas(akar: pathlib.Path):
    for p in sorted(akar.rglob("*")):
        if p.is_file() and not any(bagian in ABAIKAN_DIR for bagian in p.relative_to(akar).parts):
            yield p


def bagian_markdown(teks: str) -> dict[str, str]:
    """Pecah markdown menjadi {judul '## …' (huruf kecil): isi}. Teks sebelum judul pertama berkunci ''."""
    hasil, judul, isi = {}, "", []
    for baris in teks.splitlines():
        cocok = re.match(r"^##\s+(.*?)\s*$", baris)
        if cocok:
            hasil[judul] = "\n".join(isi)
            judul, isi = cocok.group(1).lower(), []
        else:
            isi.append(baris)
    hasil[judul] = "\n".join(isi)
    return hasil


def cari_bagian(bagian: dict[str, str], awalan: str) -> str | None:
    for judul, isi in bagian.items():
        if judul.startswith(awalan):
            return isi
    return None


# ---------------------------------------------------------------- pemeriksaan


def cek_struktur(akar):
    return [_galat(f"berkas wajib tidak ada: {w}", w) for w in WAJIB if not (akar / w).is_file()]


def cek_berkas(akar):
    temuan, jumlah = [], {k: 0 for k in BATAS_CITRA}
    for p in semua_berkas(akar):
        rel = p.relative_to(akar).as_posix()
        mb = p.stat().st_size / 1_000_000
        if mb > BATAS_MB:
            temuan.append(_galat(f"{rel} berukuran {mb:.1f} MB (batas {BATAS_MB} MB); kecilkan atau simpan di Drive", rel))
        if p.suffix.lower() not in EKSTENSI_CITRA:
            continue
        folder = next((k for k in BATAS_CITRA if rel.startswith(k + "/")), None)
        if folder is None:
            temuan.append(_galat(f"gambar di luar folder yang diizinkan: {rel}. Dataset lengkap disimpan di Google Drive; "
                                 f"contoh di data/sample/, hasil di outputs/figures/", rel))
        else:
            jumlah[folder] += 1
    for folder, batas in BATAS_CITRA.items():
        if jumlah[folder] > batas:
            temuan.append(_galat(f"{folder}/ berisi {jumlah[folder]} gambar (batas {batas})", folder))
    return temuan, jumlah


def baca_anggota(akar):
    path = akar / "ANGGOTA.yml"
    if not path.is_file():
        return None, []
    try:
        import yaml
    except ImportError:  # pragma: no cover - CI selalu memasang PyYAML
        return None, [_peringatan("PyYAML tidak terpasang; ANGGOTA.yml tidak diperiksa", "ANGGOTA.yml")]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        return None, [_galat(f"ANGGOTA.yml bukan YAML yang sah: {e}", "ANGGOTA.yml")]
    if not isinstance(data, dict) or not isinstance(data.get("anggota"), list):
        return None, [_galat("ANGGOTA.yml harus berisi daftar 'anggota:'", "ANGGOTA.yml")]
    return data, []


def _daftar_peran(nilai) -> list[str]:
    if isinstance(nilai, list):
        bagian = [str(v) for v in nilai]
    else:
        bagian = re.split(r"[,\s]+", str(nilai or ""))
    return [b.strip().lower() for b in bagian if b.strip()]


def cek_anggota(akar, tahap):
    data, temuan = baca_anggota(akar)
    if data is None:
        return temuan, 0
    anggota = data["anggota"]
    f = "ANGGOTA.yml"
    if not ANGGOTA_MIN <= len(anggota) <= ANGGOTA_MAKS:
        temuan.append(_galat(f"jumlah anggota {len(anggota)}; harus {ANGGOTA_MIN}–{ANGGOTA_MAKS}", f))
    tingkat_isi = _galat if tahap >= 1 else _peringatan
    username, peran_ada = [], set()
    for i, a in enumerate(anggota, 1):
        if not isinstance(a, dict):
            temuan.append(_galat(f"anggota ke-{i} harus berisi github, nama, dan peran", f))
            continue
        gh = str(a.get("github") or "").lstrip("@")
        if not POLA_USERNAME.match(gh):
            temuan.append(_galat(f"anggota ke-{i}: username GitHub '{gh}' tidak sah", f))
        username.append(gh.lower())
        for kunci in ("nim", "email", "hp", "telepon"):
            if kunci in a:
                temuan.append(_galat(f"anggota ke-{i}: hapus '{kunci}'; repo ini publik", f))
        nama = str(a.get("nama") or "")
        if not nama.strip() or POLA_ISI.search(nama):
            temuan.append(tingkat_isi(f"anggota @{gh}: nama belum diisi", f))
        peran = _daftar_peran(a.get("peran"))
        if not peran or any(POLA_ISI.search(p) or p.startswith("[") for p in peran):
            temuan.append(tingkat_isi(f"anggota @{gh}: peran belum diisi (pilih dari {', '.join(PERAN)})", f))
            continue
        salah = [p for p in peran if p not in PERAN]
        if salah:
            temuan.append(_galat(f"anggota @{gh}: peran tidak dikenal {salah}; pilih dari {', '.join(PERAN)}", f))
        peran_ada.update(p for p in peran if p in PERAN)
    kembar = sorted({u for u in username if username.count(u) > 1})
    if kembar:
        temuan.append(_galat(f"username ganda: {', '.join(kembar)}", f))
    kurang = [p for p in PERAN if p not in peran_ada]
    if kurang:
        temuan.append(tingkat_isi(f"peran belum ada yang memegang: {', '.join(kurang)}", f))
    return temuan, len(anggota)


def notebook_dijalankan(path: pathlib.Path) -> tuple[bool, str]:
    """(berhasil?, alasan). Berhasil = notebook sah, punya sel kode yang sudah dijalankan, dan tanpa keluaran error."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return False, f"bukan notebook yang sah ({e.__class__.__name__})"
    kode = [s for s in nb.get("cells", []) if s.get("cell_type") == "code"]
    if not kode:
        return False, "tidak punya sel kode"
    if not any(s.get("execution_count") for s in kode):
        return False, "belum dijalankan (simpan setelah Runtime → Restart session and run all)"
    for s in kode:
        for keluaran in s.get("outputs", []):
            if keluaran.get("output_type") == "error":
                return False, f"masih ada error {keluaran.get('ename', '')} di keluaran"
    return True, ""


def cek_notebook_sah(akar):
    temuan = []
    for p in sorted((akar / "notebooks").glob("*.ipynb")):
        try:
            nb = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(nb.get("cells"), list):
                raise ValueError
        except (ValueError, UnicodeDecodeError):
            rel = p.relative_to(akar).as_posix()
            temuan.append(_galat(f"{rel} rusak (bukan JSON notebook); simpan ulang dari Colab", rel))
    return temuan


def cek_tahap1(akar, jumlah_citra):
    temuan = []
    readme = (akar / "README.md").read_text(encoding="utf-8") if (akar / "README.md").is_file() else ""
    bagian = bagian_markdown(readme)
    lingkup = next((b for b in readme.splitlines() if b.lower().startswith("**lingkup")), "")
    if not lingkup or POLA_ISI.search(lingkup):
        temuan.append(_galat("README: baris 'Lingkup kelompok kami' belum diisi", "README.md"))
    dataset = cari_bagian(bagian, "dataset")
    if dataset is None or POLA_ISI.search(dataset):
        temuan.append(_galat("README: bagian 'Dataset (Tahap 1)' masih berisi [isi…]", "README.md"))
    data_readme = (akar / "data/README.md").read_text(encoding="utf-8") if (akar / "data/README.md").is_file() else ""
    ringkasan = cari_bagian(bagian_markdown(data_readme), "ringkasan") or ""
    if POLA_ISI.search(ringkasan):
        temuan.append(_galat("data/README.md: bagian 'Ringkasan' masih berisi [isi…]", "data/README.md"))
    if not POLA_DRIVE.search(ringkasan):
        temuan.append(_galat("data/README.md: tautan folder Google Drive belum ada", "data/README.md"))
    nb = akar / "notebooks/01_dataset_preprocessing.ipynb"
    if nb.is_file():
        ok, alasan = notebook_dijalankan(nb)
        if not ok:
            temuan.append(_galat(f"notebooks/01_dataset_preprocessing.ipynb {alasan}", "notebooks/01_dataset_preprocessing.ipynb"))
    if jumlah_citra["data/sample"] < MIN_SAMPEL:
        temuan.append(_galat(f"data/sample/ berisi {jumlah_citra['data/sample']} gambar; minimal {MIN_SAMPEL} contoh tersensor", "data/sample"))
    if jumlah_citra["outputs/figures"] < MIN_FIGUR:
        temuan.append(_galat(f"outputs/figures/ berisi {jumlah_citra['outputs/figures']} gambar; minimal {MIN_FIGUR} sebelum–sesudah", "outputs/figures"))
    return temuan


def jumlah_klasik_wajib(jumlah_anggota: int) -> int:
    return 3 if jumlah_anggota >= 5 else 2


def cek_tahap2(akar, jumlah_anggota):
    temuan = []
    readme = (akar / "README.md").read_text(encoding="utf-8") if (akar / "README.md").is_file() else ""
    bagian = bagian_markdown(readme)
    hasil = cari_bagian(bagian, "hasil")
    if hasil is None or POLA_ISI.search(hasil):
        temuan.append(_galat("README: bagian 'Hasil (Tahap 2)' masih berisi [isi…]", "README.md"))
    else:
        baris = [b for b in hasil.splitlines() if b.strip().startswith("|")]
        jenis = [b.split("|")[2].strip().lower() for b in baris[2:] if b.count("|") >= 3]
        wajib = jumlah_klasik_wajib(jumlah_anggota)
        if jenis.count("klasik") < wajib:
            temuan.append(_galat(f"README: tabel Hasil memuat {jenis.count('klasik')} metode klasik; wajib {wajib}", "README.md"))
        if jenis.count("modern") < 1:
            temuan.append(_galat("README: tabel Hasil belum memuat metode modern", "README.md"))
    demo = cari_bagian(bagian, "demo")
    if demo is None or POLA_ISI.search(demo) or not POLA_SPACE.search(demo):
        temuan.append(_galat("README: bagian 'Demo' belum berisi tautan https://huggingface.co/spaces/…", "README.md"))
    atribusi = cari_bagian(bagian, "sumber")
    if atribusi is not None and POLA_ISI.search(atribusi):
        temuan.append(_peringatan("README: tabel 'Sumber dan atribusi' masih berisi [isi]; hapus baris itu bila tidak ada sumber luar", "README.md"))
    for nama in NOTEBOOK_TAHAP2:
        path = akar / "notebooks" / nama
        if not path.is_file():
            temuan.append(_galat(f"notebooks/{nama} tidak ada", f"notebooks/{nama}"))
            continue
        ok, alasan = notebook_dijalankan(path)
        if not ok:
            temuan.append(_galat(f"notebooks/{nama} {alasan}", f"notebooks/{nama}"))
    return temuan


def periksa(akar: pathlib.Path, tahap: int) -> tuple[list[Temuan], dict]:
    temuan = cek_struktur(akar)
    t, jumlah_citra = cek_berkas(akar)
    temuan += t
    t, jumlah_anggota = cek_anggota(akar, tahap)
    temuan += t
    temuan += cek_notebook_sah(akar)
    kesiapan = {}
    syarat1 = cek_tahap1(akar, jumlah_citra)
    syarat2 = cek_tahap2(akar, jumlah_anggota)
    if tahap >= 1:
        temuan += syarat1
    if tahap >= 2:
        temuan += syarat2
    if tahap == 0:
        dasar = [x for x in cek_anggota(akar, 1)[0] if x.tingkat == "galat"]
        kesiapan["tahap-1"] = [x.pesan for x in dasar + syarat1 if x.tingkat == "galat"]
        kesiapan["tahap-2"] = [x.pesan for x in syarat2 if x.tingkat == "galat"]
    info = {"jumlah_citra": jumlah_citra, "jumlah_anggota": jumlah_anggota, "kesiapan": kesiapan}
    return temuan, info


def ringkasan_markdown(temuan, info, tahap) -> str:
    galat = [t for t in temuan if t.tingkat == "galat"]
    peringatan = [t for t in temuan if t.tingkat == "peringatan"]
    judul = {0: "struktur", 1: "Release tahap-1", 2: "Release tahap-2"}[tahap]
    teks = [f"## Cek proyek ({judul}): {'LULUS' if not galat else 'GAGAL'}", ""]
    for nama, daftar in (("Galat", galat), ("Peringatan", peringatan)):
        if daftar:
            teks.append(f"### {nama}")
            teks += [f"- {t.pesan}" for t in daftar]
            teks.append("")
    jc = info["jumlah_citra"]
    teks.append(f"Anggota: {info['jumlah_anggota']} · data/sample: {jc['data/sample']} · outputs/figures: {jc['outputs/figures']}")
    kesiapan = info["kesiapan"]
    if kesiapan:
        siap1 = not kesiapan["tahap-1"]
        siap2 = siap1 and not kesiapan["tahap-2"]
        for nama, siap, kurang in (("tahap-1", siap1, kesiapan["tahap-1"]), ("tahap-2", siap2, kesiapan["tahap-2"])):
            teks.append("")
            if siap:
                teks.append(f"### Siap {nama}")
                teks.append(f"Semua syarat otomatis {nama} terpenuhi. Release `{nama}` boleh dibuat.")
                continue
            teks.append(f"### Belum siap {nama}")
            if nama == "tahap-2" and not siap1:
                teks.append("Selain kekurangan tahap-1 di atas:")
            teks += [f"- {k}" for k in kurang]
    return "\n".join(teks) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tahap", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--akar", type=pathlib.Path, default=AKAR_BAWAAN)
    parser.add_argument("--ringkasan", type=pathlib.Path, help="tulis ringkasan markdown (mis. $GITHUB_STEP_SUMMARY)")
    arg = parser.parse_args(argv)
    temuan, info = periksa(arg.akar.resolve(), arg.tahap)
    for t in temuan:
        perintah = "error" if t.tingkat == "galat" else "warning"
        lokasi = f" file={t.berkas}" if t.berkas else ""
        print(f"::{perintah}{lokasi}::{t.pesan}")
    teks = ringkasan_markdown(temuan, info, arg.tahap)
    print(teks)
    if arg.ringkasan:
        with open(arg.ringkasan, "a", encoding="utf-8") as f:
            f.write(teks)
    return 1 if any(t.tingkat == "galat" for t in temuan) else 0


if __name__ == "__main__":
    sys.exit(main())
