# Panduan kerja kelompok

Repo ini **publik**: siapa pun bisa melihat isinya. Jangan menulis NIM, email, nomor HP, token, atau kata sandi di berkas mana pun.

## Sekali di awal (setiap anggota)

1. Terima undangan ke repo ini dari email atau notifikasi GitHub.
2. Pasang aplikasi **GitHub Mobile** untuk menyetujui *pull request* dari ponsel.
3. Isi baris Anda di `ANGGOTA.yml` (nama panggilan dan peran) lewat *pull request* pertama Anda.
4. Ketua kelompok membuat folder Google Drive bernama **`pcd2026-uji-t1`**, membagikannya ke semua anggota dan email dosen dengan akses **Terbatas** (bukan "siapa saja yang memiliki link"). Setiap anggota: buka folder itu → **Tambahkan pintasan ke Drive** → *Drive Saya*, supaya notebook menemukannya di `/content/drive/MyDrive/pcd2026-uji-t1`.

Isi folder Drive:

```text
pcd2026-uji-t1/
├── citra/            semua citra dataset (subfolder boleh)
├── anotasi.csv       anotasi, kolom sesuai data/format_anotasi.csv
└── figures/          gambar hasil notebook (lalu diunggah ke outputs/figures/)
```

## Alur kerja: dari Colab ke `main`

`main` dikunci: perubahan hanya masuk lewat *pull request* yang disetujui satu anggota lain dan lolos pemeriksaan **cek**.

1. **Buat branch** di github.com: tombol branch `main` → ketik nama baru, mis. `t1-preprocessing-andi` → *Create branch*. Dialog Colab hanya menampilkan branch yang sudah ada.
2. **Buka notebook di Colab** lewat tombol *Open in Colab* di README. Sel pertama menyalin repo ini dan menghubungkan Google Drive.
3. Kerjakan, lalu **Runtime → Restart session and run all**. Notebook harus selesai tanpa error.
4. **File → Save a copy in GitHub** → pilih repo `Prodi-TI-IBIKesatuan/pcd2026-uji-t1`, **branch Anda**, path tetap `notebooks/…` → OK.
   - Pertama kali, Colab meminta izin ke GitHub. Di daftar organisasi, klik **Request** (atau **Grant**) di samping `Prodi-TI-IBIKesatuan`. Bila repo tidak muncul, tunggu dosen menyetujui permintaan itu.
5. **Unggah gambar hasil**: buka branch Anda di github.com → folder `outputs/figures/` → *Add file → Upload files* → pilih berkas dari folder `figures/` di Drive → *Commit directly to the …* branch Anda.
6. **Buka pull request**: github.com menampilkan tombol *Compare & pull request* → isi checklist → minta satu anggota sebagai *reviewer*.
7. Setelah disetujui dan **cek** hijau, tekan *Merge*. Branch otomatis terhapus.

## Dari ponsel

| Pekerjaan | Caranya |
| --- | --- |
| Unggah foto sampel | Browser ponsel → repo → *Add file → Upload files* → di bawah pilih **Create a new branch for this commit and start a pull request** |
| Menyetujui PR teman | GitHub Mobile → *Pull requests* → buka PR → *Files changed* → **Approve** |
| Menjalankan notebook | Colab bisa dibuka di browser ponsel; menulis kode lebih nyaman di laptop atau PC kampus |

Unggah lewat browser dibatasi 25 MB per berkas, dan pemeriksaan **cek** menolak berkas di atas 10 MB. Kecilkan resolusi foto sampel (sisi terpanjang ±1280 px).

## Mengumpulkan (Release)

1. Pastikan semua *pull request* sudah di-merge dan run **Cek proyek** terakhir di `main` menulis **Siap tahap-1** (atau **Siap tahap-2**) di ringkasannya (tab *Actions*).
2. Tab **Releases → Draft a new release** → *Choose a tag* → ketik `tahap-1` atau `tahap-2` → *Create new tag on publish*, target `main`.
3. Judul: `Tahap 1 – pcd2026-uji-t1`. Klik **Publish release**.
4. Salin tautan Release, tempel di tugas Moodle.

Tag terkunci: Release tidak bisa dihapus atau dipindahkan setelah dibuat. Waktu Release dicatat server GitHub.

## Aturan pemeriksaan **cek**

| Diperiksa | Batas |
| --- | --- |
| Ukuran berkas | ≤ 10 MB per berkas |
| Gambar di `data/sample/` | ≤ 20, sudah disensor |
| Gambar di `outputs/figures/` | ≤ 60 |
| Gambar di tempat lain | tidak boleh (kecuali ≤ 10 di `app/contoh/`); dataset lengkap di Drive |
| `ANGGOTA.yml` | 3–5 anggota, nama dan peran terisi, keempat peran (`data`, `klasik`, `modern`, `evaluasi`) ada |
| Tahap 1 | README bagian Dataset terisi, `data/README.md` berisi tautan Drive, notebook 01 sudah dijalankan, ≥ 10 gambar di `outputs/figures/` |
| Tahap 2 | README bagian Hasil dan Demo terisi, tautan Hugging Face Spaces, notebook 02–04 sudah dijalankan |

Teks `[isi…]` di README menandai bagian yang belum diisi; **cek** menganggap bagian itu kosong.

## Demo di Hugging Face Spaces

1. Buat akun di huggingface.co → *New Space* → SDK **Gradio**, hardware **CPU basic** (gratis).
2. Unggah isi folder `app/` (termasuk `README.md` dan `requirements.txt` di dalamnya) ke Space.
3. Salin tautan Space ke README bagian Demo.

## Kendala

Buka issue di repo ini dan sebut anggota terkait dengan `@username`. Masalah teknis lab (Colab, izin organisasi, cek yang keliru): buka issue **Kendala Lab** di repo [`Prodi-TI-IBIKesatuan/kelas-pcd-2026`](https://github.com/Prodi-TI-IBIKesatuan/kelas-pcd-2026/issues/new/choose).
