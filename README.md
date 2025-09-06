# Deteksi Emosi Wajah (Sedih / Datar / Tersenyum)

Script Python sederhana untuk mendeteksi emosi dasar (sedih, datar, tersenyum) secara real-time dari webcam atau video file, menggunakan library [FER](https://github.com/justinshenk/fer) + OpenCV.

## Cara pakai

1. **Buat virtual env (opsional tapi direkomendasikan)**  
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependensi**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan**  
   - Kamera default:
     ```bash
     python emotion_detect.py
     ```
   - Pilih kamera lain (mis. 1):
     ```bash
     python emotion_detect.py --camera 1
     ```
   - Dari file video:
     ```bash
     python emotion_detect.py --video path/to/video.mp4
     ```

4. **Kontrol**  
   - Tekan `q` untuk keluar.
   - Opsi tambahan:
     - `--min-conf 0.6` : ambang kepercayaan untuk menampilkan label.
     - `--show-fps` : tampilkan FPS.

## Catatan akurasi
- FER sudah dilatih pada beberapa emosi (happy, neutral, sad, angry, fear, disgust, surprise). Kita mapping ke tiga kelas yang diminta.  
- Pencahayaan bagus + kamera menghadap frontal akan bantu akurasi.  
- Ada smoothing mayoritas (history 5 frame) untuk mengurangi "nge-jitter".

## Troubleshooting
- **Kamera nggak kebuka**: coba `--camera 0` atau pastikan app lain tidak memakai kamera.
- **Lambat/lag**: kecilkan resolusi kamera via driver/config, matikan app lain, atau coba tanpa `--show-fps`.
- **ImportError**: pastikan `pip install -r requirements.txt` sukses.

Selamat ngoprek! 🚀
