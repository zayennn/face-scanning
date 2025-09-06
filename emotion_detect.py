import argparse
from collections import deque, Counter
import time

import cv2
import numpy as np
from fer import FER


ID2COLOR = {
    "sedih": (60, 60, 255),
    "datar": (200, 200, 200),
    "tersenyum": (60, 200, 60),
}

def map_emotions(fer_emotions: dict) -> str:

    for k in ["happy", "neutral", "sad"]:
        fer_emotions.setdefault(k, 0.0)

    scores = {
        "tersenyum": fer_emotions.get("happy", 0.0),
        "datar": fer_emotions.get("neutral", 0.0),
        "sedih": fer_emotions.get("sad", 0.0),
    }

    return max(scores, key=scores.get)


def smooth_label(label_history: deque, k: int = 5) -> str:
    """
    Majority vote over the last k labels to reduce jitter.
    """
    if not label_history:
        return "datar"
    counts = Counter(label_history)
    return counts.most_common(1)[0][0]


def draw_label(frame, box, label: str, score: float):
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    color = ID2COLOR.get(label, (255, 255, 255))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    text = f"{label} ({int(score*100)}%)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y), color, -1)
    cv2.putText(frame, text, (x + 4, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="Deteksi emosi wajah (sedih/datar/tersenyum) real-time.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="Index kamera (default: 0)")
    src.add_argument("--video", type=str, help="Path ke file video untuk dianalisis")
    parser.add_argument("--min-conf", type=float, default=0.6,
                        help="Minimal confidence untuk menampilkan label (default: 0.6)")
    parser.add_argument("--show-fps", action="store_true", help="Tampilkan FPS di layar")
    args = parser.parse_args()

    detector = FER(mtcnn=False)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print("[ERROR] Gagal membuka kamera/video. Coba --camera 0 atau cek izin kamera.")
        return

    prev_time = time.time()
    fps = 0.0

    history = deque(maxlen=5)

    window_name = "Deteksi Emosi (q untuk keluar)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FER returns list of dicts with 'box' and 'emotions'
        results = detector.detect_emotions(rgb)

        # Draw results
        if results:
            # Aggregate the strongest face (highest happy/neutral/sad max score) for smoothing
            best_label = None
            best_score = 0.0

            for r in results:
                box = r["box"]  # (x, y, w, h)
                emotions = r["emotions"]  # dict of scores

                label = map_emotions(emotions)
                score = max(emotions.get("happy", 0), emotions.get("neutral", 0), emotions.get("sad", 0))

                # Draw only if above threshold
                if score >= args.min_conf:
                    draw_label(frame, box, label, score)

                # Track best for global smoothing
                if score > best_score:
                    best_label, best_score = label, score

            if best_label is not None:
                history.append(best_label)

        # Show a smoothed label banner (optional, gives a stable global reading)
        smoothed = smooth_label(history)
        banner = f"Terbaca: {smoothed}"
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (10, 10), (10 + tw + 10, 10 + th + 14), (30, 30, 30), -1)
        cv2.putText(frame, banner, (16, 10 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS
        if args.show_fps:
            now = time.time()
            fps = 1.0 / (now - prev_time)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
