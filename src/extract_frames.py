# src/extract_frames.py
import cv2, os, sys
from pathlib import Path

# CONFIG - adjust if needed
PROJECT_ROOT = Path(__file__).resolve().parents[1] 
DATASET_DIR = PROJECT_ROOT / "dataset"              # dataset/train, dataset/val
EVERY_N_FRAMES = 10   # take 1 out of every 10 frames (adjust lower for more frames)
IMAGE_EXT = ".jpg"

def extract_from_folder(videos_folder: Path):
    videos = [p for p in videos_folder.iterdir() if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
    if not videos:
        return
    images_out = videos_folder  # save frames to the same folder where videos are
    images_out.mkdir(parents=True, exist_ok=True)
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"⚠️ Cannot open {v}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # save every Nth frame
            if idx % EVERY_N_FRAMES == 0:
                fname = f"{v.stem}_f{idx:06d}{IMAGE_EXT}"
                cv2.imwrite(str(images_out / fname), frame)
                saved += 1
            idx += 1
        cap.release()
        print(f"Extracted {saved} frames from {v.name} -> {images_out}")
    # optional: do not delete videos here (we'll leave them so you can verify)

def main():
    # Expect dataset/train/*class* and dataset/val/*class*
    if not DATASET_DIR.exists():
        print("dataset folder not found. Expected:", DATASET_DIR)
        sys.exit(1)
    for split in ("train", "val"):
        for cls in ("ragging", "normal"):
            folder = DATASET_DIR / split / cls
            if folder.exists():
                print("Processing:", folder)
                extract_from_folder(folder)
            else:
                print("Missing folder:", folder)

if __name__ == "__main__":
    main()
