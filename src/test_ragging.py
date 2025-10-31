import cv2
from ultralytics import YOLO
from playsound import playsound
from PIL import Image
import numpy as np
import threading
import time

# ✅ Load your trained YOLOv8 classification model
model = YOLO("runs/classify/train10/weights/last.pt")

# Define class names according to your dataset
CLASS_NAMES = ["normal", "ragging"]

# 🧩 Function to play alert sound in background
def play_alert():
    try:
        playsound("alert.mp3")  # make sure alert.mp3 exists in src folder
    except Exception as e:
        print(f"⚠️ Alert sound error: {e}")

# 🧠 Choose input source
# 0 = webcam, or replace with your video file path:
# Example: source = "dataset/videos/ragging/test1.mp4"
#source = "test_video.mp4"
source = 0

# Open video capture
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("❌ Error: Could not open video source.")
    exit()

print("🔍 Starting Ragging Detection... Press 'q' to quit.")

last_alert_time = 0  # to prevent continuous sound

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame for YOLO model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Run prediction
    results = model.predict(pil_img, verbose=False)
    probs = results[0].probs

    if probs is not None:
        pred = probs.top1
        conf = probs.top1conf.item()
        label = CLASS_NAMES[pred]

        # Text color and message setup
        color = (0, 255, 0) if label == "normal" else (0, 0, 255)
        text = f"{label.upper()} ({conf:.2f})"

        # Draw label text
        cv2.putText(frame, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # If ragging detected with high confidence
        if label == "ragging" and conf > 0.85:
            # Red overlay alert on screen
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            alpha = 0.4
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.putText(frame, "⚠️ RAGGING DETECTED!", (100, 200),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 5)

            # Play sound (max once every 3 seconds)
            if time.time() - last_alert_time > 3:
                threading.Thread(target=play_alert, daemon=True).start()
                last_alert_time = time.time()

    # Display the result
    cv2.imshow("SafeSpace - Ragging Detection", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
