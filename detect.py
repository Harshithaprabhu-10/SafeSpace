import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Open video file (or webcam by changing the source)
cap = cv2.VideoCapture('test_video.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # ALERT LOGIC 👇
    for r in results:
        boxes = r.boxes
        person_count = 0

        for box in boxes:
            cls = int(box.cls[0])
            if model.names[cls] == 'person':
                person_count += 1

        if person_count >= 2:  # Example: Alert if 2 or more people detected
            print("⚠️ ALERT: Multiple people detected!")

    # Show the annotated frame
    cv2.imshow("SafeSpace Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
