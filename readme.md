# 🎯 Ragging Detection using Machine Learning

An AI-based system that detects **ragging or bullying activities** from videos or live webcam feed using the **YOLOv8 classification model**.

---

## 🧠 Overview

This project classifies frames into:
- 🟢 **Normal**
- 🔴 **Ragging**

It uses a **trained YOLOv8 model** with a custom dataset to identify and alert when ragging is detected in real-time.

---

## ⚙️ Technologies Used
- Python  
- YOLOv8 (Ultralytics)  
- PyTorch  
- OpenCV  
- Playsound  
- Google Colab (for training)  

---

## 🚀 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/ragging-detection-ml.git
cd ragging-detection-ml

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # For Windows
# or
source venv/bin/activate    # For Mac/Linux

# Install dependencies
pip install ultralytics torch opencv-python playsound gdown
```

---

## 🏋️ Training the Model

```bash
# Train the YOLOv8 classification model
yolo task=classify mode=train model=yolov8m-cls.pt data=dataset epochs=10 imgsz=224 batch=16 augment=True
```

Once completed, the model weights (`best.pt` or `last.pt`) will be saved automatically.

---

## 🎥 Testing

### ▶️ Test on Webcam
```bash
python src/test_ragging.py
```

### ▶️ Test on Video
```bash
python src/test_ragging.py
```

If ragging is detected, an **alert sound** will play.

---

## 📊 Current Status
- Model trained for 5 epochs  
- Achieved ~98% accuracy  
- Will continue improving with higher epoch training  

---

## 👩‍💻 Author
**Harshitha Prabhu**  
Ragging Detection with Machine Learning — ensuring safe environments through AI.
