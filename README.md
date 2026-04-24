# 🐦 Bird Vision – Bird Species Recognition Dashboard

Bird Vision is an AI-powered system that detects and identifies bird species using YOLO and ResNet.

## 🚀 Features

- Live detection
- Image & video upload
- Bird insights
- Settings management
## 🧠 How It Works

1. YOLOv8 detects the bird and draws a bounding box
2. The detected region is cropped
3. ResNet50 classifies the bird species
4. The result is displayed with name and confidence

---

## ⚙️ Setup

```bash
git clone https://github.com/IsuruParindya/BirdVision_Local.git
cd BirdVision_Local
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
## ▶️ Run Image / Video detection
python main.py
## ▶️ Run Live camera detection
python live.py