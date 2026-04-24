import os
import cv2
import json
import torch
import numpy as np
import tkinter as tk

from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
from torch import nn
from ultralytics import YOLO

# =========================
# START MESSAGE
# =========================
print("🚀 Starting BirdVision...")
print("⏳ Loading system, please wait...\n")

# =========================
# PATHS
# =========================
base = "models"

pth = os.path.join(base, "bird_resnet50.pth")
cls = os.path.join(base, "classes.json")
si  = os.path.join(base, "sinhala_map.json")
yolo_path = os.path.join(base, "best.pt")

for p in [pth, cls, si, yolo_path]:
    if not os.path.exists(p):
        raise Exception(f"Missing file: {p}")

# =========================
# DEVICE AUTO DETECT
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Running on:", device)

# =========================
# LOAD DATA
# =========================
with open(cls, "r", encoding="utf-8") as f:
    classes = json.load(f)

with open(si, "r", encoding="utf-8") as f:
    sinhala = json.load(f)

# =========================
# LOAD MODELS
# =========================
print("📦 Loading models...")

clf = models.resnet50(weights=None)
clf.fc = nn.Linear(clf.fc.in_features, len(classes))
clf.load_state_dict(torch.load(pth, map_location=device))
clf = clf.to(device).eval()

detector = YOLO(yolo_path)

print("✅ Models loaded successfully!\n")

# =========================
# FILE PICKER (AFTER LOAD)
# =========================
print("📂 Opening file picker...")

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select Image or Video",
    filetypes=[
        ("Media files", "*.jpg *.jpeg *.png *.webp *.mp4 *.avi *.mov *.mkv")
    ]
)

if not file_path:
    print("❌ No file selected")
    exit()

print("📄 Selected:", file_path)

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def classify(img):
    x = transform(
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(clf(x), dim=1)
        conf, pred = torch.max(probs, 1)

    return classes[int(pred.item())], float(conf.item())

# =========================
# DRAW
# =========================
def draw_detection(frame, box, raw, conf):
    x1, y1, x2, y2 = box

    name = raw.replace("_", " ").title()
    si_name = sinhala.get(raw, "Unknown")
    label_en = f"{name} ({conf*100:.0f}%)"

    box_w = x2 - x1
    font_size = max(16, min(32, int(box_w * 0.045)))

    en_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    si_font = ImageFont.truetype("fonts/NotoSansSinhala-Regular.ttf", font_size)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    padding = int(font_size * 0.25)

    en_box = draw.textbbox((0,0), label_en, font=en_font)
    si_box = draw.textbbox((0,0), si_name, font=si_font)

    text_width = max(en_box[2], si_box[2]) + padding*2
    text_height = en_box[3] + si_box[3] + padding*3

    y_top = y1 - text_height if y1 - text_height > 0 else y1 + 5

    draw.rectangle(
        [x1, y_top, x1 + text_width, y_top + text_height],
        fill=(173,216,230)
    )

    draw.text((x1 + padding, y_top + padding),
              label_en, font=en_font, fill=(0,0,0))

    draw.text((x1 + padding, y_top + en_box[3] + padding),
              si_name, font=si_font, fill=(0,0,0))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# =========================
# PROCESS FRAME
# =========================
def process_frame(frame):
    results = detector(frame, conf=0.15, imgsz=640, verbose=False)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        best_box = results.boxes[0]
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        crop = frame[y1:y2, x1:x2]

        if crop.size != 0:
            raw, conf = classify(crop)
            frame = draw_detection(frame, (x1,y1,x2,y2), raw, conf)

    return frame

# =========================
# RUN
# =========================
ext = os.path.splitext(file_path)[1].lower()

if ext in [".jpg", ".jpeg", ".png", ".webp"]:
    image = cv2.imread(file_path)
    result = process_frame(image)

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
    cap = cv2.VideoCapture(file_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "result.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    print("🎬 Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        out.write(frame)

    cap.release()
    out.release()

    print("✅ Done! Saved as result.mp4")

else:
    print("Unsupported file ❌")