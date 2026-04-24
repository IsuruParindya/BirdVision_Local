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
# FILE PICKER
# =========================
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select Image or Video",
    filetypes=[
        ("Media files", "*.jpg *.jpeg *.png *.webp *.mp4 *.avi *.mov *.mkv"),
        ("Images", "*.jpg *.jpeg *.png *.webp"),
        ("Videos", "*.mp4 *.avi *.mov *.mkv")
    ]
)

if not file_path:
    print("No file selected ❌")
    exit()

print("Selected:", file_path)

# =========================
# PATHS
# =========================
base = "models"   # model files are in the same folder as main.py

pth = os.path.join(base, "bird_resnet50.pth")
cls = os.path.join(base, "classes.json")
si  = os.path.join(base, "sinhala_map.json")
info = os.path.join(base, "species_info.json")
yolo_path = os.path.join(base, "best.pt")

for p in [pth, cls, si, info, yolo_path]:
    if not os.path.exists(p):
        raise Exception(f"Missing file: {p}")

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# LOAD JSON FILES - UTF-8 FIX
# =========================
with open(cls, "r", encoding="utf-8") as f:
    classes = json.load(f)

with open(si, "r", encoding="utf-8") as f:
    sinhala = json.load(f)

with open(info, "r", encoding="utf-8") as f:
    species_info = json.load(f)

# =========================
# LOAD MODELS
# =========================
clf = models.resnet50(weights=None)
clf.fc = nn.Linear(clf.fc.in_features, len(classes))
clf.load_state_dict(torch.load(pth, map_location=device))
clf = clf.to(device).eval()

detector = YOLO(yolo_path)
print("Models loaded ✅")

# =========================
# FONTS
# =========================
si_font_path = "fonts/NotoSansSinhala-Regular.ttf"
en_font_path = "C:/Windows/Fonts/arial.ttf"

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify(img):
    x = transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(clf(x), dim=1)
        conf, pred = torch.max(probs, 1)

    return classes[int(pred.item())], float(conf.item())

def draw_detection(frame, box, raw, conf, font_size):
    H, W = frame.shape[:2]

    x1, y1, x2, y2 = box

    name = raw.replace("_", " ").title()
    si_name = sinhala.get(raw, "Unknown")
    label_en = f"{name} ({conf * 100:.0f}%)"

    si_font = ImageFont.truetype(si_font_path, font_size)
    en_font = ImageFont.truetype(en_font_path, font_size)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    label_height = int(font_size * 2.8)
    label_width = int(font_size * 14)

    y_top = y1 - label_height if y1 - label_height > 0 else y1 + 5
    x_right = min(W - 1, x1 + label_width)

    draw.rectangle(
        [x1, y_top, x_right, y_top + label_height],
        fill=(173, 216, 230)
    )

    draw.text((x1 + 6, y_top + 2), label_en, font=en_font, fill=(0, 0, 0))
    draw.text((x1 + 6, y_top + font_size + 4), si_name, font=si_font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def process_frame(frame):
    H, W = frame.shape[:2]

    font_size = max(14, min(20, int(W * 0.022)))

    results = detector(frame, conf=0.10, imgsz=640, verbose=False)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        best_box = results.boxes[0]

        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        crop = frame[y1:y2, x1:x2]

        if crop.size != 0:
            raw, conf = classify(crop)
            frame = draw_detection(frame, (x1, y1, x2, y2), raw, conf, font_size)

    return frame

# =========================
# IMAGE OR VIDEO MODE
# =========================
ext = os.path.splitext(file_path)[1].lower()

image_exts = [".jpg", ".jpeg", ".png", ".webp"]
video_exts = [".mp4", ".avi", ".mov", ".mkv"]

if ext in image_exts:
    image = cv2.imread(file_path)

    if image is None:
        raise Exception("Could not read image.")

    result = process_frame(image)

    output_path = "result.jpg"
    cv2.imwrite(output_path, result)

    cv2.imshow("BirdVision Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Done ✅ Saved image as {output_path}")

elif ext in video_exts:
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

    print(f"Processing video ({W}x{H})...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = process_frame(frame)
        out.write(frame)

    cap.release()
    out.release()

    print("Done ✅ Saved video as result.mp4")

else:
    print("Unsupported file type ❌")