import os, cv2, json, torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
from torch import nn
from ultralytics import YOLO

# ---- Paths ----
base = "models"

pth = os.path.join(base, "bird_resnet50.pth")
cls = os.path.join(base, "classes.json")
si  = os.path.join(base, "sinhala_map.json")
yolo_path = os.path.join(base, "best.pt")

for p in [pth, cls, si, yolo_path]:
    if not os.path.exists(p):
        raise Exception(f"Missing file: {p}")

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- Load Data ----
with open(cls, "r", encoding="utf-8") as f:
    classes = json.load(f)

with open(si, "r", encoding="utf-8") as f:
    sinhala = json.load(f)

# ---- Load Models ----
clf = models.resnet50(weights=None)
clf.fc = nn.Linear(clf.fc.in_features, len(classes))
clf.load_state_dict(torch.load(pth, map_location=device))
clf = clf.to(device).eval()

detector = YOLO(yolo_path)
print("Models loaded ✅")

# ---- Fonts ----
si_font_path = "fonts/NotoSansSinhala-Regular.ttf"
en_font_path = "C:/Windows/Fonts/arial.ttf"

if not os.path.exists(si_font_path):
    raise Exception(f"Missing Sinhala font: {si_font_path}")

# ---- Transform ----
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

# ---- Auto-detect camera ----
cap = None

for i in range(5):
    temp = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if temp.isOpened():
        ret, test_frame = temp.read()
        if ret and test_frame is not None:
            print(f"Using camera index: {i} ✅")
            cap = temp
            break
        temp.release()

if cap is None:
    raise Exception("No camera found ❌ Try plugging in your webcam or closing apps using the camera.")

print("Press Q to quit")

# ---- Live Feed Loop ----
while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to read frame ❌")
        break

    H, W = frame.shape[:2]
    font_size = max(14, min(20, int(W * 0.02)))

    results = detector(frame, conf=0.25, imgsz=640, verbose=False)[0]

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

            name = raw.replace("_", " ").title()
            si_name = sinhala.get(raw, "Unknown")
            label_en = f"{name} ({conf * 100:.0f}%)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil)

            en_font = ImageFont.truetype(en_font_path, font_size)
            si_font = ImageFont.truetype(si_font_path, font_size)

            label_height = int(font_size * 2.8)
            label_width = int(font_size * 14)

            y_top = y1 - label_height if y1 - label_height > 0 else y1 + 5
            x_right = min(W - 1, x1 + label_width)

            draw.rectangle([x1, y_top, x_right, y_top + label_height], fill=(173, 216, 230))
            draw.text((x1 + 6, y_top + 2), label_en, font=en_font, fill=(0, 0, 0))
            draw.text((x1 + 6, y_top + font_size + 4), si_name, font=si_font, fill=(0, 0, 0))

            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("BirdVision Live", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()