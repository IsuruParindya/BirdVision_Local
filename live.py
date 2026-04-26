import os, cv2, json, torch, numpy as np, time
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

# ---- Load ----
classes = json.load(open(cls, encoding="utf-8"))
sinhala = json.load(open(si, encoding="utf-8"))

# =========================
# 🔥 AUTO DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ---- Classifier ----
clf = models.resnet50(weights=None)
clf.fc = nn.Linear(clf.fc.in_features, len(classes))
clf.load_state_dict(torch.load(pth, map_location=device))
clf = clf.to(device).eval()

# ---- YOLO ----
detector = YOLO(yolo_path)

# =========================
# 🔥 SMART CAMERA (FIXED)
# =========================
def get_camera():
    print("🔍 Scanning cameras...")

    working_caps = []

    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera detected at index {i}")
                working_caps.append((i, cap))
            else:
                cap.release()
        else:
            cap.release()

    if not working_caps:
        raise Exception("No camera found ❌")

    # Prefer external cam
    for idx, cap in working_caps:
        if idx != 0:
            print(f"Using external camera (index {idx}) ✅")
            return cap

    print("Using laptop camera (index 0) ⚠️")
    return working_caps[0][1]

cap = get_camera()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def classify(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(clf(x), dim=1)
        conf, pred = torch.max(probs,1)

    return classes[int(pred)], float(conf)

# ---- Tracker ----
def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, "legacy"):
        return cv2.legacy.TrackerCSRT_create()
    else:
        raise Exception("Install opencv-contrib-python")

# ---- Fonts ----
en_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
si_font = ImageFont.truetype("C:/Windows/Fonts/Nirmala.ttf", 16)

# ---- Tracking ----
tracker = None
tracking = False

# ---- Timing ----
last_detect_time = 0
DETECT_INTERVAL = 1.5  # faster response

last_label = ""
last_si = ""
last_conf = 0

print("Press Q to quit")

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    scale = 640 / w
    frame = cv2.resize(frame, (640, int(h * scale)))

    current_time = time.time()

    # ---- DETECTION ----
    if current_time - last_detect_time > DETECT_INTERVAL:

        results = detector(frame, conf=0.3, imgsz=320, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:

            b = results.boxes[0]
            x1,y1,x2,y2 = map(int, b.xyxy[0])

            crop = frame[y1:y2, x1:x2]

            if crop.size != 0:
                raw, conf = classify(crop)

                last_label = raw.replace("_"," ").title()
                last_si = sinhala.get(raw, "Unknown")
                last_conf = conf

                tracker = create_tracker()
                tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                tracking = True

        last_detect_time = current_time

    # ---- TRACKING ----
    if tracking and tracker is not None:
        success, box = tracker.update(frame)

        if success:
            x, y, w, h = map(int, box)
            x1, y1, x2, y2 = x, y, x+w, y+h

            label = f"{last_label} ({last_conf*100:.0f}%)"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil)

            padding = 6

            draw.text((x1 + padding, y1 - 30),
                      label, font=en_font, fill=(255,255,255))

            draw.text((x1 + padding, y1 - 10),
                      last_si, font=si_font, fill=(255,255,255))

            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        else:
            tracking = False

    cv2.imshow("BirdVision Live (FINAL)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()