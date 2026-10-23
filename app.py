from ultralytics import YOLO
from PIL import Image, ImageDraw
import os

# ----------------------
# Config
# ----------------------
IMAGE_PATH = "sample/library.jpg"
OUTPUT_PATH = "output/result.jpg"
ZONE_NAME = "Library Study Area"
CAPACITY = 12

# ----------------------
# Load model
# ----------------------
model = YOLO("yolov8s.pt")  # pretrained nano model

# ----------------------
# Run detection
# ----------------------
results = model(
    IMAGE_PATH,
    conf=0.25,
    imgsz=960,
    classes=[0],
    iou=0.4,
    augment=False
)

# Open image for drawing
image = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(image)

person_count = 0

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # class 0 = person in COCO
        if cls_id == 0:
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 15)), f"person {conf:.2f}", fill="red")

# ----------------------
# Calculate occupancy
# ----------------------
occupied = person_count
available = max(CAPACITY - occupied, 0)
occupancy_rate = occupied / CAPACITY * 100 if CAPACITY > 0 else 0

if occupancy_rate < 40:
    status = "Not Busy"
elif occupancy_rate < 70:
    status = "Moderate"
else:
    status = "Busy"

# ----------------------
# Save output image
# ----------------------
os.makedirs("output", exist_ok=True)
image.save(OUTPUT_PATH)

# ----------------------
# Print result
# ----------------------
print("===== LibSpace AI Result =====")
print(f"Zone: {ZONE_NAME}")
print(f"Capacity: {CAPACITY}")
print(f"People detected: {person_count}")
print(f"Estimated occupied seats: {occupied}")
print(f"Estimated available seats: {available}")
print(f"Occupancy: {occupancy_rate:.1f}%")
print(f"Status: {status}")
print(f"Output image saved to: {OUTPUT_PATH}")
