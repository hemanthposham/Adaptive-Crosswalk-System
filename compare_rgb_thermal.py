import cv2
import torch
import numpy as np
from darknet import Darknet
from util import load_classes, write_results
import os
import csv
import argparse

# ------------------- Configuration -------------------
CONFIDENCE = 0.5
NMS_THRESH = 0.4
INPUT_DIM = 416

# ------------------- Load YOLO Model -------------------
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
model.net_info["height"] = INPUT_DIM
model.eval()

classes = load_classes("data/coco.names")

# ------------------- Prepare Logging -------------------
os.makedirs("output", exist_ok=True)
log_file = open("output/comparison_log.csv", "w", newline="")
writer = csv.writer(log_file)
writer.writerow(["Frame", "RGB Detections", "Thermal Detections", "Combined (OR)"])

# ------------------- Load Videos -------------------
parser = argparse.ArgumentParser()
parser.add_argument('--rgb', required=True, help="Path to RGB video")
parser.add_argument('--thermal', required=True, help="Path to Thermal video")
args = parser.parse_args()

cap_rgb = cv2.VideoCapture(args.rgb)
cap_thermal = cv2.VideoCapture(args.thermal)


frame_id = 0

def detect_persons(frame):
    img = cv2.resize(frame, (INPUT_DIM, INPUT_DIM))
    img = img[:, :, ::-1].transpose((2, 0, 1))  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor, CUDA=False)
        output = write_results(output, CONFIDENCE, num_classes=80, nms=True, nms_conf=NMS_THRESH)

    if output is None:
        return 0
    return sum([1 for det in output if int(det[-1]) == classes.index("person")])

# ------------------- Main Loop -------------------
while cap_rgb.isOpened() and cap_thermal.isOpened():
    ret_rgb, frame_rgb = cap_rgb.read()
    ret_thermal, frame_thermal = cap_thermal.read()
    if not ret_rgb or not ret_thermal:
        break

    # Convert thermal to 3-channel if it's grayscale
    if len(frame_thermal.shape) == 2 or frame_thermal.shape[2] == 1:
        frame_thermal = cv2.cvtColor(frame_thermal, cv2.COLOR_GRAY2BGR)

    rgb_count = detect_persons(frame_rgb)
    thermal_count = detect_persons(frame_thermal)
    combined_count = max(rgb_count, thermal_count)

    writer.writerow([frame_id, rgb_count, thermal_count, combined_count])

    # Optional: Show both frames
    display = np.hstack((frame_rgb, frame_thermal))
    cv2.imshow("RGB | Thermal", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# ------------------- Cleanup -------------------
cap_rgb.release()
cap_thermal.release()
cv2.destroyAllWindows()
log_file.close()
print("✅ Comparison complete. Log saved to output/comparison_log.csv")
