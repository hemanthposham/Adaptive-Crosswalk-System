import cv2
import torch
import numpy as np
from darknet import Darknet
from util import load_classes, write_results
import os
import csv
import argparse
from datetime import datetime
import time

# Load YOLOv3
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
model.net_info["height"] = 416
model.eval()
classes = load_classes("data/coco.names")

# Paths
parser = argparse.ArgumentParser(description="RGB vs Simulated Thermal Comparison")
parser.add_argument('--video', required=True, help="Path to input RGB video")
args = parser.parse_args()

video_path = args.video

os.makedirs("output", exist_ok=True)
base_name = os.path.splitext(os.path.basename(video_path))[0]
log_file = open(f"output/{base_name}_comparison.csv", "w", newline="")
writer = csv.writer(log_file)
writer.writerow(["Frame", "RGB Detections", "Simulated Thermal Detections", "Combined (OR)"])

# Video input
cap = cv2.VideoCapture(video_path)
frame_id = 0

person_data = {}  # {ID: (previous_center_y, previous_time)}
crosswalk_y = 0.7  # percentage of image height (adjustable)

# Detection function
def detect_persons(frame, draw_on=None):
    img = cv2.resize(frame, (416, 416))
    img_orig = frame.copy()
    img_input = img[:, :, ::-1].transpose((2, 0, 1))
    img_input = np.ascontiguousarray(img_input)
    tensor = torch.from_numpy(img_input).float().div(255.0).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor, CUDA=False)
        output = write_results(output, 0.5, num_classes=80, nms=True, nms_conf=0.4)

    count = 0
    if output is not None:
        output[:, [1, 3]] *= frame.shape[1] / 416
        output[:, [2, 4]] *= frame.shape[0] / 416

        for det in output:
            cls = int(det[-1])
            if classes[cls] != "person":
                continue
            count += 1

            x1, y1, x2, y2 = map(int, det[1:5])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            current_time = time.time()
            crosswalk_pixel = int(frame.shape[0] * crosswalk_y)

            fake_id = center_x + center_y  # simple unique identifier per person per frame

            if fake_id in person_data:
                prev_y, prev_time = person_data[fake_id]
                dy = prev_y - center_y
                dt = current_time - prev_time
                speed = dy / dt if dt > 0 else 0
                distance = abs(center_y - crosswalk_pixel)

                if speed > 0:
                    ttc = distance / speed
                    if ttc < 3:
                        print(f"\U0001F6A8 PSM: Person detected at ({center_x},{center_y}) | TTC = {ttc:.2f}s | Risk: HIGH")
                    else:
                        print(f"🟢 Safe: Person ID {fake_id} | TTC = {ttc:.2f}s")
            person_data[fake_id] = (center_y, current_time)

            if draw_on is not None:
                label = classes[cls]
                color = (0, 255, 0)
                cv2.rectangle(draw_on, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw_on, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return count

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Simulate thermal image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal_sim = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Detect in RGB and simulated thermal
    rgb_frame_vis = frame.copy()
    thermal_vis = thermal_sim.copy()

    rgb_count = detect_persons(rgb_frame_vis, draw_on=rgb_frame_vis)
    thermal_count = detect_persons(thermal_vis, draw_on=thermal_vis)
    combined = max(rgb_count, thermal_count)

    writer.writerow([frame_id, rgb_count, thermal_count, combined])

    # Optional: show side-by-side for visual demo
    stacked = np.hstack((rgb_frame_vis, thermal_vis))
    cv2.imshow("RGB | Simulated Thermal", stacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# Cleanup
cap.release()
log_file.close()
cv2.destroyAllWindows()
print("✅ Comparison done. Results saved in output/simulated_comparison.csv")
