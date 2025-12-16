import cv2
import time
import torch
import numpy as np
from darknet import Darknet
from util import load_classes, write_results
import argparse
import os
from sort.sort import Sort
import csv
from datetime import datetime
import json

# ---------------------- Argument Parser ----------------------
parser = argparse.ArgumentParser(description="YOLOv3 Video Detection")
parser.add_argument('--video', type=str, required=True, help="Path to the video file")
args = parser.parse_args()
video_path = args.video

# ---------------------- Load Model ----------------------
CONFIDENCE = 0.5
NMS_THRESH = 0.4
INPUT_DIM = 416

model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
model.net_info["height"] = INPUT_DIM
model.eval()

# ---------------------- Class Labels ----------------------
classes = load_classes("data/coco.names")
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

# ---------------------- Load Video ----------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Couldn't open video.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Create output folder if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")

# Set up video writer
output_path = "output/output_video.avi"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# ---------------------- Tracking Variables ----------------------
tracker = Sort()  
person_data = {}

# ---------------------- PSM Logs ----------------------
psm_log_path = "output/psm_log.csv"
psm_file = open(psm_log_path, mode='w', newline='')
psm_writer = csv.writer(psm_file)
psm_writer.writerow(["Timestamp", "ID", "TTC (s)", "X", "Y", "Risk Level"])

psm_json_log = open("output/psm_log.json", "w")

# ---------------------- Process Frame-by-Frame ----------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    crosswalk_y = int(frame.shape[0] * 0.65)

    img = cv2.resize(frame, (INPUT_DIM, INPUT_DIM))
    img_rgb = img[:, :, ::-1].transpose((2, 0, 1))
    img_rgb = np.ascontiguousarray(img_rgb)
    img_tensor = torch.from_numpy(img_rgb).float().div(255.0).unsqueeze(0)

    with torch.no_grad():
        predictions = model(img_tensor, CUDA=False)
        predictions = write_results(predictions, CONFIDENCE, num_classes=80, nms=True, nms_conf=NMS_THRESH)

    if predictions is not None:
        predictions[:, [1, 3]] *= frame.shape[1] / INPUT_DIM
        predictions[:, [2, 4]] *= frame.shape[0] / INPUT_DIM

        detections = []

        for pred in predictions:
            cls = int(pred[-1])
            if classes[cls] != "person":
                continue
            x1, y1, x2, y2 = map(float, pred[1:5])
            conf = float(pred[5])
            detections.append([x1, y1, x2, y2, conf])

        dets = np.array(detections)
        if dets.size == 0:
            dets = np.empty((0, 5))

        tracked_objects = tracker.update(dets)

        for x1, y1, x2, y2, obj_id in tracked_objects:
            c1 = (int(x1), int(y1))
            c2 = (int(x2), int(y2))
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            cv2.rectangle(frame, c1, c2, (0, 255, 0), 2)
            cv2.putText(frame, f"ID {int(obj_id)}", (c1[0], c1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            current_time = time.time()
            if obj_id in person_data:
                prev_y, prev_time = person_data[obj_id]
                delta_y = prev_y - center_y
                delta_t = current_time - prev_time

                if delta_t > 0 and delta_y > 0:
                    speed = delta_y / delta_t
                    distance = abs(center_y - crosswalk_y)
                    ttc = distance / speed if speed > 1 else float('inf')

                    if ttc < 3:
                        cv2.putText(frame, f"TTC: {ttc:.1f}s", (c1[0], c2[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        psm_writer.writerow([
                            datetime.now().isoformat(),
                            int(obj_id),
                            round(ttc, 2),
                            center_x,
                            center_y,
                            "HIGH"
                        ])

                        psm = {
                            "type": "PSM",
                            "timestamp": datetime.now().isoformat(),
                            "pedestrian_id": int(obj_id),
                            "ttc": round(ttc, 2),
                            "position": [center_x, center_y],
                            "risk_level": "HIGH"
                        }

                        print("\U0001F697 Sending PSM to vehicle system →", json.dumps(psm, indent=2))
                        psm_json_log.write(json.dumps(psm) + "\n")

                    else:
                        print(f"🟢 ID {int(obj_id)} | TTC: {ttc:.2f} sec")

            person_data[obj_id] = (center_y, current_time)
            cv2.circle(frame, (center_x, center_y), 4, (255, 255, 255), -1)

    cv2.line(frame, (0, crosswalk_y), (frame.shape[1], crosswalk_y), (0, 255, 255), 2)
    out.write(frame)
    cv2.imshow("YOLOv3 - Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------- Cleanup ----------------------
cap.release()
out.release()
cv2.destroyAllWindows()
psm_json_log.close()
psm_file.close()
print(f"✅ PSMs saved to {psm_log_path}")
print(f"\n✅ Detection complete. Output saved at: {output_path}")
