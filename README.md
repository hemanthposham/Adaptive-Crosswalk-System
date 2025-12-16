# 🚦 Adaptive Crosswalk System

**Adaptive Crosswalk** is a smart traffic management solution designed to enhance pedestrian safety and optimize crosswalk efficiency in smart cities. It utilizes real-time data from RGB and thermal cameras, communicates with autonomous vehicles using V2I (Vehicle-to-Infrastructure), and dynamically adjusts pedestrian signals based on environmental conditions and pedestrian presence.

---

## 🌟 Key Features

* **Sensor Fusion**: Combines RGB and thermal camera feeds for accurate pedestrian detection in all lighting and weather conditions.
* **Real-Time Coordination**: Communicates with autonomous vehicles using V2I protocols to manage crosswalk access.
* **Adaptive Timing**: Dynamically adjusts signal timing based on pedestrian density and vehicle flow.
* **Smart City Integration**: Can be deployed as part of urban smart infrastructure for scalable traffic management.

---

## 🛠️ Technologies Used

* Python
* OpenCV
* YOLOv5 (for pedestrian detection)
* MQTT or HTTP for V2I communication
* Raspberry Pi or Jetson Nano (Edge processing)
* RGB + Thermal Camera Modules

---
## 🔗 Pre-trained Weights

You can download the YOLOv3 weights from this official link:

[Download yolov3.weights](https://pjreddie.com/darknet/yolo/)

---
## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/adaptive-crosswalk.git
cd adaptive-crosswalk
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Connect Hardware

* Connect RGB and thermal cameras to your edge device.
* Configure camera streams in `config.py`.

### 4. Run the System

```bash
python main.py
```

### 5. (Optional) Set Up V2I Communication

* Configure V2I protocol parameters in `v2i_config.py`
* Deploy corresponding vehicle-side script or emulator

---

## 📸 Sample Output
![det_test_image2](https://github.com/user-attachments/assets/730c8362-6df7-4aee-93d5-49fe0fee6101)


*Pedestrian detection using fused thermal and RGB data.*

---

## 🤝 Contributing

Contributions are welcome! To contribute:

* Fork the repo
* Create a feature branch
* Submit a Pull Request with a clear description

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

Have suggestions or questions? Open an issue or reach out via GitHub.
