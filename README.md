# Person Detection using YOLOv3

This project implements real-time **person detection** using the **YOLOv3** object detection model and OpenCV in Python. The system captures video from a webcam, processes each frame to detect persons, and displays bounding boxes around detected individuals.

---

## Features
- **Real-time person detection** using YOLOv3
- **OpenCV-based webcam feed processing**
- **Non-Maximum Suppression (NMS)** to reduce redundant bounding boxes
- **Confidence thresholding** to filter out weak detections

---

## Requirements
Before running the script, ensure you have the following dependencies installed:

```sh
pip install opencv-python numpy
```

You also need the **YOLOv3 weight file (`yolov3.weights`)** and configuration file (`yolov3.cfg`), along with the **COCO class labels (`coco.names`)**.

Download them from:
- YOLOv3 weights: [Download](https://pjreddie.com/media/files/yolov3.weights)
- YOLOv3 configuration: [Download](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- COCO labels: [Download](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

Place these files in the same directory as the script.

---

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/person-detection.git
   cd person-detection
   ```
2. Ensure you have downloaded the required YOLO files.
3. Run the script:
   ```sh
   python person_detection.py
   ```
4. Press `q` to exit the webcam window.

---

## How It Works
1. **Loads the YOLOv3 model and class labels** from the provided files.
2. **Captures live video** from the webcam.
3. **Preprocesses frames** and feeds them to YOLO.
4. **Detects objects and filters out persons** based on confidence scores.
5. **Applies Non-Maximum Suppression (NMS)** to remove duplicate boxes.
6. **Draws bounding boxes** and labels on detected persons.
7. **Displays the processed frames** in a window.

---

## Future Improvements
- Implement **YOLOv4-tiny** for faster inference
- Add **multi-camera support**
- Log detected persons with timestamps
- Convert to a **Flask app for web deployment**

---

## License
This project is open-source and available under the **MIT License**.

---

## Author
ADNAN ZEYA-(https://github.com/your-username)
