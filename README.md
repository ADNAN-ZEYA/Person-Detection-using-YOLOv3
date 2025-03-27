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
