# Person-Detection-using-YOLOv3
This project implements real-time person detection using the YOLOv3 object detection model and OpenCV in Python. The system captures video from a webcam, processes each frame to detect persons, and displays bounding boxes around detected individuals.

Features

Real-time person detection using YOLOv3

OpenCV-based webcam feed processing

Non-Maximum Suppression (NMS) to reduce redundant bounding boxes

Confidence thresholding to filter out weak detections

Requirements

Before running the script, ensure you have the following dependencies installed:

pip install opencv-python numpy

You also need the YOLOv3 weight file (yolov3.weights) and configuration file (yolov3.cfg), along with the COCO class labels (coco.names).

Download them from:

YOLOv3 weights: Download

YOLOv3 configuration: Download

COCO labels: Download

Place these files in the same directory as the script.

Usage

Clone the repository:

git clone https://github.com/your-username/person-detection.git
cd person-detection

Ensure you have downloaded the required YOLO files.

Run the script:

python person_detection.py

Press q to exit the webcam window.

How It Works

Loads the YOLOv3 model and class labels from the provided files.

Captures live video from the webcam.

Preprocesses frames and feeds them to YOLO.

Detects objects and filters out persons based on confidence scores.

Applies Non-Maximum Suppression (NMS) to remove duplicate boxes.

Draws bounding boxes and labels on detected persons.

Displays the processed frames in a window.

Future Improvements

Implement YOLOv4-tiny for faster inference

Add multi-camera support

Log detected persons with timestamps

Convert to a Flask app for web deployment

License

This project is open-source and available under the MIT License.
