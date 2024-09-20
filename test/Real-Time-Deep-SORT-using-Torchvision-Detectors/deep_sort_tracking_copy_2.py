"""
Deep SORT with various object detection models.

USAGE:
python deep_sort_tracking.py 
python deep_sort_tracking.py --threshold 0.5 --imgsz 320
python deep_sort_tracking.py --threshold 0.5 --model fasterrcnn_resnet50_fpn_v2
                                                     fasterrcnn_resnet50_fpn
                                                     fasterrcnn_mobilenet_v3_large_fpn
                                                     fasterrcnn_mobilenet_v3_large_320_fpn
                                                     fcos_resnet50_fpn
                                                     ssd300_vgg16
                                                     ssdlite320_mobilenet_v3_large
                                                     retinanet_resnet50_fpn
                                                     retinanet_resnet50_fpn_v2
"""
import torch
import torchvision
import cv2
import os
import time
import argparse
import numpy as np

from torchvision.transforms import ToTensor
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import convert_detections, annotate
from coco_classes import COCO_91_CLASSES

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument(
    '--imgsz', 
    default=None,
    help='Image size, e.g., 640 will resize images to 640x640',
    type=int
)
parser.add_argument(
    '--model',
    default='retinanet_resnet50_fpn_v2',
    help='Name of the object detection model',
    choices=[
        'fasterrcnn_resnet50_fpn_v2',
        'fasterrcnn_resnet50_fpn',
        'fasterrcnn_mobilenet_v3_large_fpn',
        'fasterrcnn_mobilenet_v3_large_320_fpn',
        'fcos_resnet50_fpn',
        'ssd300_vgg16',
        'ssdlite320_mobilenet_v3_large',
        'retinanet_resnet50_fpn',
        'retinanet_resnet50_fpn_v2'
    ]
)
parser.add_argument(
    '--threshold',
    default=0.9,
    help='Detection confidence threshold',
    type=float
)
parser.add_argument(
    '--embedder',
    default='mobilenet',
    help='Feature extractor for re-identification',
    choices=[
        "mobilenet",
        "torchreid",
        "clip_RN50",
        "clip_RN101",
        "clip_RN50x4",
        "clip_RN50x16",
        "clip_ViT-B/32",
        "clip_ViT-B/16"
    ]
)
parser.add_argument(
    '--show',
    action='store_true',
    help='Visualize results in real-time'
)
parser.add_argument(
    '--cls', 
    nargs='+',
    default=[1],
    help='COCO class indices to track',
    type=int
)
args = parser.parse_args()

# Set the random seed for reproducibility
np.random.seed(42)

# Create output directory if it doesn't exist
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))

print(f"Tracking: {[COCO_91_CLASSES[idx] for idx in args.cls]}")
print(f"Detector: {args.model}")
print(f"Re-ID embedder: {args.embedder}")

# Load the object detection model
model = getattr(torchvision.models.detection, args.model)(weights='DEFAULT')
model.eval().to(device)

# Initialize the Deep SORT tracker
tracker = DeepSort(max_age=90, embedder=args.embedder)

# Capture video from the webcam (device 0 is the default webcam)
cap = cv2.VideoCapture(2)

# Get the frame width, height, and frames per second from the webcam
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(5))

# Initialize frame count and total FPS tracker
frame_count = 0
total_fps = 0

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Resize the frame if an image size is specified
        if args.imgsz:
            resized_frame = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                (args.imgsz, args.imgsz)
            )
        else:
            resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to a tensor and move it to the correct device (GPU/CPU)
        frame_tensor = ToTensor()(resized_frame).to(device)

        # Start timing for FPS calculation
        start_time = time.time()
        
        # Perform object detection
        with torch.no_grad():
            detections = model([frame_tensor])[0]

        # Convert detections to Deep SORT format
        detections = convert_detections(detections, args.threshold, args.cls)
    
        # Update the tracker with the detections
        tracks = tracker.update_tracks(detections, frame=frame)
    
        # Calculate FPS
        fps = 1 / (time.time() - start_time)
        total_fps += fps
        frame_count += 1

        # Annotate the frame with bounding boxes and tracking IDs
        if len(tracks) > 0:
            frame = annotate(tracks, frame, resized_frame, frame_width, frame_height, COLORS)

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the output in real-time if specified
        if args.show:
            cv2.imshow("Webcam Output", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        break

# Release resources and close any open windows
cap.release()
cv2.destroyAllWindows()
