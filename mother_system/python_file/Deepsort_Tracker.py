import torch
import torchvision
import cv2
import os
import time
import argparse
import numpy as np

from torchvision.transforms import ToTensor
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils_1 import convert_detections, annotate
from coco_classes import COCO_91_CLASSES

class DeepSortTrack:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--imgsz', default=None, help='image resize, 640 will resize images to 640x640', type=int)
        parser.add_argument('--model', default='retinanet_resnet50_fpn_v2', help='model name', choices=[
            'fasterrcnn_resnet50_fpn_v2',
            'fasterrcnn_resnet50_fpn',
            'fasterrcnn_mobilenet_v3_large_fpn',
            'fasterrcnn_mobilenet_v3_large_320_fpn',
            'fcos_resnet50_fpn',
            'ssd300_vgg16',
            'ssdlite320_mobilenet_v3_large',
            'retinanet_resnet50_fpn',
            'retinanet_resnet50_fpn_v2'
        ])
        parser.add_argument('--threshold', default=0.9, help='score threshold to filter out detections', type=float)
        parser.add_argument('--embedder', default='torchreid', help='type of feature extractor to use', choices=[
            "mobilenet", "torchreid", "clip_RN50", "clip_RN101", "clip_RN50x4", "clip_RN50x16", "clip_ViT-B/32", "clip_ViT-B/16"
        ])
        parser.add_argument('--show', action='store_true', help='visualize results in real-time on screen', default=True)

        parser.add_argument('--cls', nargs='+', default=[1], help='which classes to track', type=int)
        self.args = parser.parse_args()

        np.random.seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.COLORS = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))

        print(f"Tracking: {[COCO_91_CLASSES[idx] for idx in self.args.cls]}")
        print(f"Detector: {self.args.model}")
        print(f"Re-ID embedder: {self.args.embedder}")

        # Load the model
        self.model = getattr(torchvision.models.detection, self.args.model)(weights='DEFAULT')
        self.model.eval().to(self.device)

        # Initialize DeepSort tracker
        self.tracker = DeepSort(max_age=90, embedder=self.args.embedder)

        # Open webcam
        self.cap = cv2.VideoCapture(0)

    def inference_DT_Detect(self, frame):
        # 프레임을 적절한 포맷으로 변환
        frame_tensor = ToTensor()(frame).to(self.device)
        
        # 모델을 통해 추론을 수행
        self.model.eval()  # 모델을 평가 모드로 설정
        with torch.no_grad():
            results = self.model([frame_tensor])  # predict 대신 forward 방식으로 추론 수행

        return results[0]

    def run(self):
        frame_count = 0
        total_fps = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.args.imgsz:
                    resized_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.args.imgsz, self.args.imgsz))
                else:
                    resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_tensor = ToTensor()(resized_frame).to(self.device)

                start_time = time.time()
                with torch.no_grad():
                    detections = self.model([frame_tensor])[0]

                detections = convert_detections(detections, self.args.threshold, self.args.cls)

                tracks = self.tracker.update_tracks(detections, frame=frame)

                fps = 1 / (time.time() - start_time)
                total_fps += fps
                frame_count += 1

                if len(tracks) > 0:
                    frame = annotate(tracks, frame, resized_frame, frame.shape[1], frame.shape[0], self.COLORS)

                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if self.args.show:
                    cv2.imshow("Webcam Output", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def run_tracking(self, frame):
        frame_count = 0
        total_fps = 0
        
        frame = frame
            
        if self.args.imgsz:
            resized_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.args.imgsz, self.args.imgsz))
        else:
            resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_tensor = ToTensor()(resized_frame).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            detections = self.model([frame_tensor])[0]

        detections = convert_detections(detections, self.args.threshold, self.args.cls)

        tracks = self.tracker.update_tracks(detections, frame=frame)

        fps = 1 / (time.time() - start_time)
        total_fps += fps
        frame_count += 1

        if len(tracks) > 0:
            frame = annotate(tracks, frame, resized_frame, frame.shape[1], frame.shape[0], self.COLORS)

        return frame