from ultralytics import YOLO 
import cv2
import os

class WetFloorDetect:
    def __init__(self):
        self.model = YOLO("./bestWF.pt")
        
    def inference_WF_Detect(self, frame):
        results = self.model.predict(frame, verbose = False)
        return results
        