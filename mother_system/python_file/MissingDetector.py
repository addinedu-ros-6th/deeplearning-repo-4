import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter

class MissingDetect:
    def __init__(self):
        self.Person_Detect_model = YOLO('models/yolov5n.pt')  # 사람 검출용
        self.Clothes_Segment_model = YOLO('models/best_clothes_seg.pt')  # 의류 세그멘테이션용
        
        # 색상 범위 정의
        self.color_ranges = {
            'red1': [(0, 100, 100), (10, 255, 255)],
            'red2': [(170, 100, 100), (180, 255, 255)],
            'orange': [(10, 100, 100), (25, 255, 255)],
            'yello': [(25, 100, 100), (35, 255, 255)],
            'green': [(35, 100, 100), (85, 255, 255)],
            'blue': [(85, 100, 100), (125, 255, 255)],
            'navy': [(125, 100, 100), (140, 255, 255)],
            'violet': [(140, 100, 100), (170, 255, 255)],
            'white': [(0, 0, 180), (180, 30, 255)],
            'gray': [(0, 0, 100), (180, 30, 180)],
            'black': [(0, 0, 1), (180, 50, 100)]
        }

    def get_dominant_color(self, masked_img):
        # Convert masked image to HSV
        hsv_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
        
        # Prepare an empty list to store color names
        detected_colors = []
        
        # Loop through each defined color range
        for color_name, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            # Create a mask for the current color
            color_mask = cv2.inRange(hsv_img, lower, upper)
            
            # Count non-zero pixels (i.e., pixels falling in this color range)
            count = np.sum(color_mask > 0)
            if count > 0:
                detected_colors.append((color_name, count))

        if detected_colors:
            # Get the most frequent color
            dominant_color = max(detected_colors, key=lambda item: item[1])[0]
            return dominant_color
        else:
            return "unknown"

    def inference_MP_Detect2(self, frame):
        # 옷 분할 결과 가져오기 (상의: cls=1, 하의: cls=2)
        Clothes_Seg_result = self.Clothes_Segment_model.predict(source=frame, conf=0.5, verbose=False)

        # 원본 프레임에 대한 복사본 생성
        annotated_frame = frame.copy()

        # 딕셔너리 초기화
        mask_dict = {}

        # 세그멘테이션 마스크를 확인하여 중심점과 색상 추출
        if hasattr(Clothes_Seg_result[0], 'masks') and Clothes_Seg_result[0].masks is not None:
            masks = Clothes_Seg_result[0].masks.data.cpu().numpy()  # 마스크 결과 가져오기
            classes = Clothes_Seg_result[0].boxes.cls.cpu().numpy()  # 클래스 결과 가져오기

            if len(masks) > 0:
                for idx, mask in enumerate(masks):
                    # 마스크의 좌표 계산 (중심점 찾기)
                    mask_indices = np.argwhere(mask == 1)  # 마스크가 1인 부분의 좌표
                    if len(mask_indices) > 0:
                        center_y, center_x = np.mean(mask_indices, axis=0).astype(int)  # 중심점 계산

                        # 마스크의 영역에서 해당 픽셀 추출
                        mask_pixels = frame[mask == 1]  # 마스크 영역에 해당하는 픽셀 추출
                        
                        # 마스크 영역에 해당하는 이미지를 얻기
                        masked_img = frame * np.stack([mask]*3, axis=-1)  # 마스크를 RGB 영역으로 확장

                        # 가장 빈번한 색상 이름 추출
                        dominant_color_name = self.get_dominant_color(masked_img)

                        # 클래스와 색상 정보를 딕셔너리에 추가
                        mask_dict[(center_x, center_y)] = (int(classes[idx]), dominant_color_name)

            else:
                print("No valid masks found.")
        else:
            print("No mask data available in Clothes_Seg_result.")  # 마스크가 None일 경우 예외처리
        
        #print(mask_dict)
        # 결과 프레임 반환 및 딕셔너리 반환
        return annotated_frame, mask_dict
