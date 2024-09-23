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
        # 사람 감지 결과 가져오기
        Person_Detect_result = self.Person_Detect_model.predict(source=frame, classes=[0], conf=0.3, verbose=False)
        # 옷 분할 결과 가져오기 (상의: cls=1, 하의: cls=2)
        Clothes_Seg_result = self.Clothes_Segment_model.predict(source=frame, conf=0.5, verbose=False)
        #print("--------------------------------------------------------------------------------")
        #print("CS_result : ", Clothes_Seg_result[0].boxes)
        #print("--------------------------------------------------------------------------------")
        # 원본 프레임에 대한 복사본 생성
        annotated_frame = frame.copy()

        # 색상 변수 초기화
        top_color, bottom_color = None, None

        # 세그멘테이션 마스크를 화면에 표시
        if hasattr(Clothes_Seg_result[0], 'masks') and Clothes_Seg_result[0].masks is not None:
            masks = Clothes_Seg_result[0].masks.data.cpu().numpy()  # 마스크 결과 가져오기
            classes = Clothes_Seg_result[0].boxes.cls.cpu().numpy()  # 클래스 결과 가져오기

            if len(masks) > 0:
                for i, mask in enumerate(masks):
                    mask_class = classes[i]
                    mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)  # 이진화 마스크 (threshold 적용)

                    # 마스크 내부의 색상 추출
                    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                    hsv_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

                    pixel_colors = []
                    for color_name, (lower, upper) in self.color_ranges.items():
                        lower_bound = np.array(lower)
                        upper_bound = np.array(upper)
                        mask_color = cv2.inRange(hsv_frame, lower_bound, upper_bound)
                        pixel_count = cv2.countNonZero(mask_color)
                        if pixel_count > 0:
                            pixel_colors.append(color_name)

                    # 최빈 색상 추출
                    if pixel_colors:
                        most_common_color = Counter(pixel_colors).most_common(1)[0][0]
                        if mask_class == 1:  # 상의
                            top_color = most_common_color
                            print(f"상의 색상: {top_color}")
                        elif mask_class == 2:  # 하의
                            bottom_color = most_common_color
                            print(f"하의 색상: {bottom_color}")
            else:
                print("No valid masks found.")
        else:
            print("No mask data available in Clothes_Seg_result.")  # 마스크가 None일 경우 예외처리

        # 사람 경계 상자 및 중심점 그리기
        if len(Person_Detect_result[0].boxes) > 0:
            person_boxes = Person_Detect_result[0].boxes.xyxy.tolist()

            for person_box in person_boxes:
                x1, y1, x2, y2 = person_box
                # 사람 경계 상자 그리기
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # 중심점 계산
                person_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                # 중심점 표시
                cv2.circle(annotated_frame, (int(person_center[0]), int(person_center[1])), 5, (0, 255, 0), -1)

                # 상의와 하의 색상 텍스트 표시
                if top_color:
                    cv2.putText(annotated_frame, f'top: {top_color}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if bottom_color:
                    cv2.putText(annotated_frame, f'bottom: {bottom_color}', (int(x1), int(y1) + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 결과 프레임 반환
        return annotated_frame

