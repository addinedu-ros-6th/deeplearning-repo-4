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

    def inference_MP_Detect2(self, frame):
        # 옷 분할 결과 가져오기 (상의: cls=1, 하의: cls=2)
        Clothes_Seg_result = self.Clothes_Segment_model.predict(source=frame, conf=0.5, verbose=False)
        mask_dict={}
        classes = Clothes_Seg_result[0].boxes.cls.cpu().numpy()

        for result in Clothes_Seg_result:
            # Segmentation 마스크 그리기
            if result.masks is not None:
                for idx, mask in enumerate(result.masks.data):
                    mask = mask.cpu().numpy()  # Mask를 numpy 배열로 변환
                    
                    colored_mask = (mask * 255).astype('uint8')  # Mask를 컬러화
                    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_GRAY2BGR)
                    
                    # mask 데이터를 frame bit 연산이 되도록 boot type 에서 uint8 type 으로 변환 후 masking 처리
                    mask255 = (mask * 255).astype("uint8")
                    masked_frame = cv2.bitwise_and(frame, frame, mask=mask255)

                    # masking 된 frame 색을 분석하기 위해 HSV 로 변환
                    hsv_masked_area = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
                    
                    # 각 색상 범위에 맞는 마스크 생성
                    color_areas = {}
                    for color_name, (lower, upper) in self.color_ranges.items():
                        lower_bound = np.array(lower, dtype=np.uint8)
                        upper_bound = np.array(upper, dtype=np.uint8)
                        color_mask = cv2.inRange(hsv_masked_area, lower_bound, upper_bound)

                        # 마스킹 된 영역의 픽셀수 계산
                        area_size = cv2.countNonZero(color_mask)
                        color_areas[color_name] = area_size

                    # 각 색상별 masking 된 것들 중 그 영역이 가장 큰 색상 정보 추출
                    largest_color = max(color_areas, key=color_areas.get)   
                    largest_color_value = color_areas[largest_color]

                    color_areas.pop(largest_color)

                    # 두번째로 masking 영역이 큰 색상 정보 추출
                    next_largest_color = max(color_areas, key=color_areas.get)
                    next_largest_color_value = color_areas[next_largest_color]

                    show_color = largest_color

                    # 만약 두번째 큰 색상의 masking area 가 일정 비율 이상 크면, 두번째 색상도 화면에 추가되도록 설정
                    if largest_color_value < next_largest_color_value * 3.5:
                        show_color = show_color + " and " + next_largest_color

                    # 마스크를 원본 영상에 적용
                    frame = cv2.addWeighted(frame, 1, colored_mask, 0, 0)  # Segmentation 마스크를 영상에 적용
                    
                    # 바운딩 박스 그리기
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # (중심 좌표 계산)
                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(cnt)
                        
                        # 바운딩 박스 그리기
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 초록색 박스
                        
                        # 바운딩 박스의 중심 좌표 계산
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # 색상 정보를 텍스트로 표시
                        text_x, text_y = x + w + 10, y - 10  # 박스 오른쪽 위에 텍스트 표시
                        color_text = f"Color: {show_color}"
                        cv2.putText(frame, color_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 중심 좌표를 키로 하고, 클래스와 색상을 튜플로 저장

                    mask_dict[(center_x, center_y)] = (int(classes[idx]), largest_color)

        return frame, mask_dict