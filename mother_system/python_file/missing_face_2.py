import cv2
import boto3
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import os

class Missing_face:
    def __init__(self):
        # AWS Rekognition 클라이언트 설정
        self.rekognition_client = boto3.client('rekognition', region_name='ap-northeast-2')

        # YOLOv8 얼굴 검출 모델 로드
        self.model = YOLO('./models/yolov8n-face.pt')  # 얼굴 검출용으로 학습된 모델
        self.image_format = 'jpg'  # 'jpg'로 바꾸면 JPEG로 인코딩
        self.ref_embedding = None  # 참조 이미지 임베딩 초기화

    def load_reference_image(self, image_path):
            """
            참조 이미지 경로를 받아서 이미지를 로드하고 임베딩을 생성하는 함수
            """
            # 이미 임베딩이 생성되었으면 재사용
            if self.ref_embedding is not None:
                print("참조 이미지 임베딩이 이미 생성되었습니다. 다시 생성하지 않습니다.")
                return True

            try:
                reference_image = cv2.imread(image_path)
                if reference_image is None:
                    print(f"Error: Could not load image at {image_path}")
                    return False

                # 참조 이미지의 얼굴 임베딩 생성
                self.ref_embedding = DeepFace.represent(
                    img_path=reference_image,
                    model_name='Facenet',
                    enforce_detection=False
                )[0]["embedding"]

                # 리스트를 NumPy 배열로 변환
                self.ref_embedding = np.array(self.ref_embedding)
                print(f"참조 이미지 임베딩 생성 성공: {image_path}")
                return True
            except Exception as e:
                print(f"참조 이미지 임베딩 생성 오류: {e}")
                return False
    
    def face_similarity(self, frame):
        self.frame = frame
        # YOLOv8을 사용하여 얼굴 검출
        results = self.model(self.frame)

        # 결과에서 얼굴 박스 추출
        for result in results:
            boxes = result.boxes  # 바운딩 박스 리스트
            print(f"Detected boxes: {boxes}")
            for box in boxes:
                # 바운딩 박스 좌표 및 기타 정보 추출
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표
                conf = box.conf.item()  # 신뢰도 값을 스칼라로 변환
                print(f"Box coordinates: {x1}, {y1}, {x2}, {y2}, Confidence: {conf}")

                # 신뢰도 임계값 확인
                if conf > 0.5:
                    print(f"Face detected with confidence: {conf}")
                else:
                    print("No face detected with sufficient confidence.")

                    # 얼굴 영역 추출
                    face_image = frame[y1:y2, x1:x2]

                    # 로컬 얼굴 임베딩 생성
                    try:
                        face_embedding = DeepFace.represent(
                            img_path=face_image,
                            model_name='Facenet',
                            enforce_detection=False
                        )[0]["embedding"]
                        print("Face embedding generated successfully")

                        # 리스트를 NumPy 배열로 변환
                        face_embedding = np.array(face_embedding)

                        # 코사인 유사도를 직접 계산하여 유사도 계산
                        dot_product = np.dot(self.ref_embedding, face_embedding)
                        norm_ref = np.linalg.norm(self.ref_embedding)
                        norm_face = np.linalg.norm(face_embedding)
                        cosine_similarity = dot_product / (norm_ref * norm_face)

                        # 코사인 유사도를 0~100% 범위로 변환
                        similarity = (cosine_similarity + 1) / 2 * 100

                        # 유사도가 88% 이상인 경우 AWS Rekognition으로 확인
                        if similarity >= 88:
                            # AWS Rekognition 얼굴 비교 코드 (생략)

                            # 얼굴 영역에 사각형 그리기 및 유사도 표시
                            print(f"Drawing box at: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(self.frame, f"Similarity: {similarity:.2f}%", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            print("얼굴 영역 박스 출력")
                        else:
                            # 유사도가 88% 미만인 경우
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(self.frame, f"Low Similarity: {similarity:.2f}%", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                            print("얼굴 영역 박스 출력")

                    except Exception as e:
                        print(f"얼굴 처리 오류: {e}")
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(self.frame, 'Error processing face', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return self.frame

