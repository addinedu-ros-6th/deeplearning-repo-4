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

        ####################################################
        """GUI에서 사람찾기 클릭하여, 캡쳐된 사진 위치로 설정하기"""
        # 제공된 참조 이미지 로드
        self.reference_image_path = "./face_images/face_image.jpg"
        self.reference_image = cv2.imread(self.reference_image_path)
        ####################################################

        # 참조 이미지의 얼굴 임베딩 생성
        try:
            self.ref_embedding = DeepFace.represent(
                img_path=self.reference_image,
                model_name='Facenet',
                enforce_detection=False
            )[0]["embedding"]

            # 리스트를 NumPy 배열로 변환
            self.ref_embedding = np.array(self.ref_embedding)
        except Exception as e:
            print(f"참조 이미지 임베딩 생성 오류: {e}")
            exit()

        # YOLOv8 얼굴 검출 모델 로드
        self.model = YOLO('./models/yolov8n-face.pt')  # 얼굴 검출용으로 학습된 모델

        self.image_format = 'jpg'  # 'jpg'로 바꾸면 JPEG로 인코딩

    def face_similarity(self, frame):
        self.frame=frame
        # YOLOv8을 사용하여 얼굴 검출
        results = self.model(self.frame)

        # 결과에서 얼굴 박스 추출
        for result in results:
            boxes = result.boxes  # 바운딩 박스 리스트
            for box in boxes:
                # 바운딩 박스 좌표 및 기타 정보 추출
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표
                conf = box.conf.item()  # 신뢰도 값을 스칼라로 변환

                # 신뢰도 임계값 확인
                if conf > 0.5:
                    # 얼굴 영역 추출
                    face_image = frame[y1:y2, x1:x2]

                    # 로컬 얼굴 임베딩 생성
                    try:
                        face_embedding = DeepFace.represent(
                            img_path=face_image,
                            model_name='Facenet',
                            enforce_detection=False
                        )[0]["embedding"]

                        # 리스트를 NumPy 배열로 변환
                        face_embedding = np.array(face_embedding)

                        # 코사인 유사도를 직접 계산하여 유사도 계산
                        dot_product = np.dot(self.ref_embedding, face_embedding)
                        norm_ref = np.linalg.norm(self.ref_embedding)
                        norm_face = np.linalg.norm(face_embedding)
                        cosine_similarity = dot_product / (norm_ref * norm_face)

                        # 코사인 유사도를 0~100% 범위로 변환
                        similarity = (cosine_similarity + 1) / 2 * 100

                        # 유사도가 90% 이상인 경우 AWS Rekognition으로 확인
                        if similarity >= 88:
                            # 얼굴 이미지를 AWS Rekognition에 보낼 수 있도록 인코딩
                            encode_format = '.jpg' if self.image_format == 'jpg' else '.png'
                            _, face_buffer = cv2.imencode(encode_format, face_image)
                            face_bytes = {"Bytes": face_buffer.tobytes()}

                            # AWS Rekognition으로 얼굴 비교
                            response = self.rekognition_client.compare_faces(
                                SourceImage={'Bytes': open(self.reference_image_path, 'rb').read()},
                                TargetImage=face_bytes,
                                SimilarityThreshold=95  # 유사도 기준값 설정
                            )

                            # Rekognition 결과 처리
                            if response['FaceMatches']:
                                for match in response['FaceMatches']:
                                    similarity_by_aws = match['Similarity']
                                    # 얼굴 영역에 사각형 그리기 및 유사도 표시
                                    cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(self.frame, f'Doble-check By AWS: {similarity_by_aws:.2f}%', (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                                    print(f"얼굴 일치: AWS 유사도 {similarity_by_aws:.2f}%")
                        else:
                            # 유사도가 90% 미만인 경우 (로컬 유사도가 낮음)
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(self.frame, f'Similarity: {similarity:.2f}%', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    except Exception as e:
                        print(f"얼굴 처리 오류: {e}")
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(self.frame, 'face sorted error', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


        # 비교 이미지 프레임에 삽입
        ref_img_resized = cv2.resize(self.reference_image, (150, 150))  # 크기를 조절
        self.frame[10:160, 10:160] = ref_img_resized  # 좌측 상단에 삽입
        

        return self.frame
