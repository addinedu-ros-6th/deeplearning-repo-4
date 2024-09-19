import cv2
import boto3
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# AWS Rekognition 클라이언트 설정
rekognition_client = boto3.client('rekognition', region_name='ap-northeast-2')

# 로컬 이미지를 RekognitionImage 객체로 로드
reference_image_path = "./GUI/face_image/face_20240919_162511.jpg"
try:
    with open(reference_image_path, "rb") as img_file:
        reference_image = {"Bytes": img_file.read()}
except FileNotFoundError:
    print("Reference image not found.")
    exit()

# 웹캠 설정
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR을 RGB로 변환 (OpenCV는 기본적으로 BGR을 사용)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 웹캠에서 캡처된 프레임을 Rekognition에서 사용할 수 있도록 변환
    _, buffer = cv2.imencode('.jpg', frame_rgb)
    webcam_image = {"Bytes": buffer.tobytes()}

    try:
        # AWS Rekognition으로 얼굴 비교
        response = rekognition_client.compare_faces(
            SourceImage=reference_image,
            TargetImage=webcam_image,
            SimilarityThreshold=20
        )

        # 얼굴 비교 결과가 있는지 확인
        print(response)

        if response['FaceMatches']:
            for match in response['FaceMatches']:
                face = match['Face']
                similarity = match['Similarity']

                # 얼굴 위치 좌표 얻기
                box = face['BoundingBox']
                left = int(box['Left'] * frame.shape[1])
                top = int(box['Top'] * frame.shape[0])
                width = int(box['Width'] * frame.shape[1])
                height = int(box['Height'] * frame.shape[0])

                # 얼굴 영역에 사각형 그리기
                cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
                cv2.putText(frame, f'Similarity: {similarity:.2f}%', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        else:
            print("No faces matched.")
    except rekognition_client.exceptions.InvalidParameterException as e:
        print(f"InvalidParameterException: {e}")
        break

    # 비교 이미지 프레임에 삽입
    ref_img = cv2.imread(reference_image_path)
    ref_img_resized = cv2.resize(ref_img, (150, 150))  # 크기를 조절
    frame[10:160, 10:160] = ref_img_resized  # 좌측 상단에 삽입

    # 프레임 출력
    cv2.imshow('Webcam Face Comparison', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 및 해제
cap.release()
cv2.destroyAllWindows()
