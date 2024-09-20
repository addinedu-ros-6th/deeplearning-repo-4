import socket
import struct
import cv2
import numpy as np
import base64
import json
from ultralytics import YOLO
import torchreid
import torch

# OSNet 모델 초기화 (사람 재식별을 위해)
model_reid = torchreid.models.build_model(
    name='osnet_x1_0',  # 사용할 OSNet 모델 선택
    num_classes=1000,
    pretrained=True
)
model_reid.eval()

# 서버 설정
UDP_PORT = 9999
TCP_PORT = 8888
MAX_DGRAM = 65507  # UDP 패킷의 최대 크기
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# YOLOv8 모델 로드 (사람만 감지하도록 설정)
model = YOLO("yolov8n-person.pt")

# UDP 소켓 설정 (클라이언트로부터 영상 수신)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
udp_socket.bind(('0.0.0.0', UDP_PORT))

# TCP 소켓 설정 (좌표 및 이동 값 전송)
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
tcp_socket.bind(('0.0.0.0', TCP_PORT))
tcp_socket.listen(5)

# 원 크기 설정
small_circle_radius = 30
medium_circle_radius = 100
large_circle_radius = 160
extra_large_circle_radius = 220

# 루프 카운터 설정
loop_counter = 0
loop_threshold = 2

# 이동 횟수를 추적할 변수
move_twice_counter = 0
move_twice_limit = 8
move_thrice_counter = 0
move_thrice_limit = 5

# 이미 인식된 사람을 저장하는 딕셔너리와 ID 초기화
reid_dict = {}
next_id = 1  # 다음에 부여할 ID를 추적하는 변수

# OSNet을 사용하여 특징을 추출하는 함수
def extract_features(image):
    # 이미지 크기를 OSNet 입력 크기로 조정
    image = cv2.resize(image, (256, 128))
    image = image.astype(np.float32) / 255.0
    image = image.transpose((2, 0, 1))  # C x H x W 형태로 변경
    image = torch.tensor(image).unsqueeze(0)  # 배치 차원 추가
    with torch.no_grad():
        features = model_reid(image).cpu().numpy()
    return features[0]

def process_frame(frame):
    global move_twice_counter, move_twice_limit, move_thrice_counter, move_thrice_limit, loop_counter, loop_threshold
    global next_id  # 전역 변수로 사용

    # YOLOv8 모델을 사용하여 사람만 감지
    results = model.track(frame, persist=True, conf=0.3)

    # 각 객체에 대해 처리
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.id is None:
                continue  # ID가 None인 경우 처리하지 않음

            # 바운딩 박스 좌표 및 기타 정보 추출
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표
            conf = box.conf.item()  # 신뢰도 값을 스칼라로 변환
            x_mid = (x1 + x2) // 2
            y_mid = (y1 + y2) // 2

            # 바운딩 박스 내의 사람 이미지 추출
            person_image = frame[y1:y2, x1:x2]
            if person_image.size == 0:  # 잘못된 바운딩 박스를 처리
                continue

            # 추출한 이미지에서 특징 벡터를 생성
            person_features = extract_features(person_image)

            # 재식별을 수행하여 이전에 인식된 사람인지 확인
            matched_id = -1
            max_similarity = 0.0
            for reid_id, reid_features in reid_dict.items():
                similarity = np.dot(person_features, reid_features) / (
                    np.linalg.norm(person_features) * np.linalg.norm(reid_features)
                )
                if similarity > 1.0 and similarity > max_similarity:
                    matched_id = reid_id
                    max_similarity = similarity

            # reid_dict 업데이트 또는 추가
            if matched_id == -1:
                matched_id = next_id
                reid_dict[matched_id] = person_features
                next_id += 1  # 다음에 사용할 ID를 증가
            else:
                reid_dict[matched_id] = person_features  # 특징 업데이트

            # 프레임의 중앙과 객체 중심까지의 거리 계산
            distance = ((x_mid - FRAME_WIDTH // 2) ** 2 + (y_mid - FRAME_HEIGHT // 2) ** 2) ** 0.5

            # 이동량 결정
            move_amount = 0
            if distance <= small_circle_radius:
                move_amount = 0
                move_twice_counter = 0
                move_thrice_counter = 0
            elif distance <= medium_circle_radius:
                loop_counter += 1
                if loop_counter >= loop_threshold:
                    move_amount = 1
                    loop_counter = 0
                    move_twice_counter = 0
                    move_thrice_counter = 0
            elif distance <= large_circle_radius:
                move_amount = 1
                move_twice_counter = 0
                move_thrice_counter = 0
            elif distance <= extra_large_circle_radius:
                if move_twice_counter >= move_twice_limit:
                    move_amount = 1
                else:
                    move_amount = 2
                    move_twice_counter += 1
            else:
                if move_thrice_counter >= move_thrice_limit:
                    move_amount = 2
                else:
                    move_amount = 3
                    move_thrice_counter += 1

            # 바운딩 박스와 ID 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 바운딩 박스 그리기
            cv2.putText(frame, f"ReID: {matched_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # ID 표시

            # 중앙점 그리기
            cv2.circle(frame, (x_mid, y_mid), 2, (0, 255, 0), 2)

            return (x_mid, y_mid, move_amount)

    return (-1, -1, 0)

# 클라이언트로부터 JSON 데이터 수신 및 복원 함수
def receive_json_data():
    """
    클라이언트로부터 JSON 데이터 수신 및 복원 함수.
    """
    buffer = b""
    while True:
        chunk, addr = udp_socket.recvfrom(MAX_DGRAM)
        is_last_chunk = struct.unpack('B', chunk[:1])[0]  # 첫 바이트가 마지막 청크 여부를 나타냄
        buffer += chunk[1:]  # 실제 데이터는 두 번째 바이트부터 시작

        if is_last_chunk:
            break

    # 수신된 JSON 데이터 복원
    try:
        json_data = json.loads(buffer.decode('utf-8'))
        return json_data
    except Exception as e:
        print(f"JSON 데이터 복원 오류: {e}")
        return None

try:
    while True:
        print("프레임 수신 대기 중...")

        # JSON 데이터 수신
        json_data = receive_json_data()
        if json_data is None:
            continue

        # Base64로 인코딩된 프레임 복원
        frame_base64 = json_data['frame']

        # 수신된 데이터 출력
        print("수신된 JSON 데이터:")
        print(f"bg_num: {json_data['bg_num']}")
        print(f"frame_id: {json_data['frame_id']}")
        print(f"br_code: {json_data['br_code']}")

        # Base64 디코딩 및 프레임 복원
        frame_data = base64.b64decode(frame_base64)
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # YOLOv8을 사용하여 사람을 감지하고, ReID를 사용하여 식별
        x_mid, y_mid, move_amount = process_frame(frame)

        conn, client_address = tcp_socket.accept()
        print(f"클라이언트 연결됨: {client_address}")
        with conn:
            direction_data = f"X{x_mid}Y{y_mid}M{move_amount}".encode('utf-8')
            conn.sendall(direction_data)
            print(f"좌표 전송: X={x_mid}, Y={y_mid}, Move={move_amount}")

        center_x, center_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
        cv2.circle(frame, (center_x, center_y), extra_large_circle_radius, (255, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), large_circle_radius, (255, 0, 0), 2)
        cv2.circle(frame, (center_x, center_y), medium_circle_radius, (0, 255, 255), 2)
        cv2.circle(frame, (center_x, center_y), small_circle_radius, (0, 255, 0), 2)

        cv2.imshow('Received Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    udp_socket.close()
    tcp_socket.close()
    cv2.destroyAllWindows()
