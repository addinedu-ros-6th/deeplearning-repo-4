import socket
import struct
import cv2
import numpy as np
import base64
import json
from ultralytics import YOLO

# 서버 설정
UDP_PORT = 9999
TCP_PORT = 8888
MAX_DGRAM = 65507  # UDP 패킷의 최대 크기
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# YOLOv8 모델 로드 (사전 학습된 모델 사용)
model = YOLO("yolov8n.pt")  # 경량 모델 yolov8n 사용, 더 성능을 원하면 yolov8m, yolov8l 등을 사용할 수 있음

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

def process_frame(frame):
    """
    YOLOv8 트래커를 사용하여 ID가 1인 객체의 중심 좌표 및 이동 속도를 결정하는 함수.
    """
    global move_twice_counter, move_twice_limit, move_thrice_counter, move_thrice_limit, loop_counter, loop_threshold

    # YOLOv8 모델을 사용하여 객체 감지 및 추적
    results = model.track(frame, persist=True)  # YOLOv8 추적 사용

    # 각 객체에 대해 ID가 1인 객체를 찾음
    for result in results:
        for track in result.boxes:
            if track.id is None:
                continue  # ID가 None인 경우 처리하지 않음
            
            object_id = int(track.id)  # 트래커로부터 부여된 객체 ID
            if object_id == 1:  # ID가 1인 객체만 처리
                # 바운딩 박스 좌표 계산
                x_min, y_min, x_max, y_max = track.xyxy[0].cpu().numpy()  # 좌표를 numpy 배열로 변환
                x_mid = int((x_min + x_max) / 2)
                y_mid = int((y_min + y_max) / 2)

                # 원의 중심에서 객체 중심까지의 거리 계산
                distance = ((x_mid - FRAME_WIDTH // 2) ** 2 + (y_mid - FRAME_HEIGHT // 2) ** 2) ** 0.5

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

                # 바운딩 박스와 중앙에 점 그리기
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

                # 바운딩 박스 상단에 객체 ID 출력
                cv2.putText(frame, f"ID: {object_id}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # 바운딩 박스 중앙에 점 그리기
                cv2.circle(frame, (x_mid, y_mid), 2, (0, 255, 0), 2)

                return (x_mid, y_mid, move_amount)

    return (-1, -1, 0)




def receive_json_data():
    """
    클라이언트로부터 JSON 데이터 수신 및 복원 함수.
    """
    buffer = b""
    while True:
        chunk, addr = udp_socket.recvfrom(MAX_DGRAM)
        is_last_chunk = struct.unpack('B', chunk[:1])[0]  # 첫 바이트가 마지막 청크 여부를 나타냄
        buffer += chunk[1:]  # 실제 데이터는 2번째 바이트부터

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

        # JSON에서 Base64로 인코딩된 프레임 복원
        frame_base64 = json_data['frame']

        # JSON으로부터 받은 데이터 출력
        print("수신된 JSON 데이터:")
        print(f"bg_num: {json_data['bg_num']}")
        print(f"frame_id: {json_data['frame_id']}")  # frame_id 출력
        print(f"br_code: {json_data['br_code']}")
        # print(f"frame 데이터 (일부): {frame_base64[:100]}...")  # 프레임 데이터의 처음 100글자만 출력

        # Base64 디코딩 및 프레임 복원
        frame_data = base64.b64decode(frame_base64)
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # YOLOv8을 사용하여 사람을 감지하고, ID가 1인 객체의 중앙을 추적
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
