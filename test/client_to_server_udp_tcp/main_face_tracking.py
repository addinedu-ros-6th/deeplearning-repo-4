import socket
import struct
import cv2
import mediapipe as mp
import numpy as np
import base64
import json

# 서버 설정
UDP_PORT = 9999
TCP_PORT = 8888
MAX_DGRAM = 65507  # UDP 패킷의 최대 크기
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MediaPipe 얼굴 감지기 초기화
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

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
    얼굴을 감지하고, 얼굴의 중심 좌표 및 이동 속도를 결정하는 함수.
    """
    global move_twice_counter, move_twice_limit, move_thrice_counter, move_thrice_limit, loop_counter, loop_threshold
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    center_x, center_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x_mid = int((bboxC.xmin + bboxC.width / 2) * FRAME_WIDTH)
            y_mid = int((bboxC.ymin + bboxC.height / 2) * FRAME_HEIGHT)

            distance = ((x_mid - center_x) ** 2 + (y_mid - center_y) ** 2) ** 0.5

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

            cv2.circle(frame, (x_mid, y_mid), 2, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(bboxC.xmin * FRAME_WIDTH), int(bboxC.ymin * FRAME_HEIGHT)),
                          (int((bboxC.xmin + bboxC.width) * FRAME_WIDTH), int((bboxC.ymin + bboxC.height) * FRAME_HEIGHT)),
                          (0, 0, 255), 2)

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

        # JSON으로부터 받은 데이터 출력 (여기에 print 문 추가)
        print("수신된 JSON 데이터:")
        print(f"bg_num: {json_data['bg_num']}")
        print(f"frame_time: {json_data['frame_time']}")
        print(f"br_code: {json_data['br_code']}")
        # print(f"frame 데이터 (일부): {frame_base64[:100]}...")  # 프레임 데이터의 처음 100글자만 출력

        # Base64 디코딩 및 프레임 복원
        frame_data = base64.b64decode(frame_base64)
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)


        # 얼굴 감지 및 이동 값 계산
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
