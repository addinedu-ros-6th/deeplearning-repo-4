import socket
import cv2
import struct
import base64
import json
import time
import serial

from unittest.mock import MagicMock

# 아두이노가 없을 때 MagicMock을 사용하여 오류를 방지
try:
    import serial
except ModuleNotFoundError:
    serial = MagicMock()

# 서버 IP 및 포트 설정
SERVER_IP = '192.168.0.51'
UDP_PORT = 9999
TCP_PORT = 8888
MAX_DGRAM = 65507 - 1

# UDP 소켓 생성
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 웹캠 열기 (0번 카메라 사용)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 아두이노와의 직렬 통신 설정 (아두이노가 없으면 MagicMock이 대신 사용됨)
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
except serial.SerialException:
    print("아두이노에 연결할 수 없습니다. MagicMock을 사용합니다.")
    arduino = MagicMock()

# 로봇 번호 및 응답 코드 설정
bg_num = 1
br_code = 10

# frame_id 초기화
frame_id = 1

def send_frame_with_json(frame):
    """
    프레임과 관련 데이터를 JSON 형식으로 전송하는 함수.
    """

    global frame_id  # frame_id를 전역 변수로 설정하여 계속 증가할 수 있도록 함

    _, frame_jpg = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(frame_jpg).decode('utf-8')

    # JSON 데이터 구성
    json_data = {
        'bg_num': bg_num,
        'frame': frame_base64,
        'frame_id': frame_id,
        'br_code': br_code
    }

    # JSON 데이터를 직렬화하여 UTF-8로 인코딩
    json_data_str = json.dumps(json_data).encode('utf-8')

    size = len(json_data_str)
    num_chunks = (size // MAX_DGRAM) + 1

    for i in range(num_chunks):
        start = i * MAX_DGRAM
        end = min(start + MAX_DGRAM, size)
        chunk = json_data_str[start:end]

        is_last_chunk = 1 if i == num_chunks - 1 else 0
        udp_socket.sendto(struct.pack('B', is_last_chunk) + chunk, (SERVER_IP, UDP_PORT))

    frame_id += 1  # 프레임을 전송한 후 frame_id를 증가

def receive_and_move_servos():
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((SERVER_IP, TCP_PORT))
        coord_data = tcp_socket.recv(1024).decode('utf-8').strip()

        # JSON 형식으로 데이터 파싱
        direction_data = json.loads(coord_data)

        x_mid = direction_data.get('x', 0)
        y_mid = direction_data.get('y', 0)
        move_amount = direction_data.get('move', 0)
        mode = direction_data.get('mode', 'T')  # 모드 값 (T: Tracking, P: Patrol)

        print(f"서버로부터 받은 데이터 : Mode: {mode}, X: {x_mid}, Y: {y_mid}, Move: {move_amount}")

        # 아두이노로 모드 값을 포함하여 데이터 전송 (모드를 맨 앞에 추가)
        command = f"{mode}X{x_mid}Y{y_mid}M{move_amount}\n"
        arduino.write(command.encode())
        print(f"서보 모터 이동 명령 전송: {command}")

        tcp_socket.close()
    except Exception as e:
        print(f"좌표 수신 중 오류 발생: {e}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            continue

        frame_time = time.time()

        frame = cv2.flip(frame, 1)

        send_frame_with_json(frame)
        print(f"프레임 및 시간 전송: {frame_time}")

        receive_and_move_servos()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("클라이언트 종료 중...")

finally:
    cap.release()
    udp_socket.close()
    arduino.close()
