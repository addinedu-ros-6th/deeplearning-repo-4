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
SERVER_IP = '127.0.0.1'
UDP_PORT = 9999
TCP_PORT = 8888
MAX_DGRAM = 1400  # 패킷 단편화를 방지하기 위해 패킷 크기를 줄임

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

frame_id = 0  # 프레임 식별자

def send_frame(frame):
    """
    프레임을 바이너리 데이터로 전송하는 함수.
    """
    global frame_id
    frame_id = (frame_id + 1) % 256  # 0-255 사이에서 순환하도록

    # JPEG로 인코딩 (압축률 조정 가능)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # 압축률을 높여 이미지 크기 감소
    result, frame_jpg = cv2.imencode('.jpg', frame, encode_param)
    data = frame_jpg.tobytes()

    # 데이터 분할
    chunk_size = MAX_DGRAM - 7  # 헤더 크기 고려 (frame_id(1) + total_chunks(1) + seq_num(1) + data_len(2) + bg_num(1) + br_code(1)) = 7 bytes
    size = len(data)
    total_chunks = (size + chunk_size - 1) // chunk_size

    header_format = '=BBBHBB'  # 패딩 없음 (frame_id, total_chunks, seq_num, data_len, bg_num, br_code)
    for seq_num in range(total_chunks):
        start = seq_num * chunk_size
        end = min(start + chunk_size, size)
        chunk = data[start:end]

        # 헤더 구성: frame_id(1바이트), total_chunks(1바이트), seq_num(1바이트), data_len(2바이트), bg_num(1바이트), br_code(1바이트)
        header = struct.pack(header_format, frame_id, total_chunks, seq_num, len(chunk), bg_num, br_code)
        udp_socket.sendto(header + chunk, (SERVER_IP, UDP_PORT))

def receive_and_move_servos():
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((SERVER_IP, TCP_PORT))
        coord_data = tcp_socket.recv(1024).decode('utf-8').strip()

        if coord_data and 'X' in coord_data and 'Y' in coord_data and 'M' in coord_data:
            x_mid = int(coord_data[coord_data.find('X')+1:coord_data.find('Y')])
            y_mid = int(coord_data[coord_data.find('Y')+1:coord_data.find('M')])
            move_amount = int(coord_data[coord_data.find('M')+1:])
            print(f"서버로부터 받은 좌표: X: {x_mid}, Y: {y_mid}, Move: {move_amount}")

            command = f"X{x_mid}Y{y_mid}M{move_amount}\n"
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

        frame = cv2.flip(frame, 1)

        send_frame(frame)
        print(f"프레임 {frame_id} 전송 완료")

        # receive_and_move_servos()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("클라이언트 종료 중...")

finally:
    cap.release()
    udp_socket.close()
    arduino.close()
