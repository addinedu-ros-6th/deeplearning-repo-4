import socket
import cv2
import pickle
import struct
import time

# 서버 IP 및 포트 설정
SERVER_IP = '192.168.45.166 '  # 서버 IP를 실제 IP로 변경
UDP_PORT = 9999
TCP_PORT = 8888  # 헬스 체크용 TCP 포트

# UDP 소켓 생성
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 웹캠 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

MAX_DGRAM = 65507 - 1  # UDP 데이터그램의 최대 크기에서 1바이트는 청크 구분 용도로 사용
HEALTH_CHECK_INTERVAL = 5  # 헬스 체크 주기를 10초로 늘림
TCP_TIMEOUT = 3  # TCP 타임아웃을 5초로 설정

# 상태 플래그
connected_to_server = False  # 처음에는 연결 상태를 확인하지 않은 상태

def serialize_frame(frame):
    """프레임을 직렬화하여 데이터로 변환"""
    data = pickle.dumps(frame)
    return data

def send_frame_in_chunks(frame_data):
    """프레임 데이터를 청크로 나누어 전송"""
    size = len(frame_data)
    num_chunks = (size // MAX_DGRAM) + 1

    for i in range(num_chunks):
        start = i * MAX_DGRAM
        end = min(start + MAX_DGRAM, size)
        chunk = frame_data[start:end]

        # 마지막 청크인지 표시
        if i == num_chunks - 1:
            client_socket.sendto(struct.pack("B", 1) + chunk, (SERVER_IP, UDP_PORT))  # 마지막 청크
        else:
            client_socket.sendto(struct.pack("B", 0) + chunk, (SERVER_IP, UDP_PORT))  # 중간 청크

def check_server_status():
    """TCP로 서버 상태 확인"""
    global connected_to_server  # 연결 상태를 저장하는 플래그
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(TCP_TIMEOUT)  # 서버 응답 타임아웃을 5초로 설정
        tcp_socket.connect((SERVER_IP, TCP_PORT))
        tcp_socket.sendall(b"heartbeat")
        response = tcp_socket.recv(1024)
        tcp_socket.close()

        # 서버에서 올바른 응답이 오면 연결 성공
        if response == b"alive":
            if not connected_to_server:  # 이전에 연결되지 않았던 경우
                print("서버 연결 성공!")
                connected_to_server = True  # 연결 상태를 기록
            return True
        else:
            return False
    except (socket.timeout, ConnectionRefusedError):
        if connected_to_server:  # 이전에 연결됐던 경우
            print("서버 응답 없음. 송출을 중지합니다.")
            connected_to_server = False  # 연결 실패 상태로 변경
        return False

print("UDP 영상 송출 시작")
last_health_check = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            continue

        # 프레임 직렬화
        frame_data = serialize_frame(frame)

        try:
            # 청크로 나누어 전송
            send_frame_in_chunks(frame_data)
        except Exception as e:
            print(f"데이터 송출 오류: {e}")
            continue

        # 주기적으로 서버 상태 확인 (헬스 체크 실패 시 프로그램 종료)
        if time.time() - last_health_check > HEALTH_CHECK_INTERVAL:
            if not check_server_status():
                print("서버 응답 없음, 송출을 중지합니다.")
                break  # 헬스 체크 실패 시 송출 중지 및 프로그램 종료
            last_health_check = time.time()

        # 영상 송출 속도 조절
        time.sleep(0.03)

except KeyboardInterrupt:
    print("클라이언트 종료 중...")

finally:
    # 리소스 해제
    cap.release()
    client_socket.close()
