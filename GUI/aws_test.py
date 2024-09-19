import cv2
import boto3
import socket
import pickle
import struct

MAX_DGRAM = 65507  # UDP의 최대 패킷 크기

# AWS Rekognition 클라이언트 설정
rekognition_client = boto3.client('rekognition', region_name='ap-northeast-2')

# 참조 이미지 로드 함수
def load_reference_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            reference_image = {"Bytes": img_file.read()}
        return reference_image
    except FileNotFoundError:
        print("Reference image not found.")
        return None

# 라즈베리파이에서 영상 수신 및 처리
def process_video_stream(sock, reference_image):
    buffer = b""
    try:
        # 라즈베리파이에서 영상 청크 수신
        chunk, _ = sock.recvfrom(MAX_DGRAM)
        is_last_chunk = struct.unpack("B", chunk[:1])[0]
        buffer += chunk[1:]

        if is_last_chunk:  # 마지막 청크라면
            # 프레임 복원
            frame = pickle.loads(buffer)
            buffer = b""

            # AWS Rekognition에서 사용할 수 있도록 BGR -> RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer_jpg = cv2.imencode('.jpg', frame_rgb)
            webcam_image = {"Bytes": buffer_jpg.tobytes()}

            try:
                # AWS Rekognition으로 얼굴 비교
                response = rekognition_client.compare_faces(
                    SourceImage=reference_image,
                    TargetImage=webcam_image,
                    SimilarityThreshold=20
                )

                return response, frame  # 얼굴 비교 결과와 처리된 프레임 반환

            except rekognition_client.exceptions.InvalidParameterException as e:
                print(f"InvalidParameterException: {e}")
                return None, frame  # 오류 발생 시에도 프레임 반환

    except Exception as e:
        print(f"Error receiving video: {e}")
        return None, None

    return None, None

# UDP 소켓 설정 함수
def setup_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 9999))  # 라즈베리파이에서 수신
    return sock

# aws_test.py가 직접 실행될 때만 실행
if __name__ == '__main__':
    print("aws_test.py가 직접 실행되었습니다.")
    # 여기서 테스트 코드나 직접 실행 코드를 넣습니다.
    # 예를 들어, 소켓을 설정하고 서버에 연결하는 테스트 코드:
    
    sock = setup_socket()
    reference_image = load_reference_image('path_to_image.jpg')
    process_video_stream(sock, reference_image)
