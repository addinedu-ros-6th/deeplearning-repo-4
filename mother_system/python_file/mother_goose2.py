from multiprocessing import Pipe, Process
from g_manager import GManager
import threading
import cv2
from WetFloorDetector import WetFloorDetect
from MissingDetector2 import MissingDetect
from missing_face import Missing_face
import socket
import json
import struct
import numpy as np
import base64
import pickle
import sys
import hashlib
import boto3
import os
from deepface import DeepFace


MAX_DGRAM = 65507  # UDP의 최대 패킷 크기
UDP_PORT = 9999
TCP_PORT = 8888  # 헬스 체크용 TCP 포트
FRAME_WIDTH = 640  # 수신되는 영상의 너비
FRAME_HEIGHT = 480  # 수신되는 영상의 높이
ANIMATION_DURATION = 8000  # 창 크기 조정 애니메이션 지속 시간 (밀리초, 8초로 설정)mother_goose.py
# 사용자 입력을 받을 전역 변수
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

mother_req = 40
stop_event = threading.Event()

class DManager:
    def __init__(self, d_pipe):
        
        self.WetFloorDetect = WetFloorDetect()
        self.MissingDetect = MissingDetect()
        self.MissingFace = Missing_face()

        self.d_pipe = d_pipe
        # UDP 소켓 설정 (클라이언트로부터 영상 수신)
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.bind(('0.0.0.0', UDP_PORT))

        # TCP 소켓 설정 (좌표 및 이동 값 전송)
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_socket.bind(('0.0.0.0', TCP_PORT))
        self.tcp_socket.listen(5)
    
    def process_input(self):
        global mother_req
        
        while True:
            try:
                mother_req =22
                if mother_req == 1 or mother_req == 2:
                    pass
            except ValueError:
                print("올바른 숫자를 입력하세요.")
    
    def receive_json_data(self):
        """
        클라이언트로부터 JSON 데이터 수신 및 복원 함수.
        """
        buffer = b""
        while True:
            chunk, addr = self.udp_socket.recvfrom(MAX_DGRAM)
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
            #print(f"수신된 데이터: {buffer.decode('utf-8', errors='ignore')}")
            return None
    
    def connect_and_modelsel(self):
        while True: 
            print("프레임 수신 대기 중...")

            # JSON 데이터 수신
            self.json_data = self.receive_json_data()
            if self.json_data is None:
                continue

            self.frame_base64 = self.json_data['frame']
            self.frame_data = base64.b64decode(self.frame_base64)
            self.frame = cv2.imdecode(np.frombuffer(self.frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.frame = cv2.resize(self.frame, (640, 480))

            
            if mother_req == 10:
                results = self.WetFloorDetect.inference_WF_Detect(self.frame)
                annotated_frame = results[0].plot()
                resized_frame = cv2.resize(annotated_frame, (640, 480))
                self.frame = cv2.putText(img=resized_frame, text="WetFloor", \
                                    org=(30, 30), \
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                                    fontScale=2, color=(0, 0, 255),\
                                    thickness=2)
            
            #elif mother_req == 222222: #Missing Detector
            #    self.frame = self.MissingDetect.inference_MP_Detect(self.frame)
            #    self.frame = cv2.putText(img=self.frame, text="MISSING", \
            #                        org=(30, 30), \
            #                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
            #                        fontScale=2, color=(0, 0, 255),\
            #                        thickness=2)
            #
            elif mother_req == 22:

                self.frame = self.MissingDetect.inference_MP_Detect2(self.frame)
                self.frame = cv2.putText(img=self.frame, text="MISSING", \
                                    org=(30, 30), \
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                                    fontScale=2, color=(0, 0, 255),\
                                    thickness=2)
            elif mother_req == 23:
                #주영님 코드#
                pass

            elif mother_req == 40:
                self.frame = self.MissingFace.face_similarity(self.frame)

            serialized_frame = pickle.dumps(self.frame)
            print(f"Serialized frame size: {sys.getsizeof(serialized_frame)} bytes")
            hash1 = hashlib.sha256(serialized_frame).hexdigest()
            print(f"Hash before sending: {hash1}")
            self.d_pipe.send(serialized_frame)
            cv2.imshow("camera frame", self.frame)
            if (cv2.waitKey(1) & 0xff == ord('q')):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

    
    def camera_and_modelsel(self):
        self.cap = cv2.VideoCapture(0)
        while True: 
            
            ret, self.frame = self.cap.read()
            self.frame = cv2.resize(self.frame, (640, 480))
            if not ret:
                print("Error : Could not read frame")
                break

            if mother_req == 10:
                results = self.WetFloorDetect.inference_WF_Detect(self.frame)
                annotated_frame = results[0].plot()
                resized_frame = cv2.resize(annotated_frame, (640, 480))
                self.frame = cv2.putText(img=resized_frame, text="WetFloor", \
                                    org=(30, 30), \
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                                    fontScale=2, color=(0, 0, 255),\
                                    thickness=2)
            
            elif mother_req == 22:
                self.frame = self.MissingDetect.inference_MP_Detect2(self.frame)
                self.frame = cv2.putText(img=self.frame, text="MISSING", \
                                    org=(30, 30), \
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                                    fontScale=2, color=(0, 0, 255),\
                                    thickness=2)
                
            cv2.imshow("camera frame", self.frame)
            if (cv2.waitKey(1) & 0xff == ord('q')):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        
        
        self.udp_socket.close()
        
    
   
    def initialize_threads(self):
        global stop_event
        stop_event.set()  # 실행 중인 모든 스레드에 종료 신호 보내기
        stop_event.clear()  # 새로 실행될 스레드를 위해 이벤트 초기화

if __name__ == "__main__":
    g_pipe, d_pipe = Pipe(duplex=True)
    gm = GManager()
    dm = DManager(d_pipe)
    g_process = Process(target=gm.run, args=(g_pipe, ))
    
    g_process.start()
    
    connect_thread = threading.Thread(target=dm.connect_and_modelsel)
    connect_thread.start()

    g_process.join()