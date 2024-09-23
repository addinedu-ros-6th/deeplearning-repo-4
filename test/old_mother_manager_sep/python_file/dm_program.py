import cv2
import socket
import threading
import json
import base64 
import numpy as np

from queue import Queue

from WetFloorDetector import WetFloorDetect
from MissingDetector import MissingDetect
from custom_deque import *

SERVER_IP = "192.168.1.16"
BABY_SERVER_PORT = 8888  # BABY goose 접속 TCP 포트
GUI_SERVER_PORT = 8889  # GUI 접속 TCP 포트
FRAME_WIDTH = 640  # 수신되는 영상의 너비
FRAME_HEIGHT = 480  # 수신되는 영상의 높이

mother_req = 22 #  test 용
stop_event = threading.Event()

class BabyConnector:
    def __init__(self):
        self.server_socket =socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((SERVER_IP, BABY_SERVER_PORT))
        self.server_socket.listen(2)
        print(f"Baby 서버가 {BABY_SERVER_PORT} 포트에서 대기 중...")
        
    def accept_connections(self, baby_send_queue, gui_req_queue):
        # Baby Goose 연결을 비동기로 수랑
        self.baby_conn, self.baby_addr = self.server_socket.accept()
        print(f"Baby Goose 연결됨. IP: {self.baby_addr}")

        self.handle_baby(baby_send_queue, gui_req_queue)        

    def handle_baby(self, baby_send_queue, gui_req_queue):
        """Baby Goose 로 부터 영상을 포함한 JSON 을 받아옴"""
        buffer = b""  # 데이터를 누적할 버퍼
        try:
            while True:
                chunk = self.baby_conn.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                
                try:
                    # JSON 데이터를 버퍼에서 복원
                    while True:
                        # JSON 문자열의 끝을 찾기 위해 버퍼에서 추출
                        json_data_str = buffer.decode('utf-8')
                        json_obj, idx = json.JSONDecoder().raw_decode(json_data_str)
                        
                        gui_req_queue.put(json.dumps(json_obj.copy()).encode('utf-8'))
                
                        # base64 로 인코딩된 프레임 복원
                        frame_base64 = json_obj["frame"]
                        frame_jpg = base64.b64decode(frame_base64)

                        # JPEG 이미지를  numpy 배열로 변환
                        frame_np = np.frombuffer(frame_jpg, dtype=np.uint8)
                        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

                        # 우선 frame 만 던짐 
                        baby_send_queue.put(frame)
                        
                        # 처리된 JSON 문자열을 버퍼에서 제거
                        buffer = buffer[idx:]

                except json.JSONDecodeError:
                    # 아직 완전한 JSON이 수신되지 않음, 계속 수신 대기
                    continue

        except Exception as e:
            print(f"Baby Goose 로부터 온 data 처리 중 오류 발생: {e}")
        

    def initialize_threads(self):
        global stop_event
        stop_event.set()  # 실행 중인 모든 스레드에 종료 신호 보내기
        stop_event.clear()  # 새로 실행될 스레드를 위해 이벤트 초기화

class GUIConnector:
    def __init__(self):
        self.server_socket =socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((SERVER_IP, GUI_SERVER_PORT))
        self.server_socket.listen(1)
        print(f"GUI 서버가 {GUI_SERVER_PORT} 포트에서 대기 중...")
        
    def accecpt_connections(self, baby_send_queue, gui_req_queue):
        # GUI 연결을 비동기로 수락
        self.gui_conn, self.gui_addr = self.server_socket.accept()
        print(f"GUI 연결됨. IP: {self.gui_addr}")

        self.handle_gui(baby_send_queue, gui_req_queue)

    def handle_gui(self, baby_send_queue, gui_req_queue):
        """GUI 로 영상을 포함한 JSON 전송"""
        try:
            while True:
                if gui_req_queue.size() > 0:
                    data = gui_req_queue.get()
                    print(f"gui_req_queue.size :{gui_req_queue.size()}")
                    self.gui_conn.send(data)

        except Exception as e:
            print(f"GUI 로 data 전송하기 위한 데이터 처리 중 오류 발생: {e}")



    def initialize_threads(self):
        global stop_event
        stop_event.set()  # 실행 중인 모든 스레드에 종료 신호 보내기
        stop_event.clear()  # 새로 실행될 스레드를 위해 이벤트 초기화


class DManager:
    def __init__(self):        
        self.WetFloorDetect = WetFloorDetect()
        self.MissingDetect = MissingDetect()
    
    def model_select(self, baby_send_queue):
        while True:
            if baby_send_queue.size() > 0:
                frame = baby_send_queue.get() 
                resized_frame = cv2.resize(frame, (640, 480))

                if mother_req == 10:
                    results = self.WetFloorDetect.inference_WF_Detect(frame)
                    annotated_frame = results[0].plot()
                    frame = cv2.putText(img=resized_frame, text="WetFloor", \
                                        org=(30, 30), \
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                                        fontScale=2, color=(0, 0, 255),\
                                        thickness=2)

                #elif mother_req == 222222: #Missing Detector
                #    frame = self.MissingDetect.inference_MP_Detect(frame)
                #    frame = cv2.putText(img=frame, text="MISSING", \
                #                        org=(30, 30), \
                #                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                #                        fontScale=2, color=(0, 0, 255),\
                #                        thickness=2)
                #
                elif mother_req == 22:
                    frame = self.MissingDetect.inference_MP_Detect2(frame)
                    frame = cv2.putText(img=frame, text="MISSING", \
                                        org=(30, 30), \
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                                        fontScale=2, color=(0, 0, 255),\
                                        thickness=2)       

    def initialize_threads(self):
        global stop_event
        stop_event.set()  # 실행 중인 모든 스레드에 종료 신호 보내기
        stop_event.clear()  # 새로 실행될 스레드를 위해 이벤트 초기화

if __name__ == "__main__":
        baby_send_queue = CircularQueue(max_size=10)   # baby -> DM
        baby_req_queue = CircularQueue(max_size=10)    # DM -> baby

        gui_send_queue = CircularQueue(max_size=10)    # gui -> DM
        gui_req_queue = CircularQueue(max_size=10)     # DM -> gui
            
        dm = DManager()

        #통신으로 프레임 받고 추론    
        model_thread = threading.Thread(target=dm.model_select, args=(baby_send_queue, ))
        model_thread.start()

        gui_connector = GUIConnector()  
        threading.Thread(target=gui_connector.accecpt_connections, args=(baby_send_queue, gui_req_queue, )).start()
  
        baby_connector = BabyConnector()
        threading.Thread(target=baby_connector.accept_connections, args=(baby_send_queue, gui_req_queue, )).start()

