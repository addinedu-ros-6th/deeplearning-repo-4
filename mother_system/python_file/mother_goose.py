from multiprocessing import Pipe, Process
from g_manager import GManager
import threading
import cv2
from WetFloorDetector import WetFloorDetect
from MissingDetector import MissingDetect
from missing_face_2 import Missing_face
import socket
import struct
import numpy as np

MAX_DGRAM = 1400  # 패킷 단편화를 방지하기 위해 패킷 크기를 줄임
UDP_PORT = 9999
TCP_PORT = 8888  # 헬스 체크용 TCP 포트
FRAME_WIDTH = 640  # 수신되는 영상의 너비
FRAME_HEIGHT = 480  # 수신되는 영상의 높이
ANIMATION_DURATION = 8000  # 창 크기 조정 애니메이션 지속 시간 (밀리초, 8초로 설정)
# 사용자 입력을 받을 전역 변수
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

mother_req = 11
stop_event = threading.Event()

class DManager:
    def __init__(self):
        
        self.WetFloorDetect = WetFloorDetect()
        self.MissingDetect = MissingDetect()
        self.MissingFace = Missing_face()

        # UDP 소켓 설정 (클라이언트로부터 영상 수신)
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.bind(('127.0.0.1', UDP_PORT))
        self.udp_socket.settimeout(0.5)  # 타임아웃 설정

        # 프레임 버퍼 초기화
        self.frame_buffer = {}
        self.last_frame_id = None

        self.gui_req = None  # GUI 요청 code (기본 상태 None)
        self.is_identified = False # 영상에서 신고된 아이를 찾았다면 True 아니면 False

    def process_input(self):
        global mother_req
        
        while True:
            try:
                mother_req =22
                if mother_req == 1 or mother_req == 2:
                    pass
            except ValueError:
                print("올바른 숫자를 입력하세요.")
    
    def receive_frame(self):
        """
        프레임 데이터를 수신하고 재조합하는 함수.
        """
        while True:
            try:
                packet, addr = self.udp_socket.recvfrom(MAX_DGRAM)
                if not packet:
                    continue

                # 헤더 파싱: frame_id(1바이트), total_chunks(1바이트), seq_num(1바이트), data_len(2바이트), bg_num(1바이트), br_code(1바이트)
                header_format = '=BBBHBB'  # 패딩 없음
                header_size = struct.calcsize(header_format)  # 7 bytes
                if len(packet) < header_size:
                    print("패킷 크기가 헤더 크기보다 작습니다.")
                    continue

                header = packet[:header_size]
                frame_id, total_chunks, seq_num, data_len, bg_num, br_code = struct.unpack(header_format, header)
                data = packet[header_size:]

                if len(data) != data_len:
                    print("데이터 길이 불일치")
                    continue

                if frame_id not in self.frame_buffer:
                    self.frame_buffer[frame_id] = {'total_chunks': total_chunks, 'chunks': {}, 'bg_num': bg_num, 'br_code': br_code}

                self.frame_buffer[frame_id]['chunks'][seq_num] = data

                # 모든 청크를 수신했는지 확인
                if len(self.frame_buffer[frame_id]['chunks']) == total_chunks:
                    # 프레임 조합
                    chunks = [self.frame_buffer[frame_id]['chunks'][i] for i in range(total_chunks)]
                    frame_data = b''.join(chunks)
                    # 프레임 디코딩
                    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        # bg_num과 br_code를 사용하여 필요한 처리를 할 수 있습니다.
                        print(f"Received frame_id: {frame_id}, bg_num: {bg_num}, br_code: {br_code}")
                        self.last_frame_id = frame_id
                        del self.frame_buffer[frame_id]  # 사용된 버퍼 삭제
                        return frame, bg_num, br_code
                    else:
                        print("프레임 디코딩 실패")
                        del self.frame_buffer[frame_id]  # 오류 발생 시 버퍼 삭제
            except socket.timeout:
                # 타임아웃 발생 시 오래된 버퍼 삭제
                if self.last_frame_id is not None and self.last_frame_id in self.frame_buffer:
                    del self.frame_buffer[self.last_frame_id]
                continue

    def connect_and_modelsel(self):
        while True: 
            # GUI 요청 확인
            # GUI 요청이 발생하는 경우, 아래 코드에 의헤 self.gui_req 값이 변경됩니다.
            # 기본 상태 : None
            # 미아 얼굴 촬영됨 (이미지 새로 저장됨) : 11
            # 부모가 보내준 얼굴을 확인 yes : 28,  No : 29
            if self.d_pipe.poll():
                self.gui_req = self.d_pipe.recv()
                print(f"GUI Request : {self.gui_req}")
            else:
                self.gui_req = None

            print("프레임 수신 대기 중...")

            # 프레임 수신
            self.frame, bg_num, br_code = self.receive_frame()
            if self.frame is None:
                continue

            self.frame = cv2.resize(self.frame, (640, 480))
            
            # bg_num과 br_code를 이용하여 추가적인 로직을 추가할 수 있습니다.
            print(f"Processing frame with bg_num: {bg_num}, br_code: {br_code}")
            
            if mother_req == 10:
                results = self.WetFloorDetect.inference_WF_Detect(self.frame)
                annotated_frame = results[0].plot()
                resized_frame = cv2.resize(annotated_frame, (640, 480))
                self.frame = cv2.putText(img=resized_frame, text="WetFloor", \
                                    org=(30, 30), \
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                                    fontScale=2, color=(0, 0, 255),\
                                    thickness=2)
                
            if mother_req == 11:
                image_path = './face_images/face_image.jpg'
                if self.MissingFace.load_reference_image(image_path):
                    print("참조 이미지 임베딩이 성공적으로 생성되었습니다.")
                    self.frame = self.MissingFace.face_similarity(self.frame)

                    self.frame = cv2.putText(img=self.frame, text="Similarity", \
                    org=(30, 30), \
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                    fontScale=2, color=(0, 0, 255),\
                    thickness=2)
                    # AWS 유사도가 95% 이상이고 캡쳐가 완료되지 않았다면
                    if self.MissingFace.capture_done:
                        print("Image already captured.")
                        self.d_pipe.send(27)
                    else:
                        continue

        
                else:
                    print("참조 이미지 로딩에 실패했습니다.")

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
            
            # g-manager 에게 보낼 데이터 정리
            self.d_pipe.send((mother_req, self.is_identified, self.frame))


    # def camera_and_modelsel(self):
    #     self.cap = cv2.VideoCapture(0)
    #     while True: 
            
    #         ret, self.frame = self.cap.read()
    #         self.frame = cv2.resize(self.frame, (640, 480))
    #         if not ret:
    #             print("Error : Could not read frame")
    #             break

    #         if mother_req == 10:
    #             results = self.WetFloorDetect.inference_WF_Detect(self.frame)
    #             annotated_frame = results[0].plot()
    #             resized_frame = cv2.resize(annotated_frame, (640, 480))
    #             self.frame = cv2.putText(img=resized_frame, text="WetFloor", \
    #                                 org=(30, 30), \
    #                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
    #                                 fontScale=2, color=(0, 0, 255),\
    #                                 thickness=2)
            
    #         elif mother_req == 22:
    #             self.frame = self.MissingDetect.inference_MP_Detect2(self.frame)
    #             self.frame = cv2.putText(img=self.frame, text="MISSING", \
    #                                 org=(30, 30), \
    #                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
    #                                 fontScale=2, color=(0, 0, 255),\
    #                                 thickness=2)

    #         cv2.imshow("camera frame", self.frame)
    #         if (cv2.waitKey(1) & 0xff == ord('q')):
    #             break

    #     self.cap.release()
    #     cv2.destroyAllWindows()
    
   
    def initialize_threads(self):
        global stop_event
        stop_event.set()  # 실행 중인 모든 스레드에 종료 신호 보내기
        stop_event.clear()  # 새로 실행될 스레드를 위해 이벤트 초기화

    def run(self, d_pipe):
        
        self.d_pipe = d_pipe
        
        #self.initialize_threads()
        # 사용자 입력을 처리할 쓰레드 시작
        #input_thread = threading.Thread(target=self.process_input)
        #input_thread.start()
        
        #자기 자신 웹캠
        #camera_thread = threading.Thread(target=MANAGER.camera_and_modelsel)
        #camera_thread.start()
    
        #통신으로 프레임 받고 추론    
        connect_thread = threading.Thread(target=self.connect_and_modelsel)
        connect_thread.start()



if __name__ == "__main__":
    g_pipe, d_pipe = Pipe(duplex=True)
    gm = GManager()
    dm = DManager()

    g_process = Process(target=gm.run, args=(g_pipe, ))
    d_process = Process(target=dm.run, args=(d_pipe, ))
    
    
    #d_process = Process(target=dm.__init__)
    g_process.start()
    d_process.start()

    g_process.join()