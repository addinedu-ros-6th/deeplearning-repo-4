from multiprocessing import Pipe, Process
from g_manager import GManager
import threading
import cv2
from WetFloorDetector import WetFloorDetect
from MissingDetector import MissingDetect
from missing_face_2 import Missing_face
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
from DeepSortTracker import DeepSortTrack
import time

"""q
missing_face_2(deepface&yolo로 임베딩 후 aws rekognition 더블체크)로 하셔야합니다.
AWS 95% 이상 나오면, 한 번 캠쳐해서 check_images에 저장(27번 코드 송출)
"""


MAX_DGRAM = 1400  # 패킷 단편화를 방지하기 위해 패킷 크기를 줄임
UDP_PORT = 9999
TCP_PORT = 8888  # 헬스 체크용 TCP 포트
FRAME_WIDTH = 640  # 수신되는 영상의 너비
FRAME_HEIGHT = 480  # 수신되는 영상의 높이
ANIMATION_DURATION = 8000  # 창 크기 조정 애니메이션 지속 시간 (밀리초, 8초로 설정)
# 사용자 입력을 받을 전역 변수
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# mother_req 값에 따른 변경 모드
# 0  : 모든 딥러닝 모델은 동작 하지 않고 데이터 전송에 필요한 loop 만 돌음
# 23 : 모든 딥러닝 모델 동작
mother_req = 0
stop_event = threading.Event()

class DManager:
    def __init__(self, d_pipe):
        self.WetFloorDetect = WetFloorDetect()
        self.MissingDetect = MissingDetect()
        self.MissingFace = Missing_face()
        self.Tracker = DeepSortTrack()

        self.d_pipe = d_pipe

        # UDP 소켓 설정 (클라이언트로부터 영상 수신)
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.bind(('0.0.0.0', UDP_PORT))
        self.udp_socket.settimeout(0.5)  # 타임아웃 설정
    
        # 프레임 버퍼 초기화
        self.frame_buffer = {}
        self.last_frame_id = None

        self.gui_req = None  # GUI 요청 code (기본 상태 None)
        self.top_color = ""
        self.bottom_color = ""
        self.is_identified = False # 영상에서 신고된 아이를 찾았다면 True 아니면 False

        self.location_key_cls_and_color_value = {}
        self.id_key_location_value = {}
        self.motor_track_id = None

    # def process_input(self):
    #     global mother_req
        
    #     while True:
    #         try:
    #             mother_req =22
    #             if mother_req == 1 or mother_req == 2:
    #                 pass
    #         except ValueError:
    #             print("올바른 숫자를 입력하세요.")
    
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
        global mother_req
        while True: 
            # GUI 요청 확인
            # GUI 요청이 발생하는 경우, 아래 코드에 의헤 self.gui_req 값이 변경됩니다.
            # 기본 상태 : None
            # 미아 얼굴 촬영됨 (이미지 새로 저장됨) : 11
            # 부모가 보내준 얼굴을 확인 yes : 28,  No : 29

            print(f"mother_req : {mother_req}, gui_req: {self.gui_req}")
   
            if self.d_pipe.poll():
                self.gui_req, data = self.d_pipe.recv()
                print(f"req_num: {self.gui_req}, data: {data}")
                
                if self.gui_req == 1:
                    print("Find Missing Child is requested!!")
                elif self.gui_req == 11:
                    mother_req = 23
                    self.top_color = data["top_color"]
                    self.bottom_color = data["bottom_color"]
                    print(f"GUI Request : {self.gui_req}, Top: {self.top_color}, Bottom: {self.bottom_color} ")
                elif self.gui_req == 28:
                    mother_req = 28
                    print("Mather Accept!!")
                elif self.gui_req == 29:
                    print("Mother Reject!!")
            else:
                self.gui_req = 0

            if mother_req == 0 and self.gui_req == 0:
                time.sleep(0.01)  # 모든 모델 동작 X 10ms 대기
            else:
                print("프레임 수신 대기 중...")

                # 프레임 수신
                self.frame, bg_num, br_code = self.receive_frame()
                if self.frame is None:
                    continue
                self.location_key_cls_and_color_value = {}
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

                elif mother_req == 11:
                    image_path = './face_images/face_image.jpg'
                    if self.MissingFace.load_reference_image(image_path):
                        print("참조 이미지 임베딩이 성공적으로 생성되었습니다.")
                        self.frame = self.MissingFace.face_similarity(self.frame)

                    else:
                        print("참조 이미지 로딩에 실패했습니다.")


                #
                elif mother_req == 22:
                    self.frame,self.location_key_cls_and_color_value = self.MissingDetect.inference_MP_Detect2(self.frame)
                    self.frame = cv2.putText(img=self.frame, text="MISSING", \
                                        org=(30, 30), \
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                                        fontScale=2, color=(0, 0, 255),\
                                        thickness=2)

                    print(self.location_key_cls_and_color_value)

                elif mother_req == 23:
                    self.frame, self.id_key_location_value = self.Tracker.run_tracking(self.frame)
                    self.frame, self.location_key_cls_and_color_value = self.MissingDetect.inference_MP_Detect2(self.frame)
                    # self.frame = cv2.putText(img=self.frame, text="MISSING", \
                    #                     org=(30, 30), \
                    #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                    #                     fontScale=2, color=(0, 0, 255),\
                    #                     thickness=2)

                    image_path = './face_images/face_image.jpg'
                    if self.MissingFace.load_reference_image(image_path):
                        print("참조 이미지 임베딩이 성공적으로 생성되었습니다.")
                        self.frame, face_center = self.MissingFace.face_similarity(self.frame)
                    #print(self.location_key_cls_and_color_value)
                    #print(self.id_key_location_value)

                    self.information = self.merge_information(self.id_key_location_value, self.location_key_cls_and_color_value)
                    print(self.information)

                    self.motor_track_id = self.find_id(self.information, face_center, self.motor_track_id)
                    print('추적대상 : ', self.motor_track_id)

                elif mother_req == 28:
                    self.frame, self.id_key_location_value = self.Tracker.run_tracking(self.frame)
                    self.frame, self.location_key_cls_and_color_value = self.MissingDetect.inference_MP_Detect2(self.frame)

                    self.information = self.merge_information(self.id_key_location_value, self.location_key_cls_and_color_value)
                    print(self.information)        
                    print('추적대상 : ', self.motor_track_id)

                # g-manager 에게 보낼 데이터 정리
                self.d_pipe.send((mother_req, self.is_identified, self.frame))

    def find_id(self, information_dict, find_center, motor_track_id=None, threshold=50):
        """
        information_dict에서 find_center의 X좌표와 가장 가까운 값을 가진 'center' 키를 찾아 motor_track_id로 설정하여 반환하는 함수.
        threshold 값은 X좌표 차이가 얼마까지 허용되는지를 결정.
        """
        if find_center is not None:
            find_x = find_center[0]  # find_center의 X좌표

        closest_id = None
        closest_distance = float('inf')  # 비교할 최소 거리를 무한대로 설정

        # information_dict를 순회하면서 'center'의 X좌표와 비교
        if find_center is not None:
            for track_id, info in information_dict.items():
                center_x = info['center'][0]  # 각 객체의 'center' X좌표
                
                # X좌표 차이를 계산
                distance = abs(find_x - center_x)

                # 거리가 threshold 이내일 경우에만 motor_track_id로 설정
                if distance <= threshold and distance < closest_distance:
                    closest_distance = distance
                    closest_id = track_id  # 가장 가까운 ID 업데이트

        # closest_id가 찾은 결과가 있으면 motor_track_id로 설정, 없으면 기본 motor_track_id 유지
        motor_track_id = closest_id if closest_id is not None else motor_track_id

        return motor_track_id


    def merge_information(self, deepsort_dict, segmentation_dict, threshold=50):
        # information 딕셔너리 초기화
        information = {}
        if len(deepsort_dict) > 0:
            for deep_id, deep_center in deepsort_dict.items():
                # 기본 구조를 None으로 설정
                information[deep_id] = {
                    1: None,  # 상의 클래스 (1)
                    2: None,  # 하의 클래스 (2)
                    'center': deep_center  # deepsort 박스의 중심좌표
                }

                # deepsort 박스 중심의 X좌표
                deep_x = deep_center[0]

                # segmentation_dict에서 비슷한 X좌표를 가진 바운딩 박스를 찾음
                for seg_center, (cls, color) in segmentation_dict.items():
                    seg_x = seg_center[0]

                    # X좌표 차이가 threshold 이내일 경우 같은 객체로 간주
                    if abs(deep_x - seg_x) <= threshold:
                        # cls 값에 따라 상의 또는 하의 값을 업데이트
                        if cls == 1:  # 상의 클래스
                            information[deep_id][1] = color
                        elif cls == 2:  # 하의 클래스
                            information[deep_id][2] = color

        return information
    
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
        
        
    def initialize_threads(self):
        global stop_event
        stop_event.set()  # 실행 중인 모든 스레드에 종료 신호 보내기
        stop_event.clear()  # 새로 실행될 스레드를 위해 이벤트 초기화

    def run(self):    
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

    dm = DManager(d_pipe)
    dm.run()

    gm = GManager()
    g_process = Process(target=gm.run, args=(g_pipe, ))    
    g_process.start()
    
    g_process.join()