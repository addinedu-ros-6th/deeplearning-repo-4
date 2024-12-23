from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QDialogButtonBox, QPushButton, QFrame
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt 
import cv2
import sys
import socket
import pickle
import struct
import os 
import datetime 
import numpy as np
import aws_test  # aws_test.py 파일을 가져옵니다.
print(dir(aws_test))  # aws_test 모듈에 어떤 함수가 있는지 확인


MAX_DGRAM = 65507  # UDP의 최대 패킷 크기

class VideoReceiver(QThread):
    frame_received = pyqtSignal(QImage, str)  # QLabel로 보낼 처리된 프레임

    def __init__(self, udp_ip, udp_port, label_name, reference_image):
        super().__init__()
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.label_name = label_name
        self.reference_image = reference_image  # 참조 이미지를 저장

        # 소켓 생성 및 SO_REUSEADDR 설정
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 포트 재사용 설정
        self.socket.bind((self.udp_ip, self.udp_port))  # 라즈베리파이에서 영상 수신
        self.running = True
        self.buffer = b""  # 데이터를 받을 버퍼

    def run(self):
        while self.running:
            try:
                # aws_test.py의 process_video_stream 함수를 사용하여 라즈베리파이에서 받은 영상을 처리
                response, frame = aws_test.process_video_stream(self.socket, self.reference_image)
                
                if response:  # 얼굴 비교가 성공했을 때만 처리
                    print(f"얼굴 비교 결과: {response}")
                    
                    # OpenCV의 BGR 포맷을 RGB로 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.frame_received.emit(q_img, self.label_name)

            except Exception as e:
                print(f"영상 수신 오류: {e}")

    def stop(self):
        self.running = False  # 스레드 종료
        self.socket.close()
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # main.ui 파일을 불러와서 메인 윈도우로 사용 (GUI 폴더 경로 추가)
        uic.loadUi('GUI/main.ui', self)

        # 메인 화면에 있는 버튼을 찾고 클릭 이벤트 연결
        self.btn_airport_info = self.findChild(QPushButton, 'airport_info')
        self.btn_find_man = self.findChild(QPushButton, 'find_man')

        # 버튼 클릭 이벤트 연결
        if self.btn_airport_info is not None:
            self.btn_airport_info.clicked.connect(self.show_airport_info)

        if self.btn_find_man is not None:
            self.btn_find_man.clicked.connect(self.show_input_face)

        

    # air_port_info 창 열기
    def show_airport_info(self):
        self.airport_info_window = QDialog()
        uic.loadUi('GUI/air_port_info.ui', self.airport_info_window)
        
        map1_label = self.airport_info_window.findChild(QLabel, 'map1')
        map2_label = self.airport_info_window.findChild(QLabel, 'map2')
        
        pixmap1 = QPixmap('./GUI/etc_image/airport_map1.png')
        pixmap2 = QPixmap('./GUI/etc_image/airport_map2.png')
        
        if map1_label is not None :
            map1_label.setPixmap(pixmap1)
            
        if map2_label is not None :
            map2_label.setPixmap(pixmap2)
            
        self.airport_info_window.exec_()

    # input_face 창 열기 및 사람찾기 버튼 클릭 시 실행
    def show_input_face(self):
        self.input_face_dialog = InputFaceDialog()
        self.input_face_dialog.exec_()

        if self.input_face_dialog.isHidden():
            self.show_find_man()

    # find_man 창 열기
    def show_find_man(self):
        print("show_find_man 호출됨")
        self.find_man_window = FindManWindow(self.reference_image)
        self.find_man_window.show()  # show()로 창을 띄움

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()  # 다이얼로그 종료


class InputFaceDialog(QDialog):
    def __init__(self):
        super().__init__()

        # input_face.ui 파일 로드
        uic.loadUi('GUI/input_face.ui', self)

        # QLabel 찾기 (QLabel의 이름은 'input_video')
        self.webcam_label = self.findChild(QLabel, 'input_video')
      
        # OpenCV를 사용하여 웹캠 연결
        self.cap = cv2.VideoCapture(0)

        # 얼굴 인식을 위한 Haar Cascade XML 파일 경로 설정
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

        # QTimer 설정하여 주기적으로 프레임 업데이트
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms마다 업데이트

        self.dialog_button_box = self.findChild(QDialogButtonBox, 'register_button')
        self.dialog_button_box.accepted.connect(self.save_current_frame)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()  # 다이얼로그 종료

    def save_current_frame(self):
        ret, frame = self.cap.read()  # 웹캠에서 현재 프레임 읽기
        if ret:
            face_frame = frame  # 현재 프레임을 저장
            self.save_face_image(face_frame)  # 얼굴 저장 함수 호출 
            self.cap.release()
            print("저장 완료")
        self.show_cloth_pop_dialog()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret == True:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            self.webcam_label.setPixmap(QPixmap.fromImage(qimg))

    def save_face_image(self, frame):
        save_dir = os.path.expanduser('./GUI/face_image')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(save_dir, f'face_{timestamp}.jpg')

        cv2.imwrite(file_path, frame)
        print(f"얼굴 이미지가 {file_path}에 저장되었습니다.")

    def show_cloth_pop_dialog(self):
        self.cloth_pop_dialog = ClothPopDialog()
        self.cloth_pop_dialog.exec_()  # exec_()로 다이얼로그 띄움
        self.close()


class ClothPopDialog(QDialog):
    def __init__(self):
        super().__init__()

        # cloth_pop.ui 파일 로드
        uic.loadUi('GUI/cloth_pop.ui', self)

        # QLabel 찾기 (spectrum_image1, spectrum_image2)
        self.spectrum_image1 = self.findChild(QLabel, 'spectrum_image1')
        self.spectrum_image2 = self.findChild(QLabel, 'spectrum_image2')

        # QFrame 찾기 (color1, color2)
        self.color_frame1 = self.findChild(QFrame, 'color1')
        self.color_frame2 = self.findChild(QFrame, 'color2')

        # QDialogButtonBox 찾기
        self.dialog_button_box = self.findChild(QDialogButtonBox, 'register_button')

        # OK 버튼 누르면 find_man 창 열기
        if self.dialog_button_box is not None:
            self.dialog_button_box.accepted.connect(self.show_find_man)

        # spectrum.png 이미지를 OpenCV로 로드 (픽셀 처리)
        self.cv_image = cv2.imread('./GUI/etc_image/spectrum.png')
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환

        # QLabel에 이미지 설정
        height, width, channel = self.cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.spectrum_image1.setPixmap(pixmap)
        self.spectrum_image2.setPixmap(pixmap)

        # spectrum_image1, spectrum_image2 클릭 이벤트 연결
        self.spectrum_image1.mousePressEvent = self.get_color_from_image1
        self.spectrum_image2.mousePressEvent = self.get_color_from_image2

    def get_color_from_image1(self, event):
        x = event.pos().x()
        y = event.pos().y()
        color = self.find_nearest_color(x, y)
        
        if color is not None:
            color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            self.color_frame1.setStyleSheet(f"background-color: {color_hex};")
            print(f"Color1: {color_hex}")

    def get_color_from_image2(self, event):
        x = event.pos().x()
        y = event.pos().y()
        color = self.find_nearest_color(x, y)
        
        if color is not None:
            color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            self.color_frame2.setStyleSheet(f"background-color: {color_hex};")
            print(f"Color2: {color_hex}")

    def find_nearest_color(self, x, y, search_radius=5):
        """
        클릭한 좌표 (x, y)를 중심으로 주어진 반경 내에서 가장 가까운 흰색이 아닌 색을 찾는 함수
        """
        height, width, _ = self.cv_image.shape
        
        # 흰색으로부터 얼마나 벗어나야 흰색이 아닌 것으로 간주할지 설정
        tolerance = 10
        
        # 탐색 범위를 설정
        for radius in range(1, search_radius+1):
            for i in range(max(0, x-radius), min(width, x+radius+1)):
                for j in range(max(0, y-radius), min(height, y+radius+1)):
                    color = self.cv_image[j, i]  # (y, x) 좌표에서 RGB 값을 가져옴
                    if not all([abs(c - 255) < tolerance for c in color]):
                        # 흰색이 아닌 색을 찾으면 반환
                        return color
        
        # 만약 흰색이 아닌 색을 찾지 못했다면, 흰색 반환
        return np.array([255, 255, 255])


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()  # 다이얼로그 종료



class FindManWindow(QMainWindow):
    def __init__(self, reference_image):
        super().__init__()
        # 참조 이미지 로드
        self.reference_image = aws_test.load_reference_image('./path_to_reference_image.jpg')
        # find_man.ui 파일 로드
        uic.loadUi('GUI/find_man.ui', self)

        # QLabel 찾기 (QLabel의 이름은 'goose_video1', 'goose_video2', 'goose_video3')
        self.goose_video1 = self.findChild(QLabel, 'goose_video1')
        self.goose_video2 = self.findChild(QLabel, 'goose_video2')
        self.goose_video3 = self.findChild(QLabel, 'goose_video3')

        # 라즈베리파이 IP 및 포트 설정 (예시)
        raspberry_pi_ips = ['192.168.0.32', '192.168.0.102', '192.168.45.90']
        udp_ports = [9999, 9999, 9999]

        # 3개의 라즈베리파이로부터 영상 수신
        self.video_threads = [
            VideoReceiver(raspberry_pi_ips[0], udp_ports[0], 'goose_video1', reference_image),
            VideoReceiver(raspberry_pi_ips[1], udp_ports[1], 'goose_video2', reference_image),
            VideoReceiver(raspberry_pi_ips[2], udp_ports[2], 'goose_video3', reference_image)
        ]

        # 각 스레드의 frame_received 신호를 QLabel 업데이트 함수에 연결
        self.video_threads[0].frame_received.connect(self.update_frame)
        self.video_threads[1].frame_received.connect(self.update_frame)
        self.video_threads[2].frame_received.connect(self.update_frame)

        # 스레드 시작
        for thread in self.video_threads:
            thread.start()

    # 프레임을 QLabel에 업데이트하는 함수
    def update_frame(self, q_img, label_name):
        if label_name == 'goose_video1':
            self.goose_video1.setPixmap(QPixmap.fromImage(q_img))
        elif label_name == 'goose_video2':
            self.goose_video2.setPixmap(QPixmap.fromImage(q_img))
        elif label_name == 'goose_video3':
            self.goose_video3.setPixmap(QPixmap.fromImage(q_img))

    # 창 닫을 때 스레드 종료
    def closeEvent(self, event):
        for thread in self.video_threads:
            thread.stop()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            QApplication.quit()  # 프로그램 종료
        else:
            super().keyPressEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
