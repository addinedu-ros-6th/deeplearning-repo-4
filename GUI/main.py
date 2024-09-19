from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QDialogButtonBox, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import cv2
import sys
import socket
import pickle
import struct
import os 
import datetime 

MAX_DGRAM = 65507  # UDP의 최대 패킷 크기

class VideoReceiver(QThread):
    frame_received = pyqtSignal(QImage, str)  # 프레임이 수신될 때 신호를 발생시킴

    def __init__(self, udp_ip, udp_port, label_name):
        super().__init__()
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.label_name = label_name
        
        # 소켓 생성 및 SO_REUSEADDR 설정
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 포트 재사용 설정
        self.socket.bind(('0.0.0.0', self.udp_port))  # 해당 IP와 포트에 바인딩
        self.running = True
        self.buffer = b""  # 데이터를 받을 버퍼

    def run(self):
        while self.running:
            try:
                chunk, _ = self.socket.recvfrom(MAX_DGRAM)  # 데이터 수신
                is_last_chunk = struct.unpack("B", chunk[:1])[0]  # 첫 번째 바이트로 마지막 청크 여부 확인
                self.buffer += chunk[1:]  # 첫 번째 바이트 제외하고 버퍼에 추가

                if is_last_chunk:  # 마지막 청크라면
                    frame = pickle.loads(self.buffer)  # 버퍼에서 프레임 복원
                    self.buffer = b""  # 버퍼 초기화

                    # OpenCV의 BGR 포맷을 RGB로 변환
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # QImage로 변환
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

                    # QLabel 업데이트
                    self.frame_received.emit(q_img, self.label_name)  # 신호 발생, QLabel에 전달

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
        self.airport_info_window = QMainWindow()
        uic.loadUi('GUI/air_port_info.ui', self.airport_info_window)
        self.airport_info_window.show()

    # input_face 창 열기 및 사람찾기 버튼 클릭 시 실행
    def show_input_face(self):
        self.input_face_dialog = InputFaceDialog()
        self.input_face_dialog.exec_()

        if self.input_face_dialog.isHidden():
            self.show_find_man()

    # find_man 창 열기
    def show_find_man(self):
        print("show_find_man 호출됨")
        self.find_man_window = FindManWindow()
        self.find_man_window.show()  # show()로 창을 띄움


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
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # QTimer 설정하여 주기적으로 프레임 업데이트
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms마다 업데이트
        
    def accept(self):
        self.clean_up()
        super().accept()

    def reject(self):
        self.clean_up()
        super().reject()

    def clean_up(self):
        # 스레드 및 소켓 해제, 타이머 중지
        self.cap.release()  # 웹캠 해제
        self.timer.stop()

    # 창 닫을 때 웹캠 해제
    def closeEvent(self, event):
        self.clean_up()  # 자원 해제
        event.accept()

    # 웹캠 프레임을 QLabel에 업데이트하고 얼굴 인식 후 저장
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 얼굴 인식
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # 얼굴이 인식되면 사각형 그리기 및 얼굴 저장
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face_frame = frame[y:y+h, x:x+w]  # 얼굴 영역만 잘라서 저장
                    self.save_face_image(face_frame)  # 얼굴 저장 함수 호출

                # 얼굴이 인식되고 저장되면 OK 버튼 누른 것처럼 cloth_pop 창 열기
                self.timer.stop()  # 타이머 중지 (더 이상 얼굴 탐지하지 않도록)
                self.cap.release()  # 카메라 해제
                self.show_cloth_pop_dialog()

            # OpenCV 프레임을 QImage로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # QLabel에 QPixmap으로 변환한 이미지 표시
            self.webcam_label.setPixmap(QPixmap.fromImage(qimg))

    # 얼굴 이미지 저장
    def save_face_image(self, frame):
        save_dir = os.path.expanduser('./GUI')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 타임스탬프를 이용해 파일명 생성
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(save_dir, f'face_{timestamp}.jpg')

        # 이미지 저장
        cv2.imwrite(file_path, frame)
        print(f"얼굴 이미지가 {file_path}에 저장되었습니다.")

    # cloth_pop 다이얼로그 열기
    def show_cloth_pop_dialog(self):
        self.cloth_pop_dialog = ClothPopDialog()
        self.cloth_pop_dialog.exec_()  # exec_()로 다이얼로그 띄움
        self.close()

    # 창 닫을 때 웹캠 해제
    def closeEvent(self, event):
        self.cap.release()  # 웹캠 해제
        self.timer.stop()
        event.accept()


class ClothPopDialog(QDialog):
    def __init__(self):
        super().__init__()

        # cloth_pop.ui 파일 로드
        uic.loadUi('GUI/cloth_pop.ui', self)

        # QDialogButtonBox 찾기
        self.dialog_button_box = self.findChild(QDialogButtonBox, 'register_button')

        # OK 버튼 누르면 find_man 창 열기
        if self.dialog_button_box is not None:
            self.dialog_button_box.accepted.connect(self.show_find_man)


class FindManWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # find_man.ui 파일 로드
        uic.loadUi('GUI/find_man.ui', self)

        # QLabel 찾기 (QLabel의 이름은 'goose_video1', 'goose_video2', 'goose_video3')
        self.goose_video1 = self.findChild(QLabel, 'goose_video1')
        self.goose_video2 = self.findChild(QLabel, 'goose_video2')
        self.goose_video3 = self.findChild(QLabel, 'goose_video3')

        # 라즈베리파이 IP 및 포트 설정 (예시)
        raspberry_pi_ips = ['192.168.45.90', '192.168.0.102', '192.168.45.90']  # 라즈베리파이 IP
        udp_ports = [9999, 9999, 9999]  # 라즈베리파이에서 송출하는 포트

        # 3개의 라즈베리파이로부터 영상 수신
        self.video_threads = [
            VideoReceiver(raspberry_pi_ips[0], udp_ports[0], 'goose_video1'),
            VideoReceiver(raspberry_pi_ips[1], udp_ports[1], 'goose_video2'),
            VideoReceiver(raspberry_pi_ips[2], udp_ports[2], 'goose_video3')
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())