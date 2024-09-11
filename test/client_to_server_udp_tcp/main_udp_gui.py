import sys
import socket
import pickle
import struct
import cv2
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QGridLayout
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, QPropertyAnimation, QRect, QEasingCurve
import torch
from ultralytics import YOLO
import numpy as np

MAX_DGRAM = 65507  # UDP의 최대 패킷 크기
TCP_PORT = 8888  # 헬스 체크용 TCP 포트
FRAME_WIDTH = 640  # 수신되는 영상의 너비
FRAME_HEIGHT = 480  # 수신되는 영상의 높이
ANIMATION_DURATION = 8000  # 창 크기 조정 애니메이션 지속 시간 (밀리초, 8초로 설정)

# YOLOv8 모델 불러오기
model = YOLO('/home/sehyung/dev_ws/machine_learning/tcp_video_project/best_clothes_seg.pt')  # Segmentation 모델 경로

# 색상 범위 정의
color_ranges = {
    'red1': [(0, 100, 100), (10, 255, 255)],
    'red2': [(170, 100, 100), (180, 255, 255)],
    'orange': [(10, 100, 100), (25, 255, 255)],
    'yellow': [(25, 100, 100), (35, 255, 255)],
    'green': [(35, 100, 100), (85, 255, 255)],
    'blue': [(85, 100, 100), (125, 255, 255)],
    'navy': [(125, 100, 100), (140, 255, 255)],
    'violet': [(140, 100, 100), (170, 255, 255)],
    'white': [(0, 0, 125), (180, 30, 255)],
    'gray': [(0, 0, 70), (180, 30, 125)],
    'black': [(0, 0, 1), (180, 50, 70)]
}

class VideoReceiver(QThread):
    frame_received = Signal(QImage, str)

    def __init__(self, server_socket, main_window):
        super().__init__()
        self.server_socket = server_socket
        self.main_window = main_window  # MainWindow 객체 참조
        self.running = True
        self.buffer = b""

    def run(self):
        try:
            while self.running:
                # 데이터 수신
                chunk, addr = self.server_socket.recvfrom(MAX_DGRAM)
                client_address = addr[0]  # 클라이언트 주소로 구분
                is_last_chunk = struct.unpack("B", chunk[:1])[0]  # 첫 바이트는 마지막 청크 여부
                self.buffer += chunk[1:]  # 실제 데이터는 두 번째 바이트부터

                if is_last_chunk:
                    # 마지막 청크를 수신했을 때 프레임 복원
                    try:
                        frame = pickle.loads(self.buffer)
                        self.buffer = b""  # 버퍼 초기화

                        # 프레임을 RGB로 변환 후 PyQt로 변환
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                        # YOLOv8 모델로 추론
                        results = model.predict(source=frame, iou=0.25, conf=0.7)
                        predicted_classes = [model.names[int(class_id)] for class_id in results[0].boxes.cls]

                        selected_color = self.main_window.color_selector.currentText()

                        for idx, result in enumerate(results):
                            if result.masks is not None:
                                for mask in result.masks.data:
                                    mask = mask.cpu().numpy()
                                    colored_mask = (mask * 255).astype('uint8')
                                    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_GRAY2BGR)
                                    mask255 = (mask * 255).astype("uint8")
                                    masked_frame = cv2.bitwise_and(frame, frame, mask=mask255)

                                    hsv_masked_area = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

                                    color_areas = {}
                                    for color_name, (lower, upper) in color_ranges.items():
                                        lower_bound = np.array(lower, dtype=np.uint8)
                                        upper_bound = np.array(upper, dtype=np.uint8)
                                        color_mask = cv2.inRange(hsv_masked_area, lower_bound, upper_bound)
                                        area_size = cv2.countNonZero(color_mask)
                                        color_areas[color_name] = area_size

                                    largest_color = max(color_areas, key=color_areas.get)
                                    largest_color_value = color_areas[largest_color]
                                    next_largest_color_value = max(color_areas.values())

                                    if largest_color_value < next_largest_color_value * 3.5:
                                        show_color = f"{largest_color} and {max(color_areas, key=color_areas.get)}"
                                    else:
                                        show_color = largest_color

                                    frame = cv2.addWeighted(frame, 1, colored_mask, 0, 0)
                                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    if contours:
                                        cnt = max(contours, key=cv2.contourArea)
                                        M = cv2.moments(cnt)
                                        if M["m00"] != 0:
                                            cX = int(M["m10"] / M["m00"])
                                            cY = int(M["m01"] / M["m00"])

                                            class_name = predicted_classes[idx]

                                            if selected_color in show_color:
                                                color_text = f"color : {show_color}, class: {class_name}, detected"
                                                contour_color = (255, 0, 0)
                                                cv2.circle(frame, (cX, cY), 10, (255, 0, 0), 3)  # 반지름 10, 두께 3으로 설정
                                                print(f"빨간 점 좌표: ({cX}, {cY})")
                                            else:
                                                color_text = f"color : {show_color}, class: {class_name}"
                                                contour_color = (0, 255, 0)

                                            cv2.putText(frame, color_text, (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, contour_color, 2)
                                            cv2.drawContours(frame, [cnt], -1, contour_color, 2)

                        h, w, ch = frame.shape
                        q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)

                        self.frame_received.emit(q_img, client_address)

                    except Exception as e:
                        print(f"프레임 복원 중 오류 발생: {e}")
                        self.buffer = b""
        except Exception as e:
            print(f"데이터 수신 오류: {e}")
        finally:
            self.server_socket.close()

    def stop(self):
        self.running = False
        self.wait()

class TCPHealthCheckServer(QThread):
    def __init__(self, tcp_socket):
        super().__init__()
        self.tcp_socket = tcp_socket
        self.running = True

    def run(self):
        while self.running:
            client_conn, client_address = self.tcp_socket.accept()
            print(f"TCP 헬스 체크 요청 수신됨: {client_address}")
            try:
                client_conn.sendall(b"alive")
            except Exception as e:
                print(f"TCP 응답 오류: {e}")
            finally:
                client_conn.close()

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Client UDP Video Receiver")

        self.layout = QGridLayout()
        self.labels = {}
        self.current_row = 0
        self.current_col = 0
        self.client_count = 0

        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: white;
            }
            QLabel {
                background-color: #2e2e2e;
                border: 2px solid #ffffff;
                border-radius: 5px;
                padding: 10px;
            }
            QComboBox {
                background-color: #555;
                color: white;
                font-size: 14px;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #333;
                color: white;
                selection-background-color: #555;
                selection-color: #ffffff;
            }
        """)

        # 드롭다운 메뉴 (색상 선택)
        self.color_selector = QComboBox()
        self.color_selector.addItems(color_ranges.keys())
        self.color_selector.setFixedWidth(200)
        self.color_selector.setFixedHeight(40)

        central_widget = QWidget(self)
        central_layout = QVBoxLayout()
        central_layout.addWidget(self.color_selector, alignment=Qt.AlignHCenter)
        central_layout.addLayout(self.layout)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        # 첫 번째 영상이 들어오기 전에 '영상 수신 대기' 화면을 미리 세팅
        self.init_empty_screen()

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind(('0.0.0.0', 9999))

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.bind(('0.0.0.0', TCP_PORT))
        self.tcp_socket.listen(5)

        self.receiver_thread = VideoReceiver(self.server_socket, self)
        self.receiver_thread.frame_received.connect(self.update_frame)
        self.receiver_thread.start()

        self.tcp_health_thread = TCPHealthCheckServer(self.tcp_socket)
        self.tcp_health_thread.start()

    def init_empty_screen(self):
        # 초기 빈 화면에 '영상 수신 대기' 메시지 표시
        label = QLabel(self)
        label.setAlignment(Qt.AlignCenter)
        label.setText("영상 수신 대기 중...")
        label.setStyleSheet("background-color: #1c1c1c; color: white; font-size: 16px;")
        label.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)
        self.layout.addWidget(label, self.current_row, self.current_col)
        self.labels["waiting"] = label

    def add_new_client_label(self, client_address):
        if "waiting" in self.labels:
            # '영상 수신 대기 중' 메시지를 삭제
            self.layout.removeWidget(self.labels["waiting"])
            self.labels["waiting"].deleteLater()
            del self.labels["waiting"]

        label = QLabel(self)
        label.setAlignment(Qt.AlignCenter)
        label.setText(f"Waiting for {client_address} video stream...")
        label.setStyleSheet("background-color: #1c1c1c; color: white;")
        label.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)
        self.layout.addWidget(label, self.current_row, self.current_col)
        self.labels[client_address] = label

        self.current_col += 1
        if self.current_col >= 2:
            self.current_col = 0
            self.current_row += 1

        self.client_count += 1
        if self.client_count > 1:
            self.animate_window_resize()

    def animate_window_resize(self):
        new_width = max(FRAME_WIDTH * self.client_count, self.width())
        new_height = FRAME_HEIGHT
        animation = QPropertyAnimation(self, b"geometry")
        animation.setDuration(ANIMATION_DURATION)
        animation.setStartValue(self.geometry())
        animation.setEndValue(QRect(self.x(), self.y(), new_width, new_height))
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        animation.start()
        while animation.state() == QPropertyAnimation.Running:
            QApplication.processEvents()

    def update_frame(self, q_img, client_address):
        if client_address not in self.labels:
            self.add_new_client_label(client_address)
        self.labels[client_address].setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.receiver_thread.stop()
        self.tcp_health_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
