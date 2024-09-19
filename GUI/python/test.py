from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QDialogButtonBox, QFrame, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import QTimer
import cv2
import sys

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
            self.btn_find_man.clicked.connect(self.show_find_man)

    # air_port_info 창 열기
    def show_airport_info(self):
        self.airport_info_window = QMainWindow()
        uic.loadUi('GUI/air_port_info.ui', self.airport_info_window)
        self.airport_info_window.show()

    # find_man 창 열기 및 웹캠 실행
    def show_find_man(self):
        self.find_man_window = FindManWindow()  # QMainWindow 사용
        self.find_man_window.show()

class FindManWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # find_man.ui 파일 로드
        uic.loadUi('GUI/find_man.ui', self)

        # QLabel 찾기 (QLabel의 이름은 'goose_video1')
        self.webcam_label = self.findChild(QLabel, 'goose_video1')

        # OpenCV를 사용하여 웹캠 연결
        self.cap = cv2.VideoCapture(0)

        # QTimer 설정하여 주기적으로 프레임 업데이트
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms마다 업데이트

        # 창이 열리면 바로 input_face 다이얼로그 띄우기
        self.show_input_face_dialog()

    # 웹캠 프레임을 QLabel에 업데이트
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # OpenCV에서 BGR로 가져온 이미지를 RGB로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # OpenCV 프레임을 QImage로 변환
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # QLabel에 QPixmap으로 변환한 이미지 표시
            self.webcam_label.setPixmap(QPixmap.fromImage(qimg))

    # 창 닫을 때 웹캠 해제
    def closeEvent(self, event):
        self.cap.release()  # 웹캠 해제
        self.timer.stop()
        event.accept()

    # input_face 다이얼로그 열기
    def show_input_face_dialog(self):
        self.input_face_dialog = InputFaceDialog()
        self.input_face_dialog.exec_()

class InputFaceDialog(QDialog):
    def __init__(self):
        super().__init__()

        # input_face.ui 파일 로드
        uic.loadUi('GUI/input_face.ui', self)

        # QDialogButtonBox 찾기
        self.dialog_button_box = self.findChild(QDialogButtonBox, 'register_button')  # register_button은 QDialogButtonBox

        # OK 버튼 누르면 cloth_pop 다이얼로그 열기
        if self.dialog_button_box is not None:
            self.dialog_button_box.accepted.connect(self.show_cloth_pop_dialog)
        else:
            print("QDialogButtonBox를 찾지 못했습니다.")

    # cloth_pop 다이얼로그 열기
    def show_cloth_pop_dialog(self):
        print("show_cloth_pop_dialog 호출됨")
        self.cloth_pop_dialog = ClothPopDialog()
        self.cloth_pop_dialog.show()  # show()로 창을 띄움

class ClothPopDialog(QDialog):
    def __init__(self):
        super().__init__()

        # cloth_pop.ui 파일 로드
        uic.loadUi('GUI/cloth_pop.ui', self)

        # QLabel 찾기 (이름: spectrum_image1, spectrum_image2)
        self.spectrum_image1 = self.findChild(QLabel, 'spectrum_image1')
        self.spectrum_image2 = self.findChild(QLabel, 'spectrum_image2')

        # 선택한 색상을 미리보기할 QFrame 찾기 (이름: color1, color2)
        self.color_frame1 = self.findChild(QFrame, 'color1')
        self.color_frame2 = self.findChild(QFrame, 'color2')

        # color_frame1, color_frame2가 제대로 로드되었는지 확인
        if self.color_frame1 is None or self.color_frame2 is None:
            print("color1 또는 color2 프레임을 찾을 수 없습니다. UI 파일을 확인하세요.")
        else:
            print("color1 및 color2 프레임을 성공적으로 찾았습니다.")

        # spectrum.png 이미지를 QLabel에 삽입
        self.pixmap = QPixmap('GUI/spectrum.png')
        if self.spectrum_image1 is not None:
            self.spectrum_image1.setPixmap(self.pixmap)
        if self.spectrum_image2 is not None:
            self.spectrum_image2.setPixmap(self.pixmap)

        # QLabel 클릭 이벤트를 위한 마우스 이벤트 필터 설정
        self.spectrum_image1.mousePressEvent = self.get_color_from_spectrum1
        self.spectrum_image2.mousePressEvent = self.get_color_from_spectrum2

    # 스펙트럼 1에서 클릭한 위치의 색상 추출 (color1에 반영)
    def get_color_from_spectrum1(self, event):
        x = event.pos().x()
        y = event.pos().y()

        # QPixmap을 QImage로 변환하여 클릭한 좌표의 색상 추출
        img = self.pixmap.toImage()
        color = img.pixel(x, y)

        # QColor 객체로 변환하여 RGB 값 추출
        qcolor = QColor(color)
        rgb_color = (qcolor.red(), qcolor.green(), qcolor.blue())
        print(f"spectrum_image1에서 선택한 색상: RGB {rgb_color}")

        # 선택한 색상을 color1 프레임에 채우기
        if self.color_frame1 is not None:
            self.update_color_preview(self.color_frame1, qcolor)

    # 스펙트럼 2에서 클릭한 위치의 색상 추출 (color2에 반영)
    def get_color_from_spectrum2(self, event):
        x = event.pos().x()
        y = event.pos().y()

        # QPixmap을 QImage로 변환하여 클릭한 좌표의 색상 추출
        img = self.pixmap.toImage()
        color = img.pixel(x, y)

        # QColor 객체로 변환하여 RGB 값 추출
        qcolor = QColor(color)
        rgb_color = (qcolor.red(), qcolor.green(), qcolor.blue())
        print(f"spectrum_image2에서 선택한 색상: RGB {rgb_color}")

        # 선택한 색상을 color2 프레임에 채우기
        if self.color_frame2 is not None:
            self.update_color_preview(self.color_frame2, qcolor)

    # 선택한 색상으로 프레임의 배경색 변경
    def update_color_preview(self, frame, qcolor):
        # QFrame에 스타일 시트를 적용하여 배경색을 설정
        frame.setStyleSheet(f"background-color: rgb({qcolor.red()}, {qcolor.green()}, {qcolor.blue()});")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
