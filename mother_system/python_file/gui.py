from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QLabel, QDialogButtonBox, QPushButton, QFrame
from PyQt5.QtGui import QPixmap, QImage,QLinearGradient, QPainter, QColor
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt 
import cv2
import queue
import sys
import socket
import pickle
import struct
import os 
import datetime 
import numpy as np
import colorsys

MAX_DGRAM = 65507  # UDP의 최대 패킷 크기

# class VideoReceiver(QThread):
#     frame_received = pyqtSignal(QImage, str)  # QLabel로 보낼 처리된 프레임

#     def __init__(self, gmanager,label_name,parent=None):
#         super().__init__(parent)
#         self.gmanager = gmanager
#         self.label_name = label_name        
#         self.running = True


#     def run(self):
#         while self.running:
#             if not self.gmanager.comm_queue.empty():
#                 frame = self.gmanager.comm_queue.get()
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 # QImage로 변환 rgb
#                 h, w, ch = frame_rgb.shape
#                 bytes_per_line = ch * w
#                 q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
#                 if q_img.isNull():
#                     print("QImage 변환 실패")
#                 else:
#                     print(f"QImage 변환 성공: 크기 {q_img.size()}")

#                 # QLabel에 이미지 설정
#                 self.frame_received.emit(q_img, self.label_name)

#     def stop(self):
#         self.running = False  # 스레드 종료
#         #self.socket.close()
#         self.wait()

class MainWindow(QMainWindow):
    def __init__(self, gmanager):
        super().__init__()
        self.gmanager = gmanager

        # main.ui 파일을 불러와서 메인 윈도우로 사용 (GUI 폴더 경로 추가)
        uic.loadUi('UI_file/main.ui', self)

        # 메인 화면에 있는 버튼을 찾고 클릭 이벤트 연결
        self.btn_airport_info = self.findChild(QPushButton, 'airport_info')
        self.btn_find_man = self.findChild(QPushButton, 'find_man')

        # 버튼 클릭 이벤트 연결
        if self.btn_airport_info is not None:
            self.btn_airport_info.clicked.connect(self.show_airport_info)
        else :
            print("ui를 읽어오지 못함")

        if self.btn_find_man is not None:
            self.btn_find_man.clicked.connect(self.show_input_face)
        else :
            print("ui를 읽어오지 못함")
            
    # air_port_info 창 열기
    def show_airport_info(self):
        self.airport_info_window = QDialog()
        uic.loadUi('UI_file/air_port_info.ui', self.airport_info_window)
        
        map1_label = self.airport_info_window.findChild(QLabel, 'map1')
        map2_label = self.airport_info_window.findChild(QLabel, 'map2')
        
        pixmap1 = QPixmap('./etc_images/Maple_Mini_map.png')
        pixmap2 = QPixmap('./etc_images/maple_world_map.png')
        
        if map1_label is not None :
            map1_label.setPixmap(pixmap1)
        else :
            print("no search label")
            
        if map2_label is not None :
            map2_label.setPixmap(pixmap2)
        else :
            print("no search label")
            
        self.airport_info_window.exec_()
        

    # input_face 창 열기 및 사람찾기 버튼 클릭 시 실행
    def show_input_face(self):
        self.input_face_dialog = InputFaceDialog(self.gmanager)
        self.input_face_dialog.exec_()

        if self.input_face_dialog.isHidden():
            self.show_find_man()

    # find_man 창 열기
    def show_find_man(self):
        print("show_find_man 호출됨")
        self.find_man_window = FindManWindow(self.gmanager)
        self.find_man_window.show()  # show()로 창을 띄움
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()  # 다이얼로그 종료


class InputFaceDialog(QDialog):
    def __init__(self, gmanager, parent=None):
        super().__init__(parent)
        self.gmanager = gmanager
        
        self.face_detected_count = 0
        self.required_detection_count = 10  # 연속으로 인식되면 얼굴 저장
        self.center_tolerance = 50  # 중앙으로부터 50픽셀 내의 얼굴만 저장

        # input_face.ui 파일 로드
        uic.loadUi('UI_file/input_face.ui', self)

        # QLabel 찾기 (QLabel의 이름은 'input_video')
        self.webcam_label = self.findChild(QLabel, 'input_video')
      
        # OpenCV를 사용하여 웹캠 연결
        self.cap = cv2.VideoCapture(2)

        # 얼굴 인식을 위한 Haar Cascade XML 파일 경로 설정
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')  # 얼굴 인식용
        self.glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

        # QTimer 설정하여 주기적으로 프레임 업데이트
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame1)
        self.timer.start(25)  # 30ms마다 업데이트
        
        print("1")
        self.dialog_button_box = self.findChild(QDialogButtonBox, 'register_button')
        self.dialog_button_box.accepted.connect(self.save_current_frame)
        
        print("2") 
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()  # 다이얼로그 종료
        
    def save_current_frame(self):
        ret, frame = self.cap.read()  # 웹캠에서 현재 프레임 읽기
        if ret:
            face_frame = frame  # 현재 프레임을 저장
            self.save_face_image(face_frame)# 얼굴 저장 함수 호출 
            self.cap.release()
            print("저장 완료")   
        
        
        else :
            print("프레임을 읽어오지 못함", ret)
            
        self.show_cloth_pop_dialog()

    def accept(self):
        self.clean_up()
        super().accept()

    def reject(self):
        self.clean_up()
        super().reject()

    def clean_up(self):
        # 스레드 및 소켓 해제, 타이머 중지
          # 웹캠 해제
        self.timer.stop()

    # 창 닫을 때 웹캠 해제
    def closeEvent(self, event):
        self.clean_up()  # 자원 해제
        event.accept()

    # 웹캠 프레임을 QLabel에 업데이트하고 얼굴 인식 후 저장
    def update_frame1(self):
        ret, frame = self.cap.read()
        
        if ret == True:
            original_frame = frame.copy() # 얼굴 바운딩 박스 제거하
            
            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width // 2
            frame_center_y = frame_height // 2
            
            central_area_size = 100
            top_left_x = frame_center_x - central_area_size
            top_left_y = frame_center_y - central_area_size
            bottom_right_x = frame_center_x + central_area_size
            bottom_right_y = frame_center_y + central_area_size
            
                    
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            glasses = self.glasses_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30) )
            print(f"탐지된 얼굴, 안경 수 {len(faces),len(glasses)}")
            print(self.face_detected_count)
            # 얼굴이 인식되면 사각형 그리기 및 얼굴 저장
            if len(faces) or len(glasses) > 0:
                for (x, y, w, h) in faces:
                    # 너무 작거나 큰 얼굴은 무시
                    face_area = w * h
                    if face_area < 5000 or face_area > 50000:  # 얼굴 크기 기준
                        self.face_detected_count = 0
                        continue
                    
                    face_center_x = x+w//2
                    face_center_y = x+h//2
                    
                    
                    # 중앙에서 벗어난 얼굴은 무시
                    if (not (top_left_x 
                             <= face_center_x 
                             <= bottom_right_x and top_left_y 
                             <= face_center_y 
                             <= bottom_right_y)):
                        self.face_detected_count = 0
                        continue
                    #########################
                    bounding_box= (x,y,w,h)#바운딩박스 좌표정보
                    #########################
                   #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    self.face_detected_count += 1
                   

                    if self.face_detected_count >= self.required_detection_count:#10번
                        print("stop")
                        face_frame = frame[y:y + 2*h, x:x + 2*w]  # 얼굴 영역만 잘라서 저장

                        ##############################################
                        self.save_face_image(face_frame)  # 얼굴 저장 함수 호출

                        # 얼굴이 인식되고 저장되면 OK 버튼 누른 것처럼 cloth_pop 창 열기
                        
                        self.timer.stop()  # 타이머 중지 (더 이상 얼굴 탐지하지 않도록)
                        self.cap.release()  # 카메라 해제
                       # self.show_cloth_pop_dialog()
                        break  # 한 번 처리 후 루프 종료
            else:
                self.face_detected_count = 0

            #중앙 범위 바운딩박스
            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y),(0,255,0), 2)

            # OpenCV 프레임을 QImage로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # QLabel에 QPixmap으로 변환한 이미지 표시
            self.webcam_label.setPixmap(QPixmap.fromImage(qimg))

    # 얼굴 이미지 저장
    def save_face_image(self, frame):
        print("save")
        save_dir = os.path.expanduser('./face_images')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 타임스탬프를 이용해 파일명 생성
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        #timestamp = 사진 정보 저장
        #file_path = os.path.join(save_dir, f'face_{timestamp}.jpg')
        file_path = os.path.join(save_dir,'face_image.jpg')#이미지 경로
        self.gmanager.from_gui_queue.put(11) # 11: 미아 사진 등록 완료 코드

        # 이미지 저장
        cv2.imwrite(file_path, frame)
        self.close()
        self.show_cloth_pop_dialog()
        ####################################
        #self.send_image_to_Dmanager(self, )
        

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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()  # 다이얼로그 종료



class ClothPopDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.top_color = None
        self.bottom_color = None
        
        # cloth_pop.ui 파일 로드
        uic.loadUi('UI_file/cloth_pop.ui', self)

        # QFrame 찾기 (color1, color2)
        self.color_frame1 = self.findChild(QFrame, 'color1')
        self.color_frame2 = self.findChild(QFrame, 'color2')

        # Top 버튼들
        top_buttons = ['Red', 'Red2', 'Orange', 'Yellow', 'Green', 
                       'Blue', 'Navy', 'Violet', 'White', 'Black', 'Gray']
        for name in top_buttons:
            button = self.findChild(QPushButton, name)
            button.clicked.connect(self.set_color)

        # Bottom 버튼들
        bottom_buttons = ['Red_2', 'Red2_2', 'Orange_2', 'Yellow_2',
                          'Green_2','Blue_2', 'Navy_2',
                          'Violet_2', 'White_2','Black_2', 'Gray_2']
        for name in bottom_buttons:
            button = self.findChild(QPushButton, name)
            button.clicked.connect(self.set_color)

        # QDialogButtonBox 찾기
        self.dialog_button_box = self.findChild(QDialogButtonBox, 'register_button')

        # OK 버튼 누르면 find_man 창 열기
        if self.dialog_button_box is not None:
            self.dialog_button_box.accepted.connect(self.show_find_man)

    def set_color(self):
        button = self.sender()
        object_name = button.objectName()

        # Color range에 맞게 색상 설정
        if '_' in object_name:
    # '_'로 나눈 첫 번째 부분만 가져와서 bottom_color에 저장
            self.bottom_color = object_name.split('_')[0].lower()
            print(f"bottom_color: {self.bottom_color}")
            color_range = self.get_color_range(self.bottom_color)  # object_name 대신 수정된 bottom_color로 전달
            self.set_frame_color(self.color_frame2, color_range)  # Bottom 프레임에 색상 설정
        else:
            self.top_color = object_name.lower()
            print(f"top_color : {self.top_color}")
            color_range = self.get_color_range(object_name)
            self.set_frame_color(self.color_frame1, color_range)  # Top 프레임에 색상 설정

    def get_color_range(self, object_name):
        color_ranges = {
            'red1': [(0, 100, 100), (10, 255, 255)],
            'red2': [(170, 100, 100), (180, 255, 255)],
            'orange': [(10, 100, 100), (25, 255, 255)],
            'yello': [(25, 100, 100), (35, 255, 255)],
            'green': [(35, 100, 100), (85, 255, 255)],
            'blue': [(85, 100, 100), (125, 255, 255)],
            'navy': [(125, 100, 100), (140, 255, 255)],
            'violet': [(140, 100, 100), (170, 255, 255)],
            'white': [(0, 0, 180), (180, 30, 255)],
            'gray': [(0, 0, 100), (180, 30, 180)],
            'black': [(0, 0, 1), (180, 50, 100)]
        }
        return color_ranges.get(object_name, [(0, 0, 0), (0, 0, 0)])

    def set_frame_color(self, frame, color_range):
        # HSV -> RGB 변환 후 QFrame에 배경색 적용
        lower_hsv, upper_hsv = color_range

        # 상한 HSV 값을 사용하여 RGB로 변환 (더 강한 색상을 나타내기 위해)
        h, s, v = upper_hsv
        r, g, b = colorsys.hsv_to_rgb(h / 180, s / 255, v / 255)
        color_hex = f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'  # RGB 값을 HEX 코드로 변환
        print(color_hex)
        # QFrame의 스타일시트로 배경색 설정
        frame.setStyleSheet(f"background-color: {color_hex};")


# 새로운 QFrame 클래스 정의 (그라데이션을 그릴 수 있는 프레임)
class GradientFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.color_range = None  # 그라데이션 색상 범위

    def set_gradient(self, color_range):
        self.color_range = color_range
        self.update()  # 프레임을 갱신하여 paintEvent를 호출

    def paintEvent(self, event):
        if self.color_range is None:
            return  # 색상 범위가 설정되지 않으면 아무 작업도 하지 않음

        lower_hsv, upper_hsv = self.color_range

        painter = QPainter(self)
        if not painter.isActive():
            print("Painter is not active")
            return

        gradient = QLinearGradient(0, 0, self.width(), 0)  # 수평 그라데이션

        # HSV -> RGB 변환 후 그라데이션 설정
        h_start, s_start, v_start = lower_hsv[0] / 180, lower_hsv[1] / 255, lower_hsv[2] / 255
        h_end, s_end, v_end = upper_hsv[0] / 180, upper_hsv[1] / 255, upper_hsv[2] / 255

        for i in range(11):
            ratio = i / 10
            h = h_start + ratio * (h_end - h_start)
            s = s_start + ratio * (s_end - s_start)  # 채도 그라데이션
            v = v_start + ratio * (v_end - v_start)  # 명도 그라데이션
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            gradient.setColorAt(ratio, QColor(int(r * 255), int(g * 255), int(b * 255)))

        # 프레임 전체에 그라데이션 적용
        painter.fillRect(self.rect(), gradient)
        painter.end()

class FindManWindow(QMainWindow):
    def __init__(self, gmanager):
        super().__init__()
        self.gmanager = gmanager
        self.is_identified = False

        # find_man.ui 파일 로드
        uic.loadUi('UI_file/find_man.ui', self)

        # QLabel 찾기 (QLabel의 이름은 'goose_video1', 'goose_video2', 'goose_video3')
        self.goose_video1 = self.findChild(QLabel, 'goose_video1')
        self.goose_video2 = self.findChild(QLabel, 'goose_video2')
        self.goose_video3 = self.findChild(QLabel, 'goose_video3')
        
        self.find_button = self.findChild(QDialogButtonBox, 'find_button')
        

        # QTimer 설정하여 주기적으로 프레임 업데이트
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)  # Update every 30 milliseconds
        
        self.find_button.accepted.connect(self.show_check_man)
    
        
    def show_check_man(self) :
        print("check_man.ui실행")
        self.check_man_dialog = checkManDialog(self.gmanager, self)
        self.check_man_dialog.show()
        

    # 프레임을 QLabel에 업데이트하는 함수
    def update_frame(self):
        if not self.gmanager.to_gui_queue.empty():
            # D-manager 로부터 가져온 정보들 
            mother_req, is_identified, frame = self.gmanager.to_gui_queue.get()
            if is_identified:
                self.is_identified = True
                self.show_check_man_dialog()##
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            if q_img.isNull():
                print("QImage 변환 실패")
            else:
                print(f"QImage 변환 성공: 크기 {q_img.size()}")

            # QLabel에 이미지 설정
            self.goose_video1.setPixmap(QPixmap.fromImage(q_img))

    ##
    def show_check_man_dialog(self) :
        print("check_man호출")
        self.show_check_man()
    
    # 창 닫을 때 타이머 정지
    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            QApplication.quit()  # 프로그램 종료
        else:
            super().keyPressEvent(event)


class checkManDialog(QDialog) :
    def __init__(self, gmanager, main_window) :
        super().__init__()
        self.gmanager = gmanager
        self.main_window = main_window
        
        uic.loadUi('UI_file/check_man.ui', self) # ui불러오기
        self.check_image = self.findChild(QLabel, 'check_image')
        self.check_button = self.findChild(QDialogButtonBox, 'check_button')

        check_image = self.check_image
        
        image_check = QPixmap('./check_image/checkman_image.png') # 사진 확인 경로 수정
        
        if check_image is not None :
            check_image.setPixmap(image_check)
        else :
            print("no search check_image")
        
        #버튼 수락 시 show_tracking_man호출
        self.check_button.accepted.connect(self.show_tracking_man)
        
    def show_tracking_man(self) :
        print("Tracking_man.ui 실행")
        self.tracking_man_window = TrackingManWindow(self.gmanager, self.main_window)
        self.tracking_man_window.show()
        self.close() # Dialog 종료
        
class TrackingManWindow(QMainWindow) :
    def __init__(self, gmanager, main_window) :
        super().__init__()
        self.gmanager = gmanager 
        self.main_window = main_window
        
        uic.loadUi('UI_file/tracking_man.ui', self)
        
        map_image = self.findChild(QLabel, 'tracking_map')
        map_path = QPixmap('./etc_images/Maple_Mini_map.png')
        
        if map_image is not None :
            map_image.setPixmap(map_path)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.call_update_frame)
        self.timer.start(1)  # Update every 30 milliseconds
        
        
    def call_update_frame(self) :
        self.main_window.update_frame()      
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            QApplication.quit()  # 프로그램 종료
        else:
            super().keyPressEvent(event)
        

#if __name__ == '__main__':
#  #  app = QApplication(sys.argv)
#    window = MainWindow()
#    window.show()
#    sys.exit(app.exec_())
