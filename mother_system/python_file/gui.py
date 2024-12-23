from PyQt5 import uic
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog, 
                             QLabel, QDialogButtonBox,
                             QPushButton, QFrame, QVBoxLayout,
                             QTextEdit,QGraphicsOpacityEffect)
from PyQt5.QtGui import QPixmap, QImage,QLinearGradient, QPainter, QColor, QFont
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt ,QPropertyAnimation
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
        self.gmanager.from_gui_queue.put((1, None))

        self.input_face_dialog = InputFaceDialog(self.gmanager, self)
        self.input_face_dialog.exec_()

        #if self.input_face_dialog.isHidden():
        #    self.show_find_man()

    # find_man 창 열기
    def show_find_man(self, top_color, bottom_color):
        print("show_find_man 호출됨")

        if (top_color is not None) and (bottom_color is not None): 
            self.gmanager.from_gui_queue.put((11, {"top_color": top_color, "bottom_color": bottom_color})) # 11: 미아 사진 및 의상 색상 정보 입력 완료 코드

        self.find_man_window = FindManWindow(self.gmanager)
        self.find_man_window.show()  # show()로 창을 띄움
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()  # 다이얼로그 종료


class InputFaceDialog(QDialog):
    def __init__(self, gmanager, main_window, parent=None):
        super().__init__(parent)
        self.gmanager = gmanager
        self.main_window = main_window  # MainWindow 인스턴스 저장

        
        self.face_detected_count = 0
        self.required_detection_count = 5  # 연속으로 인식되면 얼굴 저장
        self.center_tolerance = 50  # 중앙으로부터 50픽셀 내의 얼굴만 저장

        # input_face.ui 파일 로드
        uic.loadUi('UI_file/input_face.ui', self)

        # QLabel 찾기 (QLabel의 이름은 'input_video')
        self.webcam_label = self.findChild(QLabel, 'input_video')
      
        # OpenCV를 사용하여 웹캠 연결
        self.cap = cv2.VideoCapture(0)

        if self.cap :
            print("camera on")
        else :
            print("not find camera")

        # 얼굴 인식을 위한 Haar Cascade XML 파일 경로 설정
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')  # 얼굴 인식용
        self.glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

        # QTimer 설정하여 주기적으로 프레임 업데이트
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame1)
        self.timer.start(1)  # 30ms마다 업데이트
        
        print("촬영시작")
        self.dialog_button_box = self.findChild(QDialogButtonBox, 'register_button')
        self.dialog_button_box.accepted.connect(self.save_current_frame)
        self.dialog_button_box.rejected.connect(self.cancel_inputface)
        
    def cancel_inputface(self) :
        print("cancel_input_face")
        self.reject()
        
        
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
            self.show_cloth_pop_dialog()########################지울 부분


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
            frame = cv2.flip(frame, 1)
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

        # 이미지 저장
        cv2.imwrite(file_path, frame)
        self.close()
        self.show_cloth_pop_dialog()
        ####################################
        #self.send_image_to_Dmanager(self, )
        

    # cloth_pop 다이얼로그 열기
    def show_cloth_pop_dialog(self):
        self.cloth_pop_dialog = ClothPopDialog(self.main_window)
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
    def __init__(self, main_window):  # main_window를 받아옴
        super().__init__()
        self.main_window = main_window  # main_window 저장

        self.top_color = None
        self.bottom_color = None

        # cloth_pop.ui 파일 로드
        uic.loadUi('UI_file/cloth_pop.ui', self)

        # QFrame 찾기 (color1, color2)
        self.color_frame1 = self.findChild(QFrame, 'color1')
        self.color_frame2 = self.findChild(QFrame, 'color2')

        # Top 버튼들
        top_buttons = {
            'Red': 'red.png',
            'Orange': 'orange.png',
            'Yellow': 'yellow.png',
            'Green': 'green.png',
            'Blue': 'blue.png',
            'Navy': 'navy.png',
            'Violet': 'violet.png',
            'White': 'white.png',
            'Black': 'black.png',
            'Gray': 'gray.png'
        }


        
        for name, image in top_buttons.items():
            button = self.findChild(QPushButton, name)
            if button:
                # 각 버튼에 맞는 이미지 배경 설정
                if name == 'Black' :
                    button.setStyleSheet(f"""
                        QPushButton {{
                            background-image: url(./etc_images/{image});
                            background-repeat: no-repeat;
                            background-position: center;
                            border: none;
                            color : white;
                        }}
                    """)
                else :
                    button.setStyleSheet(f"""
                        QPushButton {{
                            background-image: url(./etc_images/{image});
                            background-repeat: no-repeat;
                            background-position: center;
                            border: none;
                            color : black;
                        }}
                    """)
            button.clicked.connect(self.set_color)

        # Bottom 버튼들
        bottom_buttons = {
            'Red_2': 'red.png',
            'Orange_2': 'orange.png',
            'Yellow_2': 'yellow.png',
            'Green_2': 'green.png',
            'Blue_2': 'blue.png',
            'Navy_2': 'navy.png',
            'Violet_2': 'violet.png',
            'White_2': 'white.png',
            'Black_2': 'black.png',
            'Gray_2': 'gray.png'
        }
        for name, image in bottom_buttons.items():
            button = self.findChild(QPushButton, name)
            if button:
                # 각 버튼에 맞는 이미지 배경 설정
                if name == 'Black_2' :
                    button.setStyleSheet(f"""
                        QPushButton {{
                            background-image: url(./etc_images/{image});
                            background-repeat: no-repeat;
                            background-position: center;
                            border: none;
                            color : white;
                        }}
                    """)
                else :
                    button.setStyleSheet(f"""
                        QPushButton {{
                            background-image: url(./etc_images/{image});
                            background-repeat: no-repeat;
                            background-position: center;
                            border: none;
                            color : black;
                        }}
                    """)
            button.clicked.connect(self.set_color)

        # QDialogButtonBox 찾기
        self.dialog_button_box = self.findChild(QDialogButtonBox, 'spectrum_button')

        # OK 버튼 누르면 find_man 창 열기

        if self.dialog_button_box is not None:
            self.dialog_button_box.accepted.connect(lambda: self.show_find_man(self.top_color, self.bottom_color))
            self.dialog_button_box.rejected.connect(self.cancel_cloth)

    def cancel_cloth(self):
        print("cancel cloth")
        self.reject()

    def set_color(self):
        button = self.sender()
        object_name = button.objectName()

        # top_color 또는 bottom_color에 설정된 값 저장
        if '_' in object_name:
            self.bottom_color = object_name.split('_')[0].lower()
            print(f"bottom_color: {self.bottom_color}")
            # 색상 설정 관련 코드 생략
        else:
            self.top_color = object_name.lower()
            print(f"top_color: {self.top_color}")
        # 색상 설정 관련 코드 생략

    # find_man 창 열기
    def show_find_man(self, top_color, bottom_color):
        if isinstance(self.main_window, MainWindow):
            self.main_window.show_find_man(top_color, bottom_color)  # MainWindow의 show_find_man 메서드 호출
        else:
            print("MainWindow 인스턴스가 아님")
        self.close()



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
        
        self.close_button = self.findChild(QPushButton, 'close_button')
        
        self.status_widget = self.findChild(QTextEdit, 'status_log')
        
        self.find_button = self.findChild(QDialogButtonBox, 'find_button')
        self.minimap = self.findChild(QLabel, 'minimap')
        
        pix_minimap = QPixmap('./etc_images/Maple_Mini_map.png')
        
        if self.minimap is not None :
            self.minimap.setPixmap(pix_minimap)
        else :
            print("no search label")
        
        self.close_button.clicked.connect(self.close)

        # QTimer 설정하여 주기적으로 프레임 업데이트
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)  # Update every 30 milliseconds
        
        self.find_button.accepted.connect(self.show_check_man)
        self.find_button.rejected.connect(self.cancel_findman)
        self.add_log("탐색을~~~~~~~시작~~~~~~!!!~~~~~~~~~하겠습니다~~~~~~~~~~~!!!!!!!!!!!")
        
    def update_log(self) :
        self.add_log("로그 업데이트---")
        
    def add_log(self, message) :
        self.status_widget.append(message)
        
    def log_clear(self) :
        self.status_widget.clear()
        
    def cancel_findman(self) :
        print("cancel findman!0")
        self.close()

        
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
        
        self.check_button.accepted.connect(self.find_yes_man)
        self.check_button.rejected.connect(self.cancel_button)
        
        self.label = self.findChild(QLabel, 'label')
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.label.setFont(font)
        
    def find_yes_man(self) :

        self.main_window.log_clear()
        self.main_window.add_log("찾았다~~얄리~!!!!!!!~~~~!!~~~~~!!!!")
        self.gmanager.from_gui_queue.put((28, None)) # 8: 부모 Yes 버튼 클릭
        self.tracking_man_window = TrackingManWindow(self.gmanager, self.main_window)
        self.reject()
        
    def cancel_button(self) :
        print("push cancel!!")
        self.main_window.add_log("계속 탐색할게용~!!!!!!!~~~~!!~~~~~!!!!")
        self.main_window.log_clear()
        self.gmanager.from_gui_queue.put((29, None)) # 29: 부모 No 버튼 클릭
        self.reject()
        
        
        
    #def show_tracking_man(self) :
    #    print("Tracking_man.ui 실행")
    #    self.gmanager.from_gui_queue.put((28, None)) # 8: 부모 Yes 버튼 클릭
    #    self.tracking_man_window = TrackingManWindow(self.gmanager, self.main_window)
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
#    app = QApplication(sys.argv)
#    window = MainWindow()
#    window.show()
#    sys.exit(app.exec_())
