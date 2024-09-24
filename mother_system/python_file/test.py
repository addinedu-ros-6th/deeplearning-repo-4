from PyQt5 import uic
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog, 
                             QLabel, QDialogButtonBox,
                             QPushButton, QFrame, QVBoxLayout,
                             QTextEdit)
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # find_man.ui 파일 로드
        uic.loadUi('UI_file/find_man.ui', self)

        # QTextEdit 찾기 (디자이너에서 'status_log'으로 이름을 지정한 경우)
        self.status_log = self.findChild(QTextEdit, 'status_log')

        # QTimer 설정하여 로그 업데이트
        #self.timer = QTimer(self)
        #self.timer.timeout.connect(self.update_log)
        #self.timer.start(1000)  # 1초마다 로그 업데이트

    def update_log(self):
        # 로그 메시지 추가
        self.add_log("로그 업데이트 중...")

    def add_log(self, message):
        """로그 위젯에 메시지를 추가하는 함수"""
        self.status_log.append(message)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())