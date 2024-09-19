import socket
import threading
import sys
from goose_tcp import GooseTcp
import cv2


class GManager:
    def __init__(self):
        self.goose_tcp = GooseTcp()
        self.value = 1


    def run(self, g_pipe):
        # self.goose_tcp.start_server()
        while True:
            if g_pipe.poll():
                
                self.value = g_pipe.recv()
                print(f"g_manager : {self.value}")
                cv2.imshow("gcamera frame", self.value)
                cv2.waitKey(1)
                #g_pipe.send(self.value + 1)

