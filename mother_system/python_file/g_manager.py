import json
import socket
import threading
import sys
from goose_tcp import GooseTcp
from gui import *

stop_event = threading.Event()

class GManager:
    def __init__(self):
        self.goose_tcp = GooseTcp()
        
    def set_request_json(self, baby_goose_ip):
        """baby goose 에게 전송할 data json 생성"""
        request = {
            "baby_goose_ip" : baby_goose_ip,
            "mode" : 21,
            "bounding box" : (100, 200)
        }
        json_data = json.dumps(request)
        
        return json_data


    def run(self, g_pipe):
        app = QApplication(sys.argv)
        window = MainWindow()

        gui_thread = threading.Thread(target=window.show())
        gui_thread.start()
        sys.exit(app.exec_())
        
        
        
        
        
        
    #     self.goose_tcp.start_server()
    #    while True:
    #        if g_pipe.poll():
    #            
    #            self.value = g_pipe.recv()
    #            print(f"g_manager : {self.value}")
    #            cv2.imshow("gcamera frame", self.value)
    #            cv2.waitKey(1)
    #        g_pipe.send(self.value + 1)
#

    # Baby goose 와 TPC/IP 통신 코드

    # def set_request_json(self, baby_goose_ip):
    #     """baby goose 에게 전송할 data json 생성"""
    #     request = {
    #         "baby_goose_ip" : baby_goose_ip,
    #         "mode" : 21,
    #         "bounding box" : (100, 200)
    #     }
    #     json_data = json.dumps(request)
        
    #     return json_data


    # def run(self, g_pipe):
    #     self.goose_tcp.start_server()
    #     # g_pipe.send(self.value)

    #     while True:
    #         if g_pipe.poll():
    #             self.value = int(g_pipe.recv())
    #             print(f"g_manager : {self.value}")
    #             g_pipe.send(self.value + 1)

    #         request_data = self.set_request_json("192.168.0.37")
    #         self.goose_tcp.handle_send("192.168.0.37", request_data)
            

