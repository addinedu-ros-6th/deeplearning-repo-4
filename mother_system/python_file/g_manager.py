import json
import queue
import socket
import threading
import sys
from goose_tcp import GooseTcp
from gui import *
import time

stop_event = threading.Event()

class GManager:
    def __init__(self):
        self.goose_tcp = GooseTcp()
        self.gui_queue = queue.Queue()
        self.comm_queue = queue.Queue()
        
    def set_request_json(self, baby_goose_ip):
        """baby goose 에게 전송할 data json 생성"""
        request = {
            "baby_goose_ip" : baby_goose_ip,
            "mode" : 21,
            "bounding box" : (100, 200)
        }
        json_data = json.dumps(request)
        
        return json_data
    
    def communicator(self, g_pipe):
        while True:
            # 먼저 파이프를 체크합니다.
            if g_pipe.poll():
                frame = g_pipe.recv()
                print("poll!!!")
                self.comm_queue.put(frame)
            
            # gui_queue를 비블로킹으로 체크합니다.
            try:
                data = self.gui_queue.get_nowait()
                if data is None:
                    break
                print(f"Communicator : {data}")
                self.gui_queue.task_done()
            except queue.Empty:
                pass  # 큐가 비어 있으면 넘어갑니다.

            # 너무 빠른 루프를 방지하기 위해 약간 대기합니다.
            time.sleep(0.01)  # 10ms 대기



    def run(self, g_pipe):
        app = QApplication(sys.argv)
        window = MainWindow(self)
        window.show()

        comm_thread = threading.Thread(target=self.communicator, args=(g_pipe,))
        comm_thread.start()

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
            

