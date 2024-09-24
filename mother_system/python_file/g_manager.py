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
        self.from_gui_queue = queue.Queue()
        self.to_gui_queue = queue.Queue()
    
    def communicator(self, g_pipe):
        while True:
            # 먼저 파이프를 체크합니다.
            if g_pipe.poll():
                recv_data = g_pipe.recv()
                self.to_gui_queue.put(recv_data)
            
            # gui_queue를 비블로킹으로 체크합니다.
            try:
                gui_req = self.from_gui_queue.get_nowait()
                if gui_req is None:
                    break
                # print(f"!!!!!!!!!{gui_req[0]}")
                g_pipe.send(gui_req)
                self.from_gui_queue.task_done()
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
        
        
        
        