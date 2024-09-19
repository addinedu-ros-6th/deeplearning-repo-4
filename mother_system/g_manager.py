import socket
import threading
import sys
from goose_tcp import GooseTcp



class GManager:
    def __init__(self):
        self.goose_tcp = GooseTcp()
        self.value = 1


    def run(self, g_pipe):
        # self.goose_tcp.start_server()
        g_pipe.send(self.value)
        while True:
            if g_pipe.poll():
                self.value = int(g_pipe.recv())
                print(f"g_manager : {self.value}")
                g_pipe.send(self.value + 1)

