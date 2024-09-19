import json
import socket
import threading
import sys

SERVER_IP = "192.168.0.37"
TCP_PORT = 8888

class GooseTcp:
    def __init__(self):
        self.clients = {}  # 클라이언트 소켓과 주소를 저장하는 딕셔너리
        
    def handle_receive(self, client_socket, client_address):
        """클라이언트로부터 메시지를 수신하는 함수"""
        buffer = ""
        while True:
            try:
                response = client_socket.recv(1024).decode("utf-8")
                if response:
                    buffer += response
                    while "\n" in buffer:  # 구분자 '\n'을 기준으로 메시지를 나눔
                        line, buffer = buffer.split("\n", 1)
                        try:
                            parsed_data = json.loads(line)
                            print(f"received {client_address[0]}: {parsed_data}")
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                if not response:
                    print("disconnect!")
                    break
            except Exception as e:
                print(f"error! {e}")
                break

        # 클라이언트가 연결을 끊으면 딕셔너리에서 제거
        if client_address[0] in self.clients :
            del self.clients[client_address[0]]
        client_socket.close()
    
    #--------------------------------------------------------------------
    def broadcast_message_to_all(self, message):
        """모든 클라이언트에게 메시지를 전송하는 함수"""
        for client_socket in self.clients.values():
            try:
                client_socket.sendall(message.encode('utf-8'))
            except Exception as e:
                print(f"error sending message to client: {e}")

    def send_to_client(self, target_address, message):
        #"""특정 클라이언트에게 메시지를 전송하는 함수"""
        client_socket = self.clients.get(target_address)
        if client_socket:
            try:
                client_socket.sendall(message.encode('utf-8'))
            except Exception as e:
                print(f"error sending message to {target_address}: {e}")
        else:
            print(f"Client {target_address} not found!")
    #------------------------------------------------------------------------
    def handle_send(self, baby_goose_ip, message):
        """메인 스레드에서 처리하는 함수로, 입력과 broadcast 메시지 처리를 담당"""
        if  baby_goose_ip == "all": 
            self.broadcast_message_to_all(message)
        else:
            if self.clients:
                clients_list = list(self.clients.keys())
                if baby_goose_ip in clients_list :
                    print(f"send to : {baby_goose_ip} {message}")
                    self.send_to_client(baby_goose_ip, message)
                else :
                    print(f"Not exist baby goose ip : {baby_goose_ip}")

    def handle_client(self, client_socket, client_address):
        """클라이언트 연결을 처리하는 함수"""
        print(f"{client_address} connected!")
        self.clients[client_address[0]] = client_socket  # 클라이언트 주소를 키로 사용하여 소켓 저장

        receive_thread = threading.Thread(target=self.handle_receive, args=(client_socket, client_address))
        receive_thread.start()

    def start_server(self):
        """최초 TCP/IP connection 실행"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        #server_address = ('192.168.0.35', 9999)  # IP, 포트 설정
        server_address = (SERVER_IP, TCP_PORT)

        server_socket.bind(server_address)
        server_socket.listen(5)  # 최대 5개 연결 대기
        print("-----------------------------------------------------------------------------------")
        print("server started!")
        print("-----------------------------------------------------------------------------------")
        # 클라이언트 연결을 수락하는 스레드
        threading.Thread(target=self.accept_clients, args=(server_socket,)).start()


    def accept_clients(self, server_socket):
        #"""클라이언트 연결을 수락하는 함수"""
        while True:
            client_socket, client_address = server_socket.accept()
            client_thread = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
            client_thread.start()



