import socket
import threading
import sys

IP = "192.168.0.37"
PORT = 9999

class GooseTcp:
    def __init__(self):
        self.clients = {}  # 클라이언트 소켓과 주소를 저장하는 딕셔너리
        
    def handle_receive(self, client_socket, client_address):
        """클라이언트로부터 메시지를 수신하는 함수"""
        while True:
            data = client_socket.recv(1024).decode('utf-8')
            try:
                if not data:
                    print(f"({client_address[0]}) disconnected!")
                    break
                if data.startwith("sendto") :
                    parts = data.split()
                    if len(parts) >= 3:
                        target_address = parts[1]
                        message_to_send = ' '.join(parts[2:])
                        self.send_to_client(target_address, f"{client_address[0]} say: {message_to_send}")
                    else :
                        print(f"Invalid sendto format from {client_address}")
                else :
                    print(f"({client_address[0]}) : {data}")

            except Exception as e:
                print(f"({client_address[0]}) : err1or! {e}")
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
    def handle_send(self):
        """메인 스레드에서 처리하는 함수로, 입력과 broadcast 메시지 처리를 담당"""
        while True:
            message = "sendto"
            # message = input()  # 서버에서 입력 처리 (자식 프로세스나 백그라운드 스레드에서는 표준 입력이 연결되어 있지 않을 수 있습니다.)
            if message == "broadcast!!":
                broadcast_message = input("broadcast message: ")
                sys.stdout.flush()  # 출력 플러시
                self.broadcast_message_to_all(broadcast_message)
            elif message=="sendto":
                if self.clients:
                    print("Choose a client to send a message !")
                    clients_list = list(self.clients.keys())
                    for idx, client_ip in enumerate(clients_list) :
                        print("----------------------")
                        print(f"{idx +1}. {client_ip}")
                        print("----------------------")
                    try :
                        # client_choice = int(input("Enter the number of the client:"))
                        client_choice = 1
                        if 0 < client_choice <= len(clients_list) :
                            target_address = clients_list[client_choice-1]
                            # message_to_send = input("Message :")
                            message_to_send = "message!!"
                            self.send_to_client(target_address, message_to_send)
                        else :
                            print("Invalid number!")
                    except ValueError :
                        print("enter a valid number")
                else :
                    print("No client")
            elif message == "exit":
                print("Server shutting down...")
                break
            else:
                print("Invalid command! Use 'broadcast!!' or 'sendto'.")

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
        server_address = (IP, PORT)

        server_socket.bind(server_address)
        server_socket.listen(5)  # 최대 5개 연결 대기
        print("-----------------------------------------------------------------------------------")
        print("server started!")
        print("-----------------------------------------------------------------------------------")
        print("type1 | 'broadcast!!' to send a message to all clients")
        print("type2 | 'sendto <client_address> <message>' to send a message to a specific client")
        print("-----------------------------------------------------------------------------------")
        # 클라이언트 연결을 수락하는 스레드
        threading.Thread(target=self.accept_clients, args=(server_socket,)).start()

        # 메인 스레드에서 입력 대기
        self.handle_send()

    def accept_clients(self, server_socket):
        #"""클라이언트 연결을 수락하는 함수"""
        while True:
            client_socket, client_address = server_socket.accept()
            client_thread = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
            client_thread.start()



