import socket
import sys
import threading

#
sys.stdout.reconfigure(encoding='utf-8')

def handle_receive(client_socket):
    while True :
        try :
            response = client_socket.recv(1024)
            if not response :
                print("disconnect!")
                break
        except Exception as e :
            print (f"error!{e}")
            break
        print("server to", response)

    
def handle_send(client_socket) :
    while True :
        message = input()
        if message =="exit" :
            print("disconnect!")
            break
        elif message == "" :
            continue
        
        #elif message.startswith("sendto") :
        #    print(f"sendto{}")
      #--------------------------------------------------          
        client_socket.sendall(message.encode('utf-8'))
      #--------------------------------------------------
      #  

def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    server_address = ('192.168.0.35', 9999)
    
    client_socket.connect(server_address)
    print("connect!")
    print("disconnect 'exit'")
    
    receive_thread = threading.Thread(target= handle_receive, args=(client_socket,))
    send_thread = threading.Thread(target= handle_send, args=(client_socket,))
    receive_thread.start()
    send_thread.start()
    
    receive_thread.join()
    send_thread.join()
    
    client_socket.close()

if __name__ == "__main__":
    start_client()
