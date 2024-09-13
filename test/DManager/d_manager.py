import threading
import cv2
from WetFloorDetector import WetFloorDetect
from add import add, subtract

# 사용자 입력을 받을 전역 변수
model_sel = 0
stop_event = threading.Event()

class D_MANAGER:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.WetFloorDetect = WetFloorDetect()
    def process_input(self):
        global model_sel
        while True:
            try:
                model_sel = int(input("모델을 선택하세요 (1: WetFloorDetector, 2: Add Function): "))
                if model_sel == 1 or model_sel == 2:
                    pass
            except ValueError:
                print("올바른 숫자를 입력하세요.")
    
    def camera_and_modelsel(self):
        while True: 
            
            ret, self.frame = self.cap.read()
            self.frame = cv2.resize(self.frame, (640, 480))
            if not ret:
                print("Error : Could not read frame")
                break

            if model_sel == 1:
                results = self.WetFloorDetect.inference_WF_Detect(self.frame)
                annotated_frame = results[0].plot()
                resized_frame = cv2.resize(annotated_frame, (640, 480))
                self.frame = cv2.putText(img=resized_frame, text="trained", \
                                    org=(30, 30), \
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                                    fontScale=2, color=(0, 0, 255),\
                                    thickness=2)
            elif model_sel == 2: #Missing Detector
                self.frame = cv2.putText(img=resized_frame, text="MISSING", \
                                    org=(30, 30), \
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                                    fontScale=2, color=(0, 0, 255),\
                                    thickness=2)
                

            cv2.imshow("camera frame", self.frame)
            if (cv2.waitKey(1) & 0xff == ord('q')):
                break

        self.cap.release()
        cv2.destroyAllWindows()
    
   
def initialize_threads():
    
    global stop_event
    stop_event.set()  # 실행 중인 모든 스레드에 종료 신호 보내기
    stop_event.clear()  # 새로 실행될 스레드를 위해 이벤트 초기화


if __name__ == "__main__":

    MANAGER = D_MANAGER()
    
    initialize_threads()
    # 사용자 입력을 처리할 쓰레드 시작
    input_thread = threading.Thread(target=MANAGER.process_input)
    input_thread.start()

    camera_thread = threading.Thread(target=MANAGER.camera_and_modelsel)
    camera_thread.start()
