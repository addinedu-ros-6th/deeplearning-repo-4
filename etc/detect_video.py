from ultralytics import YOLO 
import cv2
import os

video_path = "/home/gon/dev_ws/yolo/day1/ch06_detect/NewJeans (뉴진스) 'Bubble Gum' Official MV [ft70sAYrFyY].webm"
daniel_model = YOLO("/home/gon/dev_ws/yolo/day1/ch06_detect/runs/detect/train4/weights/best.pt")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error : Could not read frame")
        break

    # 특정 프레임 간격으로 처리 (예: 5프레임마다)
    # if frame_count % 2 == 0:
    results = daniel_model.predict(frame)
    annotated_frame = results[0].plot()
    resized_frame = cv2.resize(annotated_frame, (640, 480))
    frame = cv2.putText(img=resized_frame, text="trained", org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=2)

    cv2.imshow("Newjeans - Daniel", frame)

    frame_count += 1
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()