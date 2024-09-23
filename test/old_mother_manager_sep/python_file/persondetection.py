from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')

model.train(data='./PD_YOLO.yaml' , epochs=100, dropout = 0.2)