
from ultralytics import YOLO

model_path = "/home/saurav/Desktop/FireYOLO/best.pt" 
model = YOLO(model_path)

model.export(format = "onxx", imgsz = 640, simplify  = True)