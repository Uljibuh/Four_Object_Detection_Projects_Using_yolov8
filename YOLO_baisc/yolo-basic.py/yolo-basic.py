import cv2
from ultralytics import YOLO

model = YOLO('/Users/uljibuh/Desktop/Object_Detection/yolo-wright/yolov8n.pt')
results = model("/Users/uljibuh/Desktop/Object_Detection/objectdetection/Chapter-5/yolo-basic.py/Images/2.png", show=True)

