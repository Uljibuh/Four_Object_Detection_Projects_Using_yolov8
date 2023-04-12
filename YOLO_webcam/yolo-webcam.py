from ultralytics import YOLO
import cv2
import cvzone
import math

# cap =cv2.VideoCapture(0) # webcam
# cap.set(3,1280)
# cap.set(3,720)
cap =cv2.VideoCapture("/Users/uljibuh/Desktop/Object_Detection/objectdetection/Videos/cars.mp4") # for video



model = YOLO('/Users/uljibuh/Desktop/Object_Detection/yolo-wright/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("/Users/uljibuh/Desktop/Object_Detection/objectdetection/Car_counter/mask.png")


while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # bounging box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            
            w, h = x2-x1,y2-y1

            # confidence
            conf = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(35, y1)), scale=0.3, thickness=int(0.5), offset=5)
                cvzone.cornerRect(img,(x1,y1,w,h),l=9)

            
    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    if cv2.waitKey(0) & 0xFF == ord('q'): # if the 'q' key is pressed
        break
cv2.destroyAllWindows() # close all windows
cap.release() # release the webcam