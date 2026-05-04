import cv2
from ultralytics import YOLO

model_path = "/home/saurav/Desktop/FireYOLO/best.pt" 
model = YOLO(model_path)


ESP32_URL = "http://10.56.153.229"

cap = cv2.VideoCapture(ESP32_URL)


cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

print(f"Connecting to ESP32 stream at {ESP32_URL}...")
print("Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame. Reconnecting or stream ended...")
        break

    frame = cv2.resize(frame, (640, 640))

    results = model(frame, stream=True)

    for result in results:
        annotated_frame = result.plot()
        
        cv2.imshow("Early Forest Fire Detection System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()