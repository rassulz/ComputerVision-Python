import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH).to('cuda')  # Run on GPU

# Open webcam
cap = cv2.VideoCapture(6)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO with improved settings
    results = model(frame, conf=0.6, iou=0.4)  # Higher confidence, lower IoU


    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = float(box.conf[0])  
            label = result.names[int(box.cls[0])]  

            # Draw only if confidence is high
            if conf > 0.6:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
