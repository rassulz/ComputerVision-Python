import cv2
import torch
from ultralytics import YOLO

# Load the YOLO model
MODEL_PATH = "cubeCamera.pt"
model = YOLO(MODEL_PATH)

# Open webcam (0 for default, change to 1 if external camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = float(box.conf[0])  
            label = result.names[int(box.cls[0])]  

            # Compute the center of the object
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # === Print coordinates in terminal ===
            print(f"{label}: X={center_x}, Y={center_y} (Confidence: {conf:.2f})")

            # === Draw bounding box ===
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # === Display coordinates near bounding box ===
            text = f"{label} ({center_x}, {center_y})"
            cv2.putText(frame, text, (center_x, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the camera feed
    cv2.imshow("YOLO Real-Time Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
