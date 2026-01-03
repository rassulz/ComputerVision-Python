import cv2

# Open default camera (0 for built-in, 1 for external)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()  # Capture frame
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera", frame)  # Show the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
