import cv2
from pyzbar.pyzbar import decode

def detect_qr_codes(frame):
    # Decode QR codes in the frame
    qr_codes = decode(frame)
    positions = []  # List to store positions of detected QR codes
    
    for qr_code in qr_codes:
        # Extract bounding box and draw rectangle around QR code
        (x, y, w, h) = qr_code.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate the center position of the QR code
        center_x = x + w // 2
        center_y = y + h // 2
        positions.append((center_x, center_y))
        
        # Extract QR code data
        qr_data = qr_code.data.decode('utf-8')
        qr_type = qr_code.type
        
        # Display the QR code data and position on the frame
        text = f"{qr_type}: {qr_data} ({center_x}, {center_y})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, qr_codes, positions

def main():
    # Open the webcam (replace 0 with the camera index if multiple cameras are present)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to quit.")
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture a frame.")
            break
        
        # Detect and display QR codes and their positions
        frame, qr_codes, positions = detect_qr_codes(frame)
        
        # Print positions to the console
        for idx, pos in enumerate(positions):
            print(f"QR Code {idx + 1}: Center at {pos}")
        
        # Show the frame
        cv2.imshow("QR Code Position Identifier", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
