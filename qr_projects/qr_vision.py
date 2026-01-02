import cv2
from pyzbar.pyzbar import decode

def detect_qr_codes(frame):
    # Decode QR codes in the frame
    qr_codes = decode(frame)
    
    for qr_code in qr_codes:
        # Extract bounding box and draw rectangle around QR code
        (x, y, w, h) = qr_code.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract QR code data
        qr_data = qr_code.data.decode('utf-8')
        qr_type = qr_code.type
        
        # Display the QR code data and type on the frame
        text = f"{qr_type}: {qr_data}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, qr_codes

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
        
        # Detect and display QR codes
        frame, qr_codes = detect_qr_codes(frame)
        
        # Show the frame
        cv2.imshow("QR Code Scanner", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
