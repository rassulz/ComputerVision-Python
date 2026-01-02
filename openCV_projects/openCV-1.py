import cv2
print(cv2.__version__)
cam = cv2.VideoCapture(0)
while True:
    ignore, frame = cam.read()
    grayFrame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    greenFrame=cv2.cvtColor(frame,cv2.IMREAD_GRAYSCALE)
    teploFrame=cv2.cvtColor(frame,cv2.COLORMAP_JET)
    cv2.resizeWindow('my WEBcam', 20, 50)
    cv2.imshow('my WEBcam', greenFrame)
    cv2.moveWindow('my WEBcam',0,0)
    # Change 'waitkey' to 'waitKey' (capital 'K')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cam.release()
cv2.destroyAllWindows()  # Optional: close all OpenCV windows
