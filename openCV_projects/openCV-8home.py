import cv2
print(cv2.__version__)
width = 200
height = 200
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
while True:
    ignore, frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam',0,0)

    cv2.imshow('my WEBRU', gray)
    cv2.moveWindow('my WEBRU', width, 0)

    cv2.imshow('my WEBcam2', frame)
    cv2.moveWindow('my WEBcam2',width,height)

    cv2.imshow('my WEBRU2', gray)
    cv2.moveWindow('my WEBRU2', 0, height)

    if cv2.waitKey(1) %0xff == ord('q'):
        break
cam.release()


