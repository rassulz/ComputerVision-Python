import cv2
import numpy as np
width = 1280
height = 720
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
while True:
    frame = np.zeros([250,250,3],dtype=np.uint8)
    frame[0:,0:]=(0,0,250)
    frame[100:,0:100]=(150,0,50)
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam', 0,0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release()
