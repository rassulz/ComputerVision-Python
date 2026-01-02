import cv2
width = 640
height = 360
myText = "Rassul is great at using of Ai and Ml"
myFront =cv2.FONT_HERSHEY_DUPLEX
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
while True:
    ignore, frame = cam.read()
    #frame[140:220,280:360] = (0,0,255)
    #cv2.rectangle(frame,(250,140),(390,220),(0,250,0),9)
    cv2.circle(frame,(320,180),25,(0,0,0),4)
    cv2.putText(frame,myText,(60,60),myFront,0.6,(250,0,0),2)
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam', 0,0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release()