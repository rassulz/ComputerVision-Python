import cv2
import time

# Настройки камеры
width = 640
height = 360
myText = "Rassul is great at using AI and ML"
myFont = cv2.FONT_HERSHEY_DUPLEX
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Переменные для расчета FPS
prev_time = time.time()
fps = 0

while True:
    # Считывание кадра с камеры
    ignore, frame = cam.read()

    # Расчет FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Отображение FPS в правом верхнем углу
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (10, 30), myFont, 0.8, (255, 0, 0), 2)
    
    # Показ видео с текстом
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam', 0, 0)

    # Условие выхода при нажатии на 'q'
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Освобождение ресурсов
cam.release()
cv2.destroyAllWindows()
