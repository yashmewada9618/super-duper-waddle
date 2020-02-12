import time

import cv2
import numpy as np
import urllib.request

face_cascade = cv2.CascadeClassifier(
    r'C:\Users\Admin\PycharmProjects\untitled\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

# Replace the URL with your own IPwebcam shot.jpg IP:port
url = 'http://100.127.41.131:8080/shot.jpg'

while True:
    # Use urllib to get the image from the IP camera
    imgResp = urllib.request.urlopen(url)

    # Numpy to convert into a array
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

    # Finally decode the array to OpenCV usable format ;) 
    img = cv2.imdecode(imgNp, -1)

    image = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[1] / 4)))
    print(image.shape)
    # put the image on screen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.25, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow('IPWebcam', image)
    # To give the processor some less stress
    time.sleep(0.0000000000001)

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
