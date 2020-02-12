import cv2

face_cascade = cv2.CascadeClassifier(r'C:\Users\Admin\PycharmProjects\untitled\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
img = cv2.imread(r'E:\yash\mumbai pics\yash2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.25, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 8)
resized = cv2.resize(img, (int(img.shape[1] / 8), int(img.shape[1] / 8)))
cv2.imshow('yash', resized)
cv2.waitKey(1000)
cv2.destroyAllWindows()

