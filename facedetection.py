import cv2

face_cascade = cv2.CascadeClassifier(r'C:\Users\Admin\PycharmProjects\untitled\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
img = cv2.imread(r'E:\yash\mumbai pics\yash2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.25, 4)
for (x, y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 5)
    resized = cv2.resize(img, (int(img.shape[1] / 8), int(img.shape[1] / 8)))
cv2.line(resized, (135, 0), (135, 178), (0,255,0), 2)
cv2.line(resized, (135, 178), (135, 356), (0,0,255), 2)
cv2.line(resized, (135, 356), (135, 540), (0,255,0), 2)
cv2.line(resized, (270, 0), (270, 178), (0,255,0), 2)
cv2.line(resized, (270, 356), (270, 540), (0,255,0), 2)
cv2.line(resized, (405, 0), (405, 178), (0,255,0), 2)
cv2.line(resized, (405, 178), (405, 356), (0,0,255), 2)
cv2.line(resized, (405, 356), (405, 540), (0,255,0), 2)
cv2.line(resized, (540, 0), (540, 540), (0,255,0), 2)
################################
cv2.line(resized,(0,178),(135,178),(255,0,0),2)
cv2.line(resized,(135,178),(405,178),(0,0,255),2)
cv2.line(resized,(405,178),(540,178),(255,0,0),2)
cv2.line(resized,(0,356),(135,356),(255,0,0),2)
cv2.line(resized,(135,356),(405,356),(0,0,255),2)
cv2.line(resized,(405,356),(540,356),(255,0,0),2)
cv2.line(resized,(0,534),(712,534),(255,0,0),2)
cv2.line(resized,(0,712),(712,712 ),(255,0,0),2)
cv2.imshow('yash', resized)
print(img)
print(img.shape)
cv2.waitKey()
cv2.destroyAllWindows()

