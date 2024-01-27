import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = np.load("faces.npy")

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

X = data[:, 1:].astype(int)
y = data[:, 0]

model = KNeighborsClassifier(4)
model.fit(X, y)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    faces = classifier.detectMultiScale(gray)
    areas = []
    for face in faces:
        x, y, w, h = face
        area = w*h
        areas.append((area, face))
    areas = sorted(areas, reverse = True)
    if len(areas) > 0:
        face = areas[0][1]
        x, y, w, h = face
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))

        flat = face_img.flatten()
        res = model.predict([flat])
        if len(res)>0:
            print(res)
        else:
            print("No Match!")
    else:
        print("No Face Detected")
        # print(res)
    for x, y, w, h in faces:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, res[0], (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("video", frame)
    if cv2.waitKey(1) > 30:
        break


cap.release()
cv2.destroyAllWindows()

