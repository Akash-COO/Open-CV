import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None

    for (x, y, w, h) in faces:
        cropped_faces = img[y:y + h, x:x + w]

    return cropped_faces


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        cv2.imshow('face samples', face)

        file_name_path = 'F:/FACE_REC/phoenix/samples/user' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

    else:
        print("FACE NOT FOUND")
        pass

    if cv2.waitKey(1) == 13 or count == 10:
        break

cap.release()
cv2.destroyAllWindows()
print('Samples Collected')