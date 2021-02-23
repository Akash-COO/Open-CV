import cv2
import numpy as np
from dlib import *
import face_recognition
import os
from datetime import datetime

#path were images are stored
path = 'samples'

#for labeling
images = []
image_names = []

#list containg all images
mylist = os.listdir(path)
print(mylist)

#image name = i_n iterative element
for i_n in mylist:
     current_image = cv2.imread(f'{path}/{i_n}')
     images.append(current_image)
     image_names.append(os.path.splitext(i_n)[0])

print(image_names)

def find_encodings(images):
    encode_list=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_enc = face_recognition.face_encodings(img)[0]
        encode_list.append(img_enc)
    return encode_list

def count_people(name):
    with open('count_people.csv','r+') as f:
        my_data_list = f.readlines()
        name_people = []
        for line in my_data_list:
            entry = line.split(',')
            name_people.append(entry[0])
        if name not in name_people:
            now =  datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

#testing encoding for known faces
encode_list_known = find_encodings(images)
#print(len(encode_list_known))
print('Encoding step completed')

#starting webcam
cap = cv2.VideoCapture(0)

while True:
    succes, image = cap.read()
    image = cv2.resize(image, (0,0), None, 0.25, 0.25)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces_in_current_frame = face_recognition.face_locations(image)
    encodings_of_current_frame = face_recognition.face_encodings(image, faces_in_current_frame)

    for encode_face, face_loc in zip(encodings_of_current_frame, faces_in_current_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_distance = face_recognition.face_distance(encode_list_known, encode_face)
        print(face_distance)
        matchIndex = np.argmin(face_distance)

        if matches[matchIndex]:
            name = image_names[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(image, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(image, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            count_people(name)

    cv2.imshow('Webcam',image)
    cv2.waitKey(1)