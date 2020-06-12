#Gerekli paketi içe aktarıyoruz.
import cv2
import numpy as np 
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
detector=cv2.CascadeClassifier(path+r'\Classifiers\face.xml')

camera = cv2.VideoCapture(0)
camera.set(3,640)
camera.set(4,480)

minW = 0.1*camera.get(3)
minH = 0.1*camera.get(4)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name = input("AD/SOYAD? ")
dirName = "./images/" + name
print(dirName)
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Klasör oluştruldu")
else:
    print("İsim önceden kullanılmış")
    sys.exit()

count = 1
while True:
    
    if count > 30:
        break
    ret, im =camera.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x, y, w, h) in faces:
        roiGray = gray[y:y+h, x:x+w]
        fileName = dirName + "/" + name + str(count) + ".jpg"
        cv2.imwrite(fileName, roiGray)
        cv2.imshow("face", roiGray)
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        count += 1

    cv2.imshow('im', im)
    key = cv2.waitKey(10)
   

    if key == 27:
        break

cv2.destroyAllWindows()
