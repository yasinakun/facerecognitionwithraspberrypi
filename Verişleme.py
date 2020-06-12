#Bu kod trainer.yml ve labels dosyalarını oluşturur
#Bunlar giriş kodunda kullanılarak kıyaslamayı mümkün kılar.
import os
import numpy as np 
from PIL import Image 
import cv2
import pickle

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# OpenCV paketinde bulunan LBPH (YEREL İKİLİ PATTERNS HISTOGRAMS) yüz tanıyıcı kullanıyoruz.
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Geçerli çalışma dizininin yolunu alınır ve images dizinine gidilir.

baseDir = os.path.dirname(os.path.abspath(__file__))
imageDir = os.path.join(baseDir, "images")

currentId = 1
labelIds = {}
yLabels = []
xTrain = []

#Her görüntü dizinine gider ve görüntüleri arar.
#Görüntü mevcutsa, NumPy dizisine dönüştürüyoruz
for root, dirs, files in os.walk(imageDir):
    print(root, dirs, files)
    for file in files:
        print(file)
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            print(label)

            if not label in labelIds:
                labelIds[label] = currentId
                print(labelIds)
                currentId += 1

#Doğru görüntülere sahip olduğumuzdan emin olmak için yüz algılamayı tekrar gerçekleştiriyoruz.
#Ve sonra kıyaslama verilerini hazırlıyoruz
            id_ = labelIds[label]
            pilImage = Image.open(path).convert("L")
            imageArray = np.array(pilImage, "uint8")
            faces = faceCascade.detectMultiScale(imageArray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = imageArray[y:y+h, x:x+w]
                xTrain.append(roi)
                yLabels.append(id_)
# Dizin adlarını ve etiket kimliklerini içeren sözlüğü saklayoruz.
with open("labels", "wb") as f:
    pickle.dump(labelIds, f)
    f.close()
# Verileri eğiterek ve dosyayı kaydediyoruz.
recognizer.train(xTrain, np.array(yLabels))
recognizer.save("trainer.yml")
print(labelIds)