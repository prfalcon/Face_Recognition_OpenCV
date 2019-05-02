import os
from PIL import Image
import cv2
import numpy as np

path = 'dataset'

#recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer = cv2.face.FisherFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(cv2.resize(img_numpy[y:y+h,x:x+w], (175, 175)))
            ids.append(id)
    return faceSamples,ids

print ("\n Training faces. It will take a few seconds. Please wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() workes on pc, but not on Pi

# Print the numer of faces trained and end program
print("\n {0} faces trained. Exiting Program".format(len(np.unique(ids))))