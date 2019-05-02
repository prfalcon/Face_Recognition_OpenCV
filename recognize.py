import os
import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0
curid = 0
curCount = 0
switchId = False
difCount = 0
lowCount = 0
userFound = False
userId = 0
confidence = "100%"
fullCount = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['Unknown','Vishnu','Demo User', 'Satya']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    #img = cv2.flip(img, -1)  # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    id, confidence = '', '0%'
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.putText(img, "Confidence: " + str(curCount), (50, 50), font, 1, (255, 255, 0), 1)
    if userFound:
        if userId == 'unknown':
            userId = 'Unknown user. Please (re)train face.'
        if userId == '':
            userId = 'No user found'
        cv2.putText(img, "User ID: " + userId, (20, 20), font, 1, (255, 255, 0), 1)
        cv2.imshow('camera', img)
        print (userId)
        break


    cv2.imshow('camera', img)

    fullCount += 1
    curCount += 1
    if (int(confidence[:len(confidence) - 1]) > 75):
        lowCount += 1
    if lowCount == 5:
        curCount = 0
        lowCount = 0
        difCount = 0
        print('Low')
        print(id)

    if id != curid:
        difCount += 1
    if difCount == 5:
        curid = id
        difCount = 0
        curCount = 0
        lowCount = 0
        print("dif")

    if not userFound and curCount >= 50:
        if curCount >= 100:
            userFound = True
            userId = 'User not found.'
        elif curid not in ['unknown', ''] or curCount >=75:
            userFound = True
            userId = curid

    if fullCount > 200:
        userFound = True
        userId = ''


    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n Recognizer closed succesfully")
while True:
    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        cam.release()
        cv2.destroyAllWindows()
        break