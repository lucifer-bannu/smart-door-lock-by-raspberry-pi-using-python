# ----------------------------------------- #
# This code will recognize the user's face  #
# and open the door/lock.                   #
# ----------------------------------------- #

# Open CV is a library for image processing.
import cv2
# OS import is used to access any os instance.(accessing files, directories, etc.)
import os
# GPIO is used to control the door lock.
# Emulation is used to emulate the door lock.
from GPIOEmulator.EmulatorGUI import GPIO
# This is the class that will be used to recognize the user's face.
import time

# Relay is used for the door lock relay in Emulator.
relay = 23
# GPIO configurations
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay, GPIO.OUT)
GPIO.output(relay, 1)
# Loading the face cascade classifier.
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Reading the trained data from the file.
recognizer.read('trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
# Loading the cascade classifier.
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
names = []
# get the length of the names list
length = len(os.listdir('dataset'))
# loop through the names list and append the names to the names list
# If there are any empty datasets, The code will break.
for i in range(length):
    name = os.listdir('dataset/' + str(i))[0].split('.')[0]
    names.append(name)

# Loop till the user presses the Esc key.
while True:
    # Read the video frame
    ret, img = cam.read()
    if not ret:
        continue
    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    # For each face in faces
    for(x, y, w, h) in faces:
        # Create rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Predict the image using our face recognizer
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            # Check if the name is in the names list
            id = names[id]
            # Put text describe who is in the picture with percentage
            confRound = round(100-confidence)
            if confRound > 40:
                confidence = "==> {0}%".format(confRound)
                # Open the door if the confidence is greater than 40.(Just for testing)
                # If we want to increase the confidence threshold, we can do it here.
                # But we need to increase the threshold in the trainer.py file.
                # That means we need to increase data set size.
                GPIO.output(relay, 0)
                print(str(confidence) + " sure user ID is " + str(id))
                cv2.putText(img, str(id), (x+5, y-5),
                            font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confRound), (x+5, y+h-5),
                            font, 1, (255, 255, 0), 1)
                # Hold the door open for 2 seconds.
                time.sleep(2)
            GPIO.output(relay, 1)
        # Handle if the confidence is less than 30
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            GPIO.output(relay, 1)
            cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x+5, y+h-5),
                        font, 1, (255, 255, 0), 1)
    # Display an image in the specified window
    cv2.imshow('camera', img)
    # Wait for the user to press some key
    k = cv2.waitKey(10) & 0xff
    # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
