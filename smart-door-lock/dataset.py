# ------------------------------------- #
# This code will capture the dataset    #
# (face images) from the webcam and     #
# save it in a folder. The dataset will #
# be used for training the model.       #
# ------------------------------------- #

import cv2
import os

# Available at https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
user_name = input('\n enter user name and press <return> ==>  ')
# Initialize individual sampling face count
count = 0
# Check if the dataset folder exists in current working directory. If not create it.
if not os.path.exists('dataset'):
    os.makedirs('dataset')
# Get the length of the folder
length = len(os.listdir('dataset'))
id = length
# create a folder with length of the dataset folder
os.makedirs('dataset/' + str(id))
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize the camera
cam = cv2.VideoCapture(0)

while(True):
    ret, img = cam.read()
    if not ret:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        # Save the captured image into the datasets folder with the name of the user id
        cv2.imwrite("dataset/" + str(id)+"/" + str(user_name) + '.' +
                    str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 45:
        break
    elif count >= 100:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
userIdLen = len(os.listdir('dataset/' + str(id)))
if userIdLen < 50:
    os.rmdir('dataset/' + str(id))
    print("\n [INFO] User not registered. Try again.")
cam.release()
cv2.destroyAllWindows()
