import numpy as np
import cv2

# download from https://raw.githubusercontent.com/kipr/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set up Video Camera
# 0 is the default camera.
cap = cv2.VideoCapture(0)

while True:
    # read picture from camera
    ret, img = cap.read()
    # convert color into gray.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect face in the gray image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # draw face rectangle on the original image
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # show image in the window
    cv2.imshow('Escape Key will close this window', img)
    key = cv2.waitKey()
    if key == 27: # break if escape key
        break

cv2.destroyAllWindows()


"""
ref:
https://ja.stackoverflow.com/questions/31537/python-%E3%81%A7-cv2-imshow-%E3%81%A8%E3%81%97%E3%81%A6%E3%82%82%E7%94%BB%E5%83%8F%E3%81%8C%E8%A1%A8%E7%A4%BA%E3%81%95%E3%82%8C%E3%81%AA%E3%81%84
https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
https://raw.githubusercontent.com/kipr/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
https://www.klv.co.jp/corner/python-opencv-video-capture.html
https://kuroro.blog/python/8DIolh7Pwggq2pvabysn/

"""
