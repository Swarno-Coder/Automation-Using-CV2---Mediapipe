import cv2
import time, numpy as np
import module.HandTrackinMod as htm

##########################
wCam , hCam = 640, 480
##########################

cTime = pTime = 0

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1, detectionCon=0.8, trackCon=0.8)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findLms(img)
    if len(lmList) != 0:
        x, y = lmList[8][1], lmList[8][2]
        print()
        