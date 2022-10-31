import cv2
import time
from module import HandTrackinMod as htm

cTime = 0
pTime = 0
cap = cv2.VideoCapture(1)
detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findLms(img)
    if len(lmList) != 0:
        print(lmList[4])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Result", img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()