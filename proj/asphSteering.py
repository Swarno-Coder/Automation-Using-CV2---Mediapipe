import cv2,time
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
        #print(lmList[8])
        x , y = lmList[8][1], lmList[8][2]
        if x in range(270,370) and y in range(4,59):
            print('nitro')
        elif x <= 315 and y <= 130:
            print('left')
        elif x <= 635 and y <= 130:
            print('right')

    cv2.rectangle(img, (5,5), (315,130), (234,76,97), 3)
    cv2.rectangle(img, (325,5), (635,130), (234,76,97), 3)
    cv2.rectangle(img, (270,4), (370,59), (234,76,97), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    #print(int(fps))
    cv2.putText(img, f'FPS: {int(fps)}', (45, 420), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow('Steering', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()