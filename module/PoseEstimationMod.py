import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, modelCom=1,
                 smothLand=True, enbleSeg=False, smothSeg=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.modelComp = modelCom
        self.smoothLand = smothLand
        self.enableSeg = enbleSeg
        self.smoothSeg = smothSeg
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComp,
                                      self.smoothLand, self.enableSeg,
                                      self.smoothSeg,
                                      self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        self.poselm = self.results.pose_landmarks
        # print(results.multi_hand_landmarks)
        if self.poselm:
            if draw:
                self.mpDraw.draw_landmarks(img, self.poselm, self.mpPose.POSE_CONNECTIONS,
                                               connection_drawing_spec=
                                               self.mpDraw.DrawingSpec((233,43,5),thickness=3))
        return img

    def findLms(self, img):
        lmList = []
        if self.poselm:
            for id, lm in enumerate(self.poselm.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList

def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(1)
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findPose(img)
        lmList = detector.findLms(img)
        print(lmList)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)
        cv2.imshow("Result", img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    main()