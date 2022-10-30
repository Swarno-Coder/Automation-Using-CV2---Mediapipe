import cv2
import mediapipe as mp
import time

class faceMesh():
    def __init__(self, mode=False, maxFaces=1, refLand=False, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.refLand = refLand
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxFaces, self.refLand, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaces(self, img, draw=True):
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                      self.drawSpec, self.drawSpec)
        return img

    def findLms(self, img, faceNo=0, draw=True):
        lmList = []
        if self.results.multi_face_landmarks:
            myFace = self.results.multi_face_landmarks[faceNo]
            for id, lm in enumerate(myFace.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                #cv2.circle(img, (cx, cy), 10, (255, 0, 219), cv2.FILLED)
        return lmList


def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(1)
    detector = faceMesh()
    while True:
        success, img = cap.read()
        img = detector.findFaces(img)
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