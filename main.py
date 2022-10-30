import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
fpsReader = cvzone.FPS()
segmentor = SelfiSegmentation()
listImg = os.listdir('Images')
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'Images/{imgPath}')
    imgList.append(img)
#print(imgList)

index = 0

def removeBG(self, img, imgBg=(255, 255, 255), threshold=0.1):
    """

    :param img: image to remove background from
    :param imgBg: BackGround Image
    :param threshold: higher = more cut, lower = less cut
    :return:
    """
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp.solutions.selfie_segmentation.SelfieSegmentation(0).process(imgRGB)
    condition = np.stack(
        (results.segmentation_mask,) * 3, axis=-1) > threshold
    print(imgBg)
    print(condition)
    if isinstance(imgBg, tuple):
        _imgBg = np.zeros(img.shape, dtype=np.uint8)
        _imgBg[:] = imgBg
        imgOut = np.where(condition, img, _imgBg)
    else:
        imgOut = np.where(condition, img, imgBg)
    return imgOut

while True:
    success, img = cap.read()
    imgout = segmentor.removeBG(img, imgBg=imgList[index], threshold=0.88)
    imgStacked = cvzone.stackImages([img, imgout], 2, 1)
    _, imgStacked = fpsReader.update(imgStacked, color= (0,0,255))
    print(index)
    cv2.imshow('Stacked Image', imgStacked)

    cp = cv2.waitKey(1)
    if cp & 0xFF == ord('q'):
        break
    elif cp == ord('a'):
        if index>0:
            index -= 1

    elif cp == ord('d'):
        if index<len(imgList) - 1:
            index += 1

