import numpy as np
import cv2
from glob import glob as gl
from tqdm import tnrange
import os
from __config__ import *


images = []
path = san+'/images/'+part+'/'
temp = []
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            # temp.append(int(file.split('.')[0].split('__')[-1]))
            temp.append(file.split('.')[0])
for x in sorted(temp):
    images.append(path + str(x) + '.jpg')
# print(images)
# print(len(images))
img2 = None
img1 = None
img = None
is_blue = 0
is_green = False
blue_x = 0
blue_y = 0
distance = 210


def printImage(image, name=None):
    if name is None:
        cv2.imshow('ok', image)
    else:
        cv2.imshow(name, image)
    cv2.waitKey(0)


def line_funtion(point1, point2):
    a = (point2[1] - point1[1]) / (point2[0] - point1[0])
    b = point1[1] - a * point1[0]
    return a, b


def point_caculator(a, b, y):
    return int((y - b) / a)


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY, img1, img2, is_blue, blue_x, blue_y, is_green
    m_x = x
    m_y = y
    img0 = img.copy()
    if event == cv2.EVENT_LBUTTONDBLCLK:  # chi save thang xanh la cay thoai
        is_green = True
        cv2.circle(img, (x, line1), 10, (0, 255, 0), -1)
        cv2.circle(img0, (x, line1), 10, (0, 255, 0), -1)
        cv2.circle(img2, (x, line1), 10, (0, 255, 0), -1)
    N = 295
    pts1 = np.float32([[0, 50], [320, 50], [0, 160], [320, 160]])
    pts2 = np.float32([[0, 0], [960, 0], [N, 160], [960 - N, 160]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img0, matrix, (960, 160))
    cv2.imshow('image', img0)
    cv2.imshow('result', result)


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
i = 0
side = -1
# ID = 804
while True:
    is_blue = 0
    is_green = False
    side = -1 * side
    img = cv2.imread(images[i], 1)
    # img = cv2.resize(img, (WIDTH, HEIGHT))
    img = img[80:, :]
    # if side== 1:
    #     cv2.rectangle(img,(0,0),(80,80),(0,0,255),-1)
    # elif side == -1:
    #     cv2.rectangle(img,(WIDTH-80,0),(WIDTH, 80),(0,0,255),-1)
    # elif side == 0:
    #     cv2.rectangle(img,(WIDTH//2-40,0),(WIDTH//2+40, 80),(0,0,255),-1)
    # elif side == -2:
    #     cv2.rectangle(img,(0,0),(80,80),(0,0,255),-1)
    #     cv2.rectangle(img,(WIDTH//2-40,0),(WIDTH//2+40, 80),(0,0,255),-1)
    # else:
    #     cv2.rectangle(img,(WIDTH-80,0),(WIDTH, 80),(0,0,255),-1)
    #     cv2.rectangle(img,(WIDTH//2-40,0),(WIDTH//2+40, 80),(0,0,255),-1)

    name = images[i].split('/')[-1]

    img0 = img.copy()
    img1 = img.copy()
    img2 = img.copy()
    # name = str(ID) + '.jpg'

    img[line1 - 5: line1 + 5, :, :] = 0
    cv2.circle(img, (WIDTH // 2, line1), 5, (0, 255, 255), -1)
    # cv2.circle(img, (WIDTH//2-60,line1), 5, (0, 255, 255), -1)
    # cv2.circle(img, (WIDTH//2+60,line1), 5, (0, 255, 255), -1)
    cv2.imshow('image', img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        break
    if k == ord('r'):
        side += 1
        if side == 3:
            side = -2
    if k == ord('a'):
        i -= 1
    if k == ord('d'):
        i = i + 1
        print('over')
    if k == ord('s'):
        i = i + 1
        # cv2.imwrite(san+'/images1/'+part+'/' + name, img1)
        cv2.imwrite(san+'/images2/'+part+'/' + name, img2)
        print('save: ' + name)
        # ID = ID + 1
cv2.destroyAllWindows()
