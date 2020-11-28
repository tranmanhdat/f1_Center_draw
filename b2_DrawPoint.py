#! /usr/bin/python3
import numpy as np
import cv2
from glob import glob as gl
from tqdm import tnrange
import os
from __config__ import *
import json

# with open('data/center_320x{}__{}+40.json'.format(HEIGHT+40, HEIGHT), 'r') as f:
#     center_data = json.load(f)

images = []
path = path_raw_resized
temp = []
tem = {}
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            if '__' in file:
                tem[path+file] = int(file.split('.')[0].split('__')[1]) + int(file.split('/')[-1].split('__')[0][-1])*10000
            else:
                tem[path+file] = int(file.split('.')[0])
            # temp.append(file.split('.')[0])

# print(sorted(tem.items()))
# tem = sorted(tem.items(), key=lambda kv: kv[1])
tem = {v: k for k, v in sorted(tem.items(), key=lambda item: item[1])}

for x in tem:
    # print('x', x)
    images.append(tem[x])
# print(tem)
# print(len(images))
img2 = None
img0 = None
img = None
is_blue = 0
is_green = False
blue_x = 0
blue_y = 0
distance = 210


def printImage(img,name=None):
    if name is None:
        cv2.imshow("ok",img)
    else:
        cv2.imshow(name,img)
    cv2.waitKey(0)
    
def line_funtion(point1, point2):
    a = (point2[1] - point1[1])/(point2[0] - point1[0])
    b = point1[1]- a*point1[0]
    return a, b

def point_caculator(a,b, y):
    return int((y-b)/a)
def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, pxstep=40, pystep=40):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pxstep+5
    while x < img.shape[1]:
        if x==WIDTH/2 :
            cv2.line(img, (x, 0), (x, img.shape[0]), color=(0,0,255), lineType=type_, thickness=thickness)
        else :
            cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep
    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY, img0, img2, is_blue, blue_x, blue_y, is_green
    m_x = x
    m_y = y
    # img0 = img.copy()
    if event == cv2.EVENT_LBUTTONDBLCLK: ## chi save thang xanh la cay thoai
        is_green = True
        cv2.circle(img, (x,line1), 5, (0, 255, 0), -1)
        cv2.circle(img0, (x,line1), 5, (0, 255, 0), -1)
        cv2.circle(img2, (x,line1), 5, (0, 255, 0), -1)
    N=295
    # pts1 = np.float32([[0, 160], [480, 160], [0, 320], [480, 320]])
    # pts2 = np.float32([[0, 0], [960, 0], [N, 320], [960 - N, 320]])
    pts1 = np.float32([[0, 50], [320, 50], [0,160], [320, 160]])
    pts2 = np.float32([[0, 0], [960, 0], [N, 160], [960-N, 160]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img0[80:,:], matrix, (960, 160))
    cv2.imshow('image', img0)
    cv2.imshow('result', result)


def get_point(image, color):
    global line1

    img = cv2.imread(image, 1)

    lower_green = np.array([50, 230, 230])
    upper_green = np.array([70, 255, 255])
    lower_blue = np.array([110, 220, 180])
    upper_blue = np.array([130, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    if color == 'blue':
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    point = []
    center = [WIDTH//2, 0]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10:
            x, y, w, h = cv2.boundingRect(cnt)
            center = [int(x+w/2), int(y+h/2)]
            point.append(center)

    # cv2.imshow('mask', mask)

    if color == 'green':
        if len(point) == 1:
            return point[0], True
        else:
            return [WIDTH//2, 0], False
    if color == 'blue':
        if len(point) == 2:
            return np.mean(point, axis=0).astype(np.uint16), True
        else:
            return [WIDTH//2, 0], False


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

i = 694
side = -1

# ID = 804
while True:
    print(images[i])
    is_blue = 0
    is_green = False
    side = -1 * side
    img = cv2.imread(images[i], 1)
    # img = cv2.resize(img, (WIDTH, HEIGHT))
    h,w = img.shape[:2]
    # img = img[h-HEIGHT:, :]

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
    img2 = img.copy()
    # name = str(ID) + '.jpg'

    # draw_grid(img, (0, 255, 0), 1, cv2.LINE_AA, 40, 40)
    draw_grid(img0, (0, 255, 0), 1, cv2.LINE_AA, 40, 40)

    img0[line1 - 5: line1 + 5, :, :] = 0
    cv2.circle(img0, (WIDTH // 2, line1), 5, (0, 255, 255), -1)


    imgname = images[i].split('/')[-1]
    fname = images[i].split('/'+imgname)[0].split('/')[-1]
    # print('fname', fname)
    # print('imgname', imgname)
    # key = san+'/images2/'+fname+'/'+imgname
    # if key in center_data[san][fname]:
    #     center = center_data[san][fname][key]
    #     x = int(center * WIDTH)
    
    labeled_path = san+'/images2/'+fname+'/'+imgname
    print('\t labeled_path', labeled_path)
    if os.path.exists(labeled_path):
        center, correct = get_point(labeled_path, 'green')
        print('\t center, correct', center, correct)
        if correct:
            cv2.circle(img0, (center[0],line1), 5, (255, 0, 0), -1)


    cv2.imshow('image', img0)
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
        cv2.imwrite(san+'/images2/'+part+'/' + name, img2)
        print('\t save: ' + san+'/images2/'+part+'/' + name)
        # ID = ID + 1
cv2.destroyAllWindows()
