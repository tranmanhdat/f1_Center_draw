import numpy as np
import cv2
from glob import glob as gl
import os
import shutil
from __config__ import *
import json

images = gl(path_labeled+'/*/*.jpg')

def get_point(img, color):
    global line1
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


def get_label(image, folder_name):
    global line1
    img = cv2.imread(image, 1)
    # img = cv2.resize(img, (WIDTH, HEIGHT))
    correct = False
    center = [WIDTH//2, 0]

    # Check lane center
    # blue point
    processArea = img[line1-15: line1+15, :, :]
    # printImage(processArea,'process area')
    center, correct = get_point(processArea, 'green')
    if not correct:
        center, correct = get_point(processArea, 'blue')
    cv2.circle(img, (center[0], line1), 5, (0, 0, 255), -1)
    
    name = image.split('/')[-1]

    if draw_guide_append:
        draw_dir = 'images_draw_more'
        crop_top = 280-(HEIGHT+40)
    else:
        draw_dir = 'images_draw'
        crop_top = 240-HEIGHT

    raw_img_path = os.path.join(san, draw_dir, folder_name, name)
    # print('raw_img_path', raw_img_path)
    if os.path.exists(raw_img_path):
        # print('Read data from ', raw_img_path)
        image = cv2.imread(raw_img_path, 1)
        image = image[crop_top:, :]
        # print(crop_top, image.shape)
        # cv2.imshow('image', image)
        # cv2.imshow('img', img)
        return image, center[0]*1.0/WIDTH, correct
    
    return None, None, None



X = []
Y = []
txt = {
    'sanphai': {
        'nuadau': {},
        'phai_lanephai': {},
        'thang_lanephai': {},
        'thang_lanetrai': {}
    },
    'santrai': {
        'nuadau': {},
        'thang_lanephai': {},
        'trai_lanephai': {},
        'trai_lanetrai': {}
    }
}
not_correct = 0

for fname in os.listdir(path_labeled): # fname = [route]_[lane] (phai_lanephai, thang_lanephai,...)
    if '_' in fname:
        route = fname.split('_')[0]
        lanepath = fname.split('_')[1]
    else: # nua dau
        route = 'nuadau'
        lanepath = 'nuadau'

    fpath = path_labeled+fname
    for imgname in os.listdir(fpath):
        imgpath = fpath+'/'+imgname
        image, center, correct = get_label(imgpath, fname)

        if correct is not None:
            if correct:
                # print('image.shape', image.shape)
                X.append(image)
                Y.append([center])

                txt[san][fname][imgpath] = center

                # X.append(image[:, ::-1,:]) # soi guong cai anh ma thoi
                # Y.append([1- center])
            else:
                not_correct += 1
                print('image :', imgpath, 'is not progressed!', not_correct)
                # shutil.move(imgpath, path_move_labeled_not_progressed+imgpath.split('/')[-1])

        # k = cv2.waitKey(1) & 0xFF
        # if k == 27:
        #     break

with open(center_save_path, 'w+') as f:
    json.dump(txt, f)

print(len(X))
X = np.array(X)
Y = np.array(Y)
print('X', X.shape)
np.save(data_path_x, X)
np.save(data_path_y, Y)
print('Save X to', data_path_x)
print('Save Y to', data_path_y)
print('Save center to', center_save_path)


cv2.destroyAllWindows()
