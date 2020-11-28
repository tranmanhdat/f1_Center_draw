import os
import json
import cv2
from __config__ import *


with open('data/center_160.json'.format(HEIGHT), 'r') as f:
    center_data = json.load(f)

path_labeled_old = '/media/fitmta/Storage/CDS/f1_Center_draw/sanphai/images2_160/'
for fname in os.listdir(path_labeled_old): # fname = [route]_[lane] (phai_lanephai, thang_lanephai,...)
    if '_' in fname:
        route = fname.split('_')[0]
        lanepath = fname.split('_')[1]
    else: # nua dau
        route = 'nuadau'
        lanepath = 'nuadau'

    fpath = path_labeled_old+fname
    for imgname in os.listdir(fpath):
        imgpath = fpath+'/'+imgname
        rawpath = '/media/fitmta/Storage/CDS/f1_Center_draw/sanphai/images_resized/'+fname+'/'+imgname

        img = cv2.imread(rawpath)

        key = 'sanphai/images2/'+fname+'/'+imgname
        center = center_data['sanphai'][fname][key]

        x = int(center * WIDTH)
        cv2.circle(img, (x,line1), 5, (0, 255, 0), -1)

        cv2.imwrite(path_labeled+fname+'/'+imgname, img)