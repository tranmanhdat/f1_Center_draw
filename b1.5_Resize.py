import os
import shutil
from __config__ import *
import cv2

for imgname in os.listdir(path_raw_origin):
    imgpath = os.path.join(path_raw_origin, imgname)
    dst = os.path.join(path_raw_resized, imgname)

    img = cv2.imread(imgpath)
    img = cv2.resize(img, (320,240))
        
    print('Save resized to ', dst)
    cv2.imwrite(dst, img)