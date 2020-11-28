import os
import cv2
from __config__ import *

crop_height = 130
path = 'sanphai/images2/thang_lanephai/'
dst_dir = 'sanphai/images2_{}/thang_lanephai/'.format(crop_height)
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
    
for imgname in os.listdir(path):
    imgpath = path+imgname
    # print('imgpath', imgpath)

    img = cv2.imread(imgpath)
    img = img[240-crop_height:, :]
    cv2.imwrite(dst_dir+imgname, img)