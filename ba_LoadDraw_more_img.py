#! /usr/bin/python3
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import cv2
import numpy as np
from tensorflow.python.keras.models import load_model

from glob import glob as gl
from tqdm import tnrange
import os
from __config__ import *
import json

import tensorflow as tf
# Fixed error Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


model = load_model('models/'+model_name+'.h5')
model.summary()


with open('{}/center.json'.format(data_dir), 'r') as f:
    center_data = json.load(f)


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

tem = {v: k for k, v in sorted(tem.items(), key=lambda item: item[1])}

for x in tem:
    images.append(tem[x])

img2 = None
img0 = None
img = None
is_blue = 0
is_green = False
blue_x = 0
blue_y = 0
distance = 210

i = 200
side = -1

while True:
    print(images[i])
    is_blue = 0
    is_green = False
    side = -1 * side
    img_raw = cv2.imread(images[i], 1)

    name = images[i].split('/')[-1]

    
    # Draw label
    imgname = images[i].split('/')[-1]
    fname = images[i].split('/'+imgname)[0].split('/')[-1]
    # print('fname', fname)
    # print('imgname', imgname)
    key = san+'/images2/'+fname+'/'+imgname
    label = None
    if key in center_data[san][fname]:
        label = center_data[san][fname][key]
        x = int(label * WIDTH)
        # print('x=',x)

        cv2.circle(img_raw, (x,line1), 5, (255, 0, 0), -1)



    # Predict
    fh, fw = img_raw.shape[:2]
    img_crop = img_raw[fh-HEIGHT:, :]

    img_more = np.zeros(shape=[40, 320, 3], dtype=np.uint8)
    if '_' in fname:
        route = fname.split('_')[0]
        bamlane = fname.split('_')[1]
    
        if route == 'thang':
            cv2.circle(img_more, org_thang, 20, (0,0,255), -1)
        elif route == 'phai':
            cv2.circle(img_more, org_phai, 20, (0,0,255), -1)
        elif route == 'trai':
            cv2.circle(img_more, org_phai, 20, (0,0,255), -1)
        
        if bamlane == 'lanephai':
            cv2.rectangle(img_more, pts_lanephai[0], pts_lanephai[1], (255,0,0), -1)
        elif bamlane == 'lanetrai':
            cv2.rectangle(img_more, pts_lanetrai[0], pts_lanetrai[1], (255,0,0), -1)

    img = cv2.vconcat([img_crop, img_more])

    predict = model.predict(np.array([img])/255.0)[0]
    print(predict, label)
    center_predict = int(predict[0]*WIDTH)


    cv2.circle(img_raw, (center_predict, line1), 5, (0, 255, 0), -1)
    img_raw = img_raw[fh-HEIGHT:, :]

    cv2.imshow('img_raw', img_raw)
    cv2.imshow('img', img)


    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        break
    if k == ord('a'):
        i -= 1
    if k == ord('d'):
        i = i + 1
        print('over')

cv2.destroyAllWindows()
