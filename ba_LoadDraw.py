import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import cv2
import numpy as np
from tensorflow.python.keras.models import load_model
import sys
import argparse
from __config__ import *
from tensorflow.python.keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_video")
args = parser.parse_args()


bamlane = 'phai'
route = 'thang'


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

cap = cv2.VideoCapture(args.input_video)

side = 0
t = 15
while True:
    ret, frame = cap.read()
    # if side==0:
    #     cv2.rectangle(frame,(0,0),(80,80),(0,0,255),-1)
    # else:
    #     cv2.rectangle(frame,(WIDTH-80,0),(WIDTH, 80),(0,0,255),-1)
    # if not ret:
    #     break
    # img = cv2.resize(frame, (240, 160))
    img_raw = cv2.resize(frame, (320, 240))
    fh, fw = img_raw.shape[:2]
    img = img_raw[fh-HEIGHT:, :]

    if route is not None:
        if route == 'thang':
            cv2.circle(img, org_thang, 20, (0,0,255), -1)
        elif route == 'phai':
            cv2.circle(img, org_phai, 20, (0,0,255), -1)
        elif route == 'trai':
            cv2.circle(img, org_phai, 20, (0,0,255), -1)
        
        if bamlane == 'phai':
            cv2.rectangle(img, pts_lanephai[0], pts_lanephai[1], (255,0,0), -1)
        elif bamlane == 'trai':
            cv2.rectangle(img, pts_lanetrai[0], pts_lanetrai[1], (255,0,0), -1)

    predict = model.predict(np.array([img])/255.0)[0]
    center = int(predict[0]*WIDTH)
    print(predict, center)

    cv2.circle(img, (center, 90), 5, (0, 255, 0), -1)
    cv2.imshow('img', img)

    k =cv2.waitKey(t)
    if k ==ord('r'):
        side = 1 - side
    if k ==32:
        t = 2-t
    if k == ord('q'):
        break
cv2.destroyAllWindows()

K.clear_session()
