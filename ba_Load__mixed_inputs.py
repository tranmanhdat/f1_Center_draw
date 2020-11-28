import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
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


# nuadau
# bamlane = 'nuadau'
# route = 'nuadau'

# # phai_lanephai
bamlane = 'lanephai'
route = 'phai'

# # thang_lanephai
# bamlane = 'lanephai'
# route = 'thang'

# # thang_lanetrai
# bamlane = 'lanetrai'
# route = 'thang'


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
t = 50
i_route = indices_to_one_hot([routes_mapping[route]], len(routes_mapping))
i_lane = indices_to_one_hot([lanes_mapping[bamlane]], len(lanes_mapping))
# print('i_route', i_route)


init_op = tf.initialize_all_variables()
# sess = tf.Session()
sess = tf.InteractiveSession()
sess.run(init_op)

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

    predict = model.predict([np.array([img])/255.0, i_route, i_lane])[0]
    center = int(predict[0]*WIDTH)
    print(predict, center)

    
    # merged_input = model.layers[17].output
    # dense1 = model.layers[19].output
    # dense = model.layers[18].output
    # leaky_re_lu_4 = model.layers[20].output # after dense
    # leaky_re_lu_5 = model.layers[21].output # after dense1
    
    # inputs = {'inputs_img:0': np.array([img])/255.0, 'inputs_route:0': i_route, 'inputs_lane:0': i_lane}
    # dense1 = sess.run(dense1, feed_dict=inputs)
    # leaky_re_lu_5 = sess.run(leaky_re_lu_5, feed_dict=inputs)
    # dense = sess.run(dense, feed_dict=inputs)
    # leaky_re_lu_4 = sess.run(leaky_re_lu_4, feed_dict=inputs)
    # merged_input = sess.run(merged_input, feed_dict=inputs)
    
    # print('merged_input', merged_input)
    # print('dense', dense)
    # print('leaky_re_lu_4', leaky_re_lu_4)
    # print('dense1', dense1)
    # print('leaky_re_lu_5', leaky_re_lu_5)

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
