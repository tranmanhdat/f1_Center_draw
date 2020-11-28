import os
import cv2
import numpy as np
from __config__ import *


route = None

# print('path_raw_resized', path_raw_resized)
for filename in os.listdir(path_raw_resized):
    if filename[0] == 'r': # right
        route = 'phai'
    elif filename[0] == 's': # straight
        route = 'thang'
    elif filename[0] == 'l': # left
        route = 'trai'
    else:
        route = None
    
    img_raw_path = path_raw_resized+filename
    # print(img_raw_path)

    img_raw = cv2.imread(img_raw_path)
    rh,rw = img_raw.shape[:2]
    # img_raw = img_raw[rh-HEIGHT:, :]
    # print('img_draw', img_draw.shape)
    
    if draw_guide_append:
        img_more = np.zeros(shape=[40, 320, 3], dtype=np.uint8)
        if route is not None:
            if route == 'thang':
                cv2.circle(img_more, org_thang, 20, (0,0,255), -1)
            elif route == 'phai':
                cv2.circle(img_more, org_phai, 20, (0,0,255), -1)
            elif route == 'trai':
                cv2.circle(img_more, org_trai, 20, (0,0,255), -1)
            
            if lanepath == 'lanephai':
                cv2.rectangle(img_more, pts_lanephai[0], pts_lanephai[1], (255,0,0), -1)
            elif lanepath == 'lanetrai':
                cv2.rectangle(img_more, pts_lanetrai[0], pts_lanetrai[1], (255,0,0), -1)

        img_draw = cv2.vconcat([img_raw, img_more])

    else:
        img_draw = img_raw.copy()

        if route is not None:
            if route == 'thang':
                cv2.circle(img_draw, org_thang, 20, (0,0,255), -1)
            elif route == 'phai':
                cv2.circle(img_draw, org_phai, 20, (0,0,255), -1)
            elif route == 'trai':
                cv2.circle(img_draw, org_trai, 20, (0,0,255), -1)
            
            if lanepath == 'lanephai':
                cv2.rectangle(img_draw, pts_lanephai[0], pts_lanephai[1], (255,0,0), -1)
            elif lanepath == 'lanetrai':
                cv2.rectangle(img_draw, pts_lanetrai[0], pts_lanetrai[1], (255,0,0), -1)
    
    cv2.imwrite(path_draw+filename, img_draw)
    cv2.imshow('img_draw', img_draw)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()