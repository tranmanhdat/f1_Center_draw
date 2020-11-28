import cv2
import numpy as np
import sys
import argparse
from __config__ import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-p', '--prefix')
args = parser.parse_args()


prefix = ''
postfix = ''
if part == 'thang_lanephai':
    prefix = 's{}__'.format(args.prefix)
if part == 'trai_lanephai':
    prefix = 'l{}__'.format(args.prefix)
if part == 'thang_lanetrai':
    prefix = 's{}__'.format(args.prefix)
    postfix = '__lanetrai'
if part == 'phai_lanephai':
    prefix = 'r{}__'.format(args.prefix)

cap = cv2.VideoCapture(args.input)

if not cap.isOpened():
    print('Unable to read camera feed')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print('frame_width : ' + str(frame_width))
print('frame_height : ' + str(frame_height))

skip = True
add = 2012
k = 0
t = 15
count = 2012
while True:
    ret, frame = cap.read()

    if ret:

        # Display the resulting frame
        cv2.imshow('frame', frame)

        key = cv2.waitKey(t)
        if key == 27:
            break
        elif key == ord('s'):
            print('start saving')
            skip = False
        elif key == ord('q'):
            skip = True
            print('stop saving')
        elif key == 82:  # up arrow
            t = t + 5
            print(t)
        elif key == 84:
            t = t - 5
            print(t)
        if not skip:
            k = k + 1
            if k % 1 == 0:
                cv2.imwrite(san+'/images/'+part+'/' + prefix + str(k+add) + postfix + '.jpg', frame)
                count = count + 1
                print(count)
    else:
        break

    # When everything done, release the video capture and video write objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
