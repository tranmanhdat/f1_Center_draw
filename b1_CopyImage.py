import cv2
import numpy as np
import os
import argparse
import shutil
from __config__ import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-p', '--prefix')
args = parser.parse_args()


for filename in os.listdir(args.input):
    filepath = args.input+'/'+filename

    new_filename = args.prefix+'__'+filename.split('_')[0]+'.jpg'
    # shutil.copy(filepath, 'images/'+new_filename)

    # resize
    img = cv2.imread(filepath)
    img = cv2.resize(img, (320,240))
    cv2.imwrite(san+'/images/'+part+'/'+new_filename, img)

