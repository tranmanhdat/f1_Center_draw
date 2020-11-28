import tensorflow as tf
from tensorflow import keras
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import argparse
from __config__ import *

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model")
args = parser.parse_args()

if args.model is None:
    args.model = model_name

export_dir = 'models/saved_model/'+args.model


# saved model to trt
# from tensorflow.python.compiler.tensorrt import trt_convert as trt # TensorFlow ≥ 1.14.1
import tensorflow.contrib.tensorrt as trt # TensorFlow <= 1.13.1
from tensorflow.python.saved_model import tag_constants

import tensorflow.keras.backend as K
K.clear_session()

input_saved_model_dir = 'models/saved_model/'+args.model
output_saved_model_dir = 'models/tf_trt/'+args.model+'_trt_FP16'

def save_tftrt():
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(max_workspace_size_bytes=(11<32))
    conversion_params = conversion_params._replace(precision_mode="FP16")
    conversion_params = conversion_params._replace( maximum_cached_engiens=100)

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir,)
    converter.convert()
    converter.save(output_saved_model_dir)
save_tftrt()
