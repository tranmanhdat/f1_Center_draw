import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from __config__ import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=None)
args = parser.parse_args()

if args.model is None:
    args.model = model_name

export_dir = 'models/saved_model/'+args.model
graph_pb = 'models/pb/'+args.model+'.pb'

if not os.path.exists(export_dir):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.gfile.GFile(graph_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sigs = {}

    with tf.Session(graph=tf.Graph()) as sess:
        # name='' is important to ensure we don't get spurious prefixing
        tf.import_graph_def(graph_def, name='')
        g = tf.get_default_graph()
        inp = g.get_tensor_by_name('input_1:0')
        out = g.get_tensor_by_name('center/Sigmoid:0')

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {'in': inp}, {'out': out})

        builder.add_meta_graph_and_variables(sess,
                                            [tag_constants.SERVING],
                                            signature_def_map=sigs)

    builder.save()



# saved model to trt
from tensorflow.python.compiler.tensorrt import trt_convert as trt # TensorFlow ≥ 1.14.1
# import tensorflow.contrib.tensorrt as trt # TensorFlow <= 1.13.1
from tensorflow.python.saved_model import tag_constants

import tensorflow.keras.backend as K
K.clear_session()

input_saved_model_dir = 'models/saved_model/'+args.model
output_saved_model_dir = 'models/tf_trt/'+args.model+'_trt_FP16'

def save_tftrt():
    converter = trt.TrtGraphConverter(
        input_saved_model_dir=input_saved_model_dir,
        max_workspace_size_bytes=(11<32),
        precision_mode='FP16',
        maximum_cached_engines=100)
    converter.convert()
    converter.save(output_saved_model_dir)
save_tftrt()
