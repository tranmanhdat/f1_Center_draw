import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Clear any previous session.
tf.keras.backend.clear_session()

save_pb_dir = 'models/pb/'
model_fname = 'models/draw_320x180__140+40__old_model_1.h5'

# def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
#     with graph.as_default():
#         graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
#         graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
#         graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
#         return graphdef_frozen

# # This line must be executed before loading Keras model.
# tf.keras.backend.set_learning_phase(0)

# model = load_model(model_fname)

# session = tf.keras.backend.get_session()

# input_names = [t.op.name for t in model.inputs]
# output_names = [t.op.name for t in model.outputs]

# # Prints input and output nodes names, take notes of them.
# print(input_names, output_names)

# frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)


# import tensorflow.contrib.tensorrt as trt

# trt_graph = trt.create_inference_graph(
#     input_graph_def=frozen_graph,
#     outputs=output_names,
#     max_batch_size=1,
#     max_workspace_size_bytes=1 << 25,
#     precision_mode='FP16',
#     minimum_segment_size=50
# )

# graph_io.write_graph(trt_graph, 'models/pb/', 'trt_graph.pb', as_text=False)


print(' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OK 1')


output_names = ['center/Sigmoid']
input_names = ['input_1']

import tensorflow as tf

def get_frozen_graph(graph_file):
    '''Read Frozen Graph file from disk.'''
    with tf.gfile.FastGFile(graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph('models/pb/trt_graph.pb')

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

print(' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OK 2')

# Get graph input size
for node in trt_graph.node:
    if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print('image_size: {}'.format(image_size))


# input and output tensor names.
input_tensor_name = input_names[0] + ':0'
output_tensor_name = output_names[0] + ':0'

print('input_tensor_name: {}\noutput_tensor_name: {}'.format(
    input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)


print(' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OK 3')


# Optional image to test model prediction.
img_path = '/home/ubuntu/catkin_ws/src/mtapos/output/2020-02-26_22-12-31/crop_cut/1039__0_25.0.jpg'

# img = image.load_img(img_path, target_size=image_size[:2])
# x = image.img_to_array(img)

img = cv2.imread(img_path)
img_more = np.zeros(shape=[40, 320, 3], dtype=np.uint8)
x = cv2.vconcat([img, img_more])

x = np.expand_dims(x, axis=0)

print(' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OK 4')

# x = preprocess_input(x)

feed_dict = {
    input_tensor_name: x
}
preds = tf_sess.run(output_tensor, feed_dict)

print('Predicted:', preds)