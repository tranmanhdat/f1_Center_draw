#! /usr/bin/python
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.activations import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential
import numpy as np
import tensorflow as tf

# # Fixed error Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# # tensorflow 2
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from __config__ import *


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    '''
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    '''
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def extractor(inputs):
    # CONV => RELU => POOL
    x = Conv2D(16, (5, 5), strides=(2, 2), padding='same')(inputs)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # CONV => RELU => POOL
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    # CONV => RELU => POOL
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # define a branch of output layers for the number of different
    # colors (i.e., red, black, blue, etc.)
    x = Flatten()(x)
    # x = Reshape([6400])(x)
    # x = Reshape([5120])(x)
    x = Dropout(0.4)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


''' Load data '''
X_img = np.load(data_path_X_img, allow_pickle=True)
X_lane = np.load(data_path_X_lane, allow_pickle=True)
X_route = np.load(data_path_X_route, allow_pickle=True)
y = np.load(data_path_y, allow_pickle=True)

print('X_img.shape', X_img.shape)
# X_lane = to_categorical(X_lane)
# X_route = to_categorical(X_route)
X_route = indices_to_one_hot(X_route, len(routes_mapping))
X_lane = indices_to_one_hot(X_lane, len(lanes_mapping))

y1 = y[:, 0]
y = y1

shuffe_index = np.random.permutation(X_img.shape[0])  # hoan vi
N = X_img.shape[0]
K = int(train_size*N)
index = np.random.permutation(N)

X_img_train = X_img[index[0:K]]
X_lane_train = X_lane[index[0:K]]
X_route_train = X_route[index[0:K]]
y_train = y[index[0:K]]

X_img_test = X_img[index[K:]]
X_lane_test = X_lane[index[K:]]
X_route_test = X_route[index[K:]]
y_test = y[index[K:]]


print(N, K, N - K)
print(X_img.shape, X_route.shape, X_lane.shape, y.shape)


def generator(X_img, X_route, X_lane, y, batch_size):
    n_samples = X_img.shape[0]
    steps = n_samples/batch_size
    counter = 0
    while True:
        batch_X_img = np.array(
            X_img[batch_size*counter: batch_size*(counter+1)]).astype(np.float32)
        batch_X_img = batch_X_img/255.0
        
        batch_X_route = np.array(
            X_route[batch_size*counter: batch_size*(counter+1)]).astype(np.float32)

        batch_X_lane = np.array(
            X_lane[batch_size*counter: batch_size*(counter+1)]).astype(np.float32)

        batch_y = np.array(
            y[batch_size*counter: batch_size*(counter+1)]).astype(np.float32)
        counter += 1

        yield {'inputs_img': batch_X_img, 'inputs_route': batch_X_route, 'inputs_lane':batch_X_lane}, batch_y
        if counter >= steps:
            counter = 0


''' Model '''
inputs_img = Input(shape=(HEIGHT, WIDTH, 3), name='inputs_img')
inputs_route = Input(shape=(len(routes_mapping), ), name='inputs_route')
inputs_lane = Input(shape=(len(lanes_mapping), ), name='inputs_lane')

x1 = extractor(inputs_img)
# x1 = extractor_new(inputs_img)

merged_inputs = [inputs_route, inputs_lane]
concat_inputs = Concatenate(axis=1, name='merged_inputs')(merged_inputs)
# x2 = Flatten()(x2)
x2 = Dense(1, name='dense_inputs')(concat_inputs)
x2 = LeakyReLU(alpha=0.1)(x2)

# Merge features from image and guide (lane+route)
merged_features = [x1, x2]
merged = Concatenate(axis=1, name='merged_features')(merged_features)

center_output = Dense(units=1, activation='sigmoid', name='center')(merged)

model = Model(
    inputs=[inputs_img, inputs_route, inputs_lane],
    outputs=[center_output],
    name='CenterNet')
optimizer = Adam(lr=learning_rate)
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.summary()


''' Train '''
checkpoint = ModelCheckpoint('models/'+model_name+'.h5',
                             monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early_stopping = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=50)

history = model.fit_generator(generator(X_img_train, X_route_train, X_lane_train, y_train, batch_size), epochs=epochs, steps_per_epoch=X_img_train.shape[0]/batch_size,
                              validation_data=generator(X_img_test, X_route_test, X_lane_test, y_test, batch_size),
                              validation_steps=X_img_test.shape[0]/batch_size,
                              callbacks=[checkpoint, early_stopping])


''' Save graph '''
# inputs:  ['dense_input']
print('inputs: ', [input.op.name for input in model.inputs])

# outputs:  ['dense_4/Sigmoid']
print('outputs: ', [output.op.name for output in model.outputs])

# model.save(model_name+'.h5')

frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[
                              out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, 'models', model_name+'.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, 'models', model_name+'.pb', as_text=False)


''' Plot '''
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
