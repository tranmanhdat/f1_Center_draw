import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import numpy as np
import tensorflow as tf

# Fixed error Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
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
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.activations import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

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
    x = MaxPooling2D(pool_size=(2, 2))(x)

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
X = np.load(data_path_x)
Y = np.load(data_path_y)
y1= Y[:,0]
Y = y1

def generator(x, y, batch_size):
    samples = x.shape[0]
    steps = samples/batch_size
    counter = 0
    while True:
        batch_x = np.array(x[batch_size*counter: batch_size*(counter+1)]).astype(np.float32)
        batch_x = batch_x/255.0 
        batch_y = np.array(y[batch_size*counter: batch_size*(counter+1)]).astype(np.float32)
        counter += 1
        yield batch_x, batch_y
        if counter >= steps:
            counter = 0
            
''' Model '''
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(64, input_dim=2, activation='relu'))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

# model.fit(X, Y, batch_size=1, nb_epoch=100, verbose=0)

inputs = Input(shape=INPUT_SHAPE)
x = extractor(inputs)
# x = extractor_new(inputs)
center_output = Dense(units=1, activation='sigmoid', name='center')(x)

model = Model(
    inputs=inputs,
    outputs=[center_output],
    name='CenterNet')
optimizer = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.summary()



checkpoint = ModelCheckpoint('models/'+model_name+'.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=50)

history = model.fit_generator(generator(X, Y, batch_size), epochs=epochs, steps_per_epoch=X.shape[0]/batch_size, callbacks=[checkpoint, early_stopping])


''' Save graph '''
# inputs:  ['dense_input']
print('inputs: ', [input.op.name for input in model.inputs])

# outputs:  ['dense_4/Sigmoid']
print('outputs: ', [output.op.name for output in model.outputs])

# model.save(model_name+'.h5')

frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, 'models/pbtxt', model_name+'.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, 'models/pb', model_name+'.pb', as_text=False)


''' Plot '''
import matplotlib.pyplot as plt

loss = history.history['loss']
epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()
