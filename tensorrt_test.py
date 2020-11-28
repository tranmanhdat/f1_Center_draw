import tensorrt.legacy as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import uff
import cv2

# from tensorrt import parsers
# from tensorrt.parsers import uffparser

from tensorrt.legacy.parsers import uffparser

# model_name = 'sign_2__ok'
model_name = 'draw_320x180__140+40__new_model_2__new'
max_batch_size = 128
input_channels, input_height, input_width = 3,180,320
# input_channels, input_height, input_width = 3,100,100


# In[1]:
# uff_model = uff.from_tensorflow('models/'+model_name+'.pb')

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

# parser = uffparser.create_uff_parser()
# parser.register_input('input_1', (input_channels, input_height, input_width), 0)
# # parser.register_output('center/Sigmoid')
# # parser.register_output('activation_1/Sigmoid')
# parser.register_output('activation_1/Softmax')

# engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1 << 20)
# trt.utils.write_engine_to_file('models/'+model_name+'.engine', engine.serialize())

# engine.destroy()
# parser.destroy()




# In[2]:
engine = trt.utils.load_engine(G_LOGGER, 'models/'+model_name+'.engine')


OUTPUT_SIZE = (1,1)

img = cv2.imread('/media/F1AC-0068/images_draw/24.jpg')
img = img[20:,:]

img = img.astype(np.float32)
#create output array to receive data
output = np.empty(OUTPUT_SIZE, dtype = np.float32)

runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

print('Ok0')


# alocate device memory
d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)

# # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
# h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
# h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
# # Allocate device memory for inputs and outputs.
# d_input = cuda.mem_alloc(h_input.nbytes)
# d_output = cuda.mem_alloc(h_output.nbytes)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

print('Ok1')

#transfer input data to device
cuda.memcpy_htod_async(d_input, img, stream)

print('Ok2')
#execute model
context.enqueue(1, bindings, stream.handle, None)

print('Ok3')
#transfer predictions back
cuda.memcpy_dtoh_async(output, d_output, stream)
#syncronize threads
stream.synchronize()

print('Ok4')

print ('Prediction: ', output)

context.destroy()
engine.destroy()
runtime.destroy()

