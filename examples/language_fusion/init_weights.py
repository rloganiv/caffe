""" Takes the glove embedding and LSTM weights from the language_fusion model into this langauge model
Run in lstm_lm directory
"""
import numpy as np
import sys

import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

# Load source network
model = '/home/ubuntu/caffe/examples/language_fusion/deploy_lm_deepfus_img512_s2vt_glove.words_to_preds.prototxt'
weights = 'snapshots/indomain_deepfusion.caffemodel'
source_net = caffe.Net(model, weights, caffe.TEST)

# Load target network
solver = caffe.SGDSolver("solver_cocolm_deepfus_img512_s2vt_glove.prototxt")
target_net = solver.net

# Transfer glove embedding layer
layer='embedding_input'
source_layer = source_net.params[layer][0].data
assert source_layer.shape == target_net.params[layer][0].data.shape		# Check transferring layers is allowed
target_net.params[layer][0].data[...] = source_layer

# Transfer languge model weights
source_layer_name='lstm_lm'
target_layer_name='lstm_lm'
source_layer = source_net.params[source_layer_name][0].data
assert source_layer.shape == target_net.params[target_layer_name][0].data.shape
target_net.params[target_layer_name][0].data[...] = source_layer
# Save operated target network
target_net.save('/home/ubuntu/data/weights/init_deepfus.caffemodel')
