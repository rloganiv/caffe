""" Takes the glove embedding and LSTM weights from the language_fusion model into this langauge model
Run in lstm_lm directory
"""
import lmdb
import numpy as np
import sys

sys.path.append("../../caffe/python/")	# Makes caffe findable, super not great coding.
import caffe


caffe.set_mode_cpu()

# Load source network
model = '../language_fusion/deploy_lm_deepfus_img512_s2vt_glove.words_to_preds.prototxt'
weights = '../language_fusion/snapshots/indomain_deepfusion.caffemodel'
source_net = caffe.Net(model, weights, caffe.TEST)


# Load target network
solver = caffe.SGDSolver("./lstm_lm_solver.prototxt")
target_net = solver.net

# Transfer glove embedding layer
layer='embedding_input'
source_layer = source_net.params[layer][0].data
assert source_layer.shape == target_net.params[layer][0].data.shape		# Check transferring layers is allowed
target_net.params[layer][0].data[...] = source_layer

# Transfer languge model weights
source_layer_name='lstm_lm'
target_layer_name='lstm1'
source_layer = source_net.params[source_layer_name][0].data
assert source_layer.shape == target_net.params[target_layer_name][0].data.shape
target_net.params[target_layer_name][0].data[...] = source_layer
# Save operated target network
target_net.save('./trainme.caffemodel')
