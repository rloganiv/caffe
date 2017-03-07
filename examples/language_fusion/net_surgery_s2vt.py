""" Takes trained language model into the sequence to sequence video captioner network"""

import numpy as np
import caffe
import pdb

caffe.set_mode_gpu()

# Load source network
model = '../lstm_lm/lstm_lm.deploy.prototxt'
weights = '../lstm_lm/snapshots/snapshot_iter_1000.caffemodel'	# Change to match snapshot
source_net = caffe.Net(model, weights, caffe.TEST)

# Load target network
model = 'deploy_lm_deepfus_img512_s2vt_glove.words_to_preds.prototxt'
weights = 'snapshots/indomain_deepfusion.caffemodel'
target_net = caffe.Net(model, weights, caffe.TEST)


# Transfer LSTM language model layer
source_layer_name='lstm1'
target_layer_name='lstm_lm'
source_layer = source_net.params[source_layer_name][0].data
assert source_layer.shape == target_net.params[target_layer_name][0].data.shape
target_net.params[target_layer_name][0].data[...] = source_layer

# Save operated target network
target_net.save('trainme_s2vt.caffemodel')