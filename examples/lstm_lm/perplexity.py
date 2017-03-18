
from collections import OrderedDict
import math
import numpy as np
import caffe

from text_to_hdf5 import *

def words_to_ids(sent, fsg):
    """ Converts sentence to list of vocab ids """ 
    stream = []
    sent = sent.split()
    for word in sent:
        word = word.strip()
        if word in fsg.vocab:
            stream.append(fsg.vocab[word])
        else:
            stream.append(fsg.vocab[UNK_IDENTIFIER])
    # increment the ids in the stream -- 0 will be the EOS character
    stream = [s+1 for s in stream] 
    stream.append(0)       # Append EOS character
    return stream 

def sentence_perplexity(net, fsg, sent, vocab):
    """ Evaluates perplexity per sentence """
    stream = words_to_ids(sent, fsg)
    previous_word = 0   # Initializes our network with EOS character
    cont = np.array([1])

    sum_perplexity = 0
    sent_len = 0 
    
    for i in range(0,len(stream[:-1])):  # Get predictions putting in all words except EOS (since nothing comes after EOS)
        current_word = stream[i]
        next_word = stream[i+1]
        data_en = np.array([previous_word])
        net.forward(cont_sentence=cont, input_sentence=data_en)   # Forwards previous word through lstm
        output_preds = net.blobs['probs'].data.reshape(-1)
        sum_perplexity +=  output_preds[current_word]
        previous_word = current_word    #Sets previous word to current word
        sent_len += 1

    sent_perplexity = sum_perplexity / sent_len
    # print sent.strip()
    # print "Perplexity score is: ", sent_perplexity, "\n"
    return sent_perplexity

def run_evaluate_iter(net, fsg, vocab):
    """ Evaluates perplexity on list of sentences""" 
    sum_perplexity = 0
    num_sents = 0
    net_init_params = np.zeros(net.blobs['lstm1'].data.shape)   # Used to reset hidden state of lstm 
    with open(SENTS_FILE, 'r') as f:
        for line in f:
            sum_perplexity += sentence_perplexity(net, fsg, line, vocab)
            num_sents += 1
            net.blobs['lstm1'].data[...] = net_init_params    # Reset hidden state after each sentence

    overall_perplexity = sum_perplexity/num_sents
    print "Overall perplexity is: ", overall_perplexity

UNK_IDENTIFIER = 'unk'

if __name__== "__main__":
    DIR='/home/who/Desktop/caffe/examples/lstm_lm/snapshots'
    VOCAB_FILE='/home/who/Desktop/caffe/examples/lstm_lm/skytrax-reviews-dataset-master/data/vocab.txt'
    model = '/home/who/Desktop/caffe/examples/lstm_lm/lstm_lm.deploy.prototxt'
    weights = DIR+'/snapshot2_iter_1000.caffemodel'
    SENTS_FILE= '/home/who/Desktop/caffe/examples/lstm_lm/sents.txt'    # Change this to match sent file to evaluate

    caffe.set_mode_cpu()
    print "Setting up LSTM"
    lstm_net = caffe.Net(model, weights, caffe.TEST)
    print "Done"

    STRATEGIES = [
        {'type': 'beam', 'beam_size': 1},
    ]

    fsg = DataGenerator(VOCAB_FILE)
    eos_string = '<EOS>'
    # add english inverted vocab 
    vocab_list = [eos_string] + fsg.vocab_inverted

    run_evaluate_iter(lstm_net, fsg, vocab_list)
