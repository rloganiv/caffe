""" Converts a text file of sentences into a hdf5 file to be read into a LSTM language model for caffe
Run in lmst_lm directory

TO CHANGE:
SENTS_FILENAME - change to match the filename of the Reddit data
"""

import h5py
import numpy as np
import os

class DataGenerator(object):
    def __init__(self, max_words=20):
        self.vocab = {UNK_IDENTIFIER: 0}
        self.vocab_inverted = []
        self.vocab_inverted.append(UNK_IDENTIFIER)
        self.max_words = max_words
        self.init_vocab(VOCAB_FILENAME)
        
    def init_vocab(self, vocab_filename):
        with open(vocab_filename, 'r') as vocab_file:
            for line in vocab_file:
                split_line = line.split()
                word = split_line[0]
                word = word.strip()
                if word == UNK_IDENTIFIER:
                    continue
                else:
                    assert word not in self.vocab
                self.vocab[word] = len(self.vocab_inverted)
                self.vocab_inverted.append(word)

    def save_to_hdf5(self, cont_sentence, input_sentence, target_sentence):
        if not(os.path.exists(OUTPUT_DIRECTORY)):
            os.makedirs(OUTPUT_DIRECTORY)

        cont_sentence = np.vstack(cont_sentence).astype(np.float)
        input_sentence = np.vstack(input_sentence).astype(np.float)
        target_sentence = np.vstack(target_sentence).astype(np.float)
        assert OUTPUT_DIRECTORY is not None
        assert OUTPUT_FILENAME is not None
        assert HDF5_TEXTFILE is not None


        i = 0
        file_size = 1000
        num_data = cont_sentence.shape[0]

        with open(OUTPUT_DIRECTORY+'/'+HDF5_TEXTFILE, 'w') as f:
            while(i*file_size < num_data):
                path_from_caffe = './examples/lstm_lm/hdf5' 
                f.write(path_from_caffe+ '/' + OUTPUT_FILENAME.format(i)+'\n')

                h5f = h5py.File(OUTPUT_DIRECTORY+ '/' + OUTPUT_FILENAME.format(i), 'w')
                low_inds = i * file_size
                high_inds = min((i+1)*file_size, num_data)
               
                h5f.create_dataset('cont_sentence', data=cont_sentence[low_inds:high_inds, :])
                h5f.create_dataset('input_sentence', data=input_sentence[low_inds:high_inds, :])
                h5f.create_dataset('target_sentence', data=target_sentence[low_inds:high_inds, :])
                h5f.close()
                i += 1
            


    def line_to_stream(self, sent):
        stream = []
        sent = sent.split()
        for word in sent:
            word = word.strip()
            if word in self.vocab:
                stream.append(self.vocab[word])
            else:
                stream.append(self.vocab[UNK_IDENTIFIER])
        # increment the ids in the stream -- 0 will be the EOS character
        stream = [s+1 for s in stream]
        return stream

    def data_to_hdf5(self, sents_filename):
        cont_sentence = []
        input_sentence = []
        target_sentence = []

        with open(sents_filename, 'r') as sents_file:
            for line in sents_file:
                stream = self.line_to_stream(line)
                pad = self.max_words - (len(stream) + 1)        # +1 is for the extra EOS character
                if pad < 0:
                    stream = stream[:self.max_words-1]         # +1 is for the extra EOS character
                    pad = 0

                cont_sent = [0] + [1]*len(stream) + [0]*pad
                input_sent = [0] + stream + [0]*pad
                target_sent = stream + [0] + [-1]*pad
                assert len(cont_sent) == self.max_words
                assert len(input_sent) == self.max_words
                assert len(target_sent) == self.max_words

                cont_sentence.append(cont_sent)
                input_sentence.append(input_sent)
                target_sentence.append(target_sent)


        self.save_to_hdf5(cont_sentence, input_sentence, target_sentence)

SENTS_FILENAME= './sents.txt'				# Directory path to the sentences file, CHANGE THIS TO MATCH FILENAME
VOCAB_FILENAME = './examples/language_fusion/vocabulary_72k_surf_intersect_glove.txt' # Directory path to the vocab file
OUTPUT_DIRECTORY='./hdf5'				# Directory containing our hdf5 files
OUTPUT_FILENAME='data{0}.h5'			# Name of our hdf5 file(s)
HDF5_TEXTFILE='hdf5_list.txt'				# Txt file listing our hdf5 files
UNK_IDENTIFIER = 'unk'
if __name__=="__main__":
    dg = DataGenerator()
    dg.data_to_hdf5(SENTS_FILENAME)

