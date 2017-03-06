"""CLI for extracting VGG features from images/movies"""
import argparse
import caffe
import numpy as np
from PIL import Image
import random
import skvideo.io
import tarfile

import pdb


# Constants
RESIZE_DIM = (256, 256)
OUT_DIM = (224, 224)
MEAN_PIXEL = np.array([123.68, 116.779, 103.939], dtype=np.float32)

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='file to process')
parser.add_argument('-o', '--output', help='output data')
args = parser.parse_args()


def extract_frames(fname):
    try:
        vreader = skvideo.io.vreader(fname)
        for i, frame in enumerate(vreader):
            if (i % 10) == 0:
                yield Image.fromarray(frame)
    except:
        print "EXCEPTION!!!"


def process_tarf(tarf, net):
    for member in tarf.getmembers():
        file_name = member.name
        file_id = file_name.split('.')[0]
        file_id = file_id.replace('/', '-')
        print file_id
        file_type = file_name.split('.')[-1]
        if file_type == 'jpg':
            img_file = tarf.extractfile(member)
            img = PIL.open(img_file)
            arr = preprocess_img(img)
            feats = extract_vgg(arr, net).flatten()
            feat_str = ', '.join(map(str, feats))
            yield file_id + '_frame_1, ' + feat_str + '\n'
        if file_type == 'avi':
            # Temporarily extract video
            avi_file = tarf.extractfile(member)
            with open('tmp/vid.avi', 'wb') as f:
                f.writelines(avi_file.readlines())
            for frame_id, img in enumerate(extract_frames('tmp/vid.avi')):
                arr = preprocess_img(img)
                feats = extract_vgg(arr, net).flatten()
                feat_str = ', '.join(map(str, feats))
                yield '%s_frame_%i, %s \n' % (file_id, frame_id, feat_str)
        else:
            continue


def preprocess_img(img):
    """Resize and randomly crop image
    
    Args:
        img - PIL.Image object

    Returns:
        np.array containing processed image data.
    """
    # resize image
    img = img.resize(RESIZE_DIM)
    # Random crop
    x = random.randint(0, RESIZE_DIM[0] - OUT_DIM[0])
    y = random.randint(0, RESIZE_DIM[1] - OUT_DIM[1])
    img = img.crop((x, y, x + OUT_DIM[0], y + OUT_DIM[1]))
    # Format as numpy matrix
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    arr -= MEAN_PIXEL
    arr = arr.transpose((2,0,1))
    arr = arr[[2, 1, 0], :, :]
    return arr


def extract_vgg(img, net):
    """Extract VGG features from image

    Args:
        img - Preprocessed image array.
        net - VGG net.

    Returns:
        Extracted fc7 features.
    """
    net.blobs['data'].reshape(1, 3, OUT_DIM[0], OUT_DIM[1])
    net.blobs['data'].data[...] = img
    net.forward()
    return net.blobs['fc7'].data


if __name__ == '__main__':
    # Set up caffe to use GPU
    caffe.set_device(0)
    caffe.set_mode_gpu()
    # Initialize vgg-network
    net = caffe.Net('VGG_ILSVRC_16_layers_deploy.prototxt',
                    'VGG_ILSVRC_16_layers.caffemodel',
                    caffe.TEST)
    # Process file
    # TODO: Add filetype handling
    with tarfile.open(args.input) as tarf:
        with open(args.output, 'w') as outfile:
            outfile.writelines(process_tarf(tarf, net))

