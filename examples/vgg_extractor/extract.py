"""CLI for extracting VGG features from images/movies"""
import argparse
import caffe
import numpy as np
import os
from PIL import Image
import random
import tarfile

import pdb


# Constants
RESIZE_DIM = (256, 256)
OUT_DIM = (224, 224)
MEAN_PIXEL = np.array([103.939, 116.779, 123.68], dtype=np.float32)

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='file to process')
parser.add_argument('-o', '--output', help='output data')
args = parser.parse_args()


def preprocess_img(img_file):
    """Resize and randomly crop image
    
    Args:
        img - An image file.

    Returns:
        processed image data.
    """
    # Open and resize image
    img = Image.open(img_file)
    img = img.resize(RESIZE_DIM)
    # Random crop
    x = random.randint(0, RESIZE_DIM[0] - OUT_DIM[0])
    y = random.randint(0, RESIZE_DIM[1] - OUT_DIM[1])
    img = img.crop((x, y, x + OUT_DIM[0], y + OUT_DIM[1]))
    # Format as numpy matrix
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    arr -= MEAN_PIXEL
    arr = arr.transpose((2,0,1))
    return arr


def extract_vgg(net, img):
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
    print net.blobs['data'].data.shape

    with open('dog.jpg', 'r') as img_file:
        img = preprocess_img(img_file)
    print extract_vgg(net, img).shape

