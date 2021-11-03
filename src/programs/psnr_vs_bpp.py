



import sys

import os

from matplotlib import pyplot as plt

import pickle

import glob

import numpy as np

from skimage.io import imread

from skimage.metrics import peak_signal_noise_ratio as ski_psnr

from block_coding_tools import (
    decompose, compose, transforms as tfs, evaluation, quantize, dequantize, encoding
)

import time

def main():
    pass

def check_size_and_crop(img, block_size):
    h, w = img.shape
    dh = h%block_size
    dw = w%block_size
    return img[0:h-dh, 0:w-dw]
if __name__=="__main__":
    _, transform, block_size, q, path_to_data, *options = sys.argv
    
    q = float(q)
    print(q)
    block_size = int(block_size)

    path_to_data = os.path.expanduser(path_to_data)

    U = tfs.__dict__[transform](int(block_size))

    for path in glob.glob(path_to_data):

        image = imread(path, as_gray=True)
        image = check_size_and_crop(image, block_size)
        # skimage.io.imread converts the image to float64 in (0, 1),
        # but if the image is already  grayscale it does not convert it.
        # Therefore normalize:
        image = image/np.max(image) - 0.5

        blocks = decompose(image, block_size)

        coeffs = (U@block@U.T for block in blocks)

        symbols = compose(image.shape, block_size, quantize(coeffs, q))

        coder = encoding.Coder.from_histogram(
                evaluation.histogram(symbols),
                keep_histogram=True
        )
        coder.set_source_shape(image.shape)
        
        
        print(coder)
        
        

        J = compose(
            image.shape,
            block_size,
            (U.T@(q*block)@U for block in decompose(symbols, block_size))
        )


        J[J>0.5] = 0.5
        J[J<-0.5] = -0.5
        psnr = evaluation.peak_signal_to_noise(image,J, 1)
        ski = ski_psnr(image,J, data_range=1)
        print(psnr)
        plt.imshow(J)
        
        plt.show()
        code_length=0
        

    
    if "write" in options:
        _, target_path = options
        with open(target_path, "wb"):
            pickle.dump(
                evaluation.Result(mean, stdd),
                target_path
            )
