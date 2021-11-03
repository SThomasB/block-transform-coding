

import sys

import os

import pickle

import glob

import numpy as np

from skimage.io import imread

from block_coding_tools import decompose, transforms as tfs, evaluation



def main():
    pass

if __name__=="__main__":
    _, transform, block_size, path_to_data,*options = sys.argv
    block_size = int(block_size)
    path_to_data = os.path.expanduser(path_to_data)

    U = tfs.__dict__[transform](int(block_size))
    Gtc = []
    i=0
    for path in glob.glob(path_to_data):

        image = imread(path, as_gray=True)-0.5

        blocks = decompose(image, block_size)

        coeffs = list(U@block@U.T for block in blocks)
        
        Gtc.append(
            evaluation.transform_coding_gain(list(c.ravel() for c in coeffs))
        )
        i += 1


    mean = np.mean(Gtc)
    stdd = np.sqrt(np.var(Gtc))

    print(f"Result for {transform}:")
    print(f"\t mean: {mean:.3f}\n\t standard deviation: {stdd:.3f}")
    
    if "write" in options:
        _, target_path = options
        with open(target_path, "wb"):
            pickle.dump(
                evaluation.Result(mean, stdd),
                target_path
            )
    
    #END

