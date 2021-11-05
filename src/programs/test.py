
import numpy as np

import concurrent.futures
import time
import os
import skimage.io
from typing import NamedTuple
from block_coding_tools import transforms, evaluation, encoding, compose, decompose, quantize
from matplotlib import pyplot as plt
tfs = [transforms.dct, transforms.dst]
block_size = 8
class Result(NamedTuple):
	entropy: float
	bpp: float
	psnr: float

def normalise(x):
	x / np.max(x)
	return x

def process_image(X, f):
	U = f(block_size)
	xs = decompose(X, block_size)
	ks = compose(X.shape, 8, (U@x@U.T for x in xs))
	res = []
	for q in range(0,50, 2):
		q = (q+1)/100
				
		symbols = np.array(list(quantize(ks, q)))
		coder = encoding.Coder.from_histogram(evaluation.histogram(symbols), keep_histogram=True)
		entropy = evaluation.entropy([prob for _, prob in coder.histogram], is_histogram=True)
		Y = compose(
			X.shape,
			block_size,
			(U.T@s@U for s in decompose(symbols*q, block_size))
		)
		# clamp Y to -0.5, 0.5
		Y = np.vectorize(lambda x: max(-0.5, min(x, 0.5)))(Y)
		
		psnr = evaluation.peak_signal_to_noise(X,Y, 1)
		plt.imshow(Y)
		plt.draw()
		res.append(Result(entropy, coder.estimate_header_size(), psnr))
	return res


path = os.path.expanduser("~/Data/test/Lenna.png")
X = skimage.io.imread(path, as_gray=True)
X = normalise(X) - 0.5

with concurrent.futures.ProcessPoolExecutor() as executor:
	results = [
		executor.submit(process_image, X.copy(), f) for f in tfs
	]	
	
	for f in concurrent.futures.as_completed(results):
		print(f.result())

