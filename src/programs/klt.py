


import numpy as np
import os
import glob
import skimage.io
import random

size = 384

blocks = [random.choice(list(range(384-8))) for _ in range(10)]



reader = lambda x: (
	row for row in (skimage.io.imread(
		x,
		as_gray=True
	)[block:block+8, block:block+8] - 0.5 for block in blocks)
)

for row in reader(os.path.expanduser("~/Data/test/Lenna.png")):
	print(row)

