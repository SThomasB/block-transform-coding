import numpy as np
def quantize(blocks, quantum: float):
    for block in blocks:
        yield np.fix(
            np.divide(block + np.sign(block)*quantum/2, quantum)
        )

def dequantize(blocks, quantum: float, block_size):

    max_index = block_size/(2*quantum)
    for block in blocks:
        yield block*quantum + (np.abs(block)!=max_index) * (-1/4*np.sign(block)*quantum)
