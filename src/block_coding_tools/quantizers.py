import numpy as np
def quantize(blocks, quantum: float):
    for block in blocks:
        yield np.fix(
            np.divide(block + np.sign(block)*quantum/2, quantum)
        )
