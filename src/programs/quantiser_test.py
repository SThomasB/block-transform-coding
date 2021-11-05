



from block_coding_tools import quantize
from matplotlib import pyplot as plt
import numpy as np
import sys

_, q = sys.argv
q = float(q)
x = np.linspace(-8,8,1601)
Q = np.array(list(quantize(x, q)))*q

plt.plot(x,Q)
plt.xlabel("k"); plt.ylabel("Q(k)")
plt.show()

