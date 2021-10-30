
.. runblock:: pycon

	>>> import numpy as np
	>>> from block_coding_tools import compose, decompose
	>>> A = np.zeros((4,4))
	>>> blocks_of_A = decompose(A, 2)
	>>> for i, a in enumerate(blocks_of_A):
	...		print(f"block{i}:", *(a+i), sep ="\n\t")

	>>> B = compose(A.shape, 2, (b+i for i,b in enumerate(decompose(A, 2))))
	>>> print("The recomposed array:\n", *B, sep="\n\t")

