""" Orthonormal transform matrices

.. module:: transforms 
   :platform: Unix, Windows
.. moduleauthor:: Thomas Berg Svendsen

"""



import numpy as np




def dct(N: int) -> np.array:
    """
        Returns the orthonormal dct-II matrix of side N as defined by:

        .. math::

            f(n,k) &= cos(\\frac{\pi k}{N}(n + \\frac{1}{2}))\\


            g(k,n) &= \\frac{1}{\sqrt{N}}\,\,if\,\, k = 0\,\, else\,\, \sqrt{\\frac{2}{N}}\\


            DCT_N(n,k) &= f(n,k)g(k),\, n=0,..N-1, \,\, k=0,..N-1
        .. runblock:: pycon

            >>> from matplotlib import pyplot as plt
            >>> import numpy as np
            >>> from block_coding_tools import transforms as tfs
            >>> from block_coding_tools import compose
            >>> base_images = (np.outer(f1, f2) for f1 in tfs.dct(8) for  f2 in tfs.dct(8).T)
            >>> #plt.imshow(compose((64,64), 8, base_images), cmap='gray')
            >>> #plt.title("DCT-II bases images")


        .. plot::
        
            >>> from matplotlib import pyplot as plt
            >>> import numpy as np
            >>> from block_coding_tools import transforms as tfs
            >>> from block_coding_tools import compose
            >>> base_images = (np.outer(f1, f2) for f1 in tfs.dct(8) for  f2 in tfs.dct(8).T)
            >>> plt.figure(figsize=(5,5)) 
            >>> plt.imshow(compose((64,64), 8, base_images), cmap='gray')
            >>> plt.title("DCT-II bases images")
            >>> plt.show()

    """
    f = lambda n, k: np.cos( np.pi/N *(n+1/2) * k )
    g = lambda k: 1/np.sqrt(N) if k == 0 else np.sqrt(2/N)
    F = [[f(n,k) * g(k) for n in range(N)] for k in range(N)]
    return np.array(F)


def dst(N: int) -> np.array:
    """
        Returns the orthonormal dst matrix of side N as **defined by:**

        .. math::

            f(k,n) &= \sqrt{\\frac{2}{N+1}}sin(\\frac{\pi(k+1)(n+1)}{N+1})


            [DST_N(k,n)] &= f(k,n)            


    """
    f = lambda k,n: np.sqrt(2/(N+1)) * np.sin( np.pi*(k+1)*(n+1)/(N+1) )
    return np.array([[f(k,n) for n in range(N)] for k in range(N)])


def haar(N: int) -> np.array:
    """
        Returns the normalised haar matrix of side N, where the haar matrix *H* is defined by:

        .. math::

            H_2 &= {\\begin{bmatrix} 1 & 1 \\\ 1 & -1 \end{bmatrix}}
            
            
            H_{2N} &= {\\begin{bmatrix} H_N  \otimes [1\,\,1] \\\ I_N  \otimes  [1\,\,-1] \end{bmatrix}}

        
        .. note::

            The side of the matrix must be a multiple of two.
        


        .. plot::
        
            >>> from matplotlib import pyplot as plt
            >>> from block_coding_tools import transforms as tfs
            >>> base_functions = (f1 for f1 in tfs.haar(8))
            >>> for i, f in enumerate(base_functions):
            ...     plt.subplot(int(f'81{i+1}'))
            ...     plt.stem(f)
            >>> plt.title("haar bases vectors")
            >>> plt.show()
        
        

    """
    def __haar(N):
        if N == 2:
            return np.array([[1,1],[1,-1]])
        else:
            return np.vstack(
                (
                    np.kron(__haar(N//2), [1,1] ),
                    np.kron( np.eye(N//2), [1,-1])
                ) 
            )
    return np.array([x/np.sqrt(x.T@x) for x in __haar(N)])


def dwht(N):
    """
        Return the DWHT (*discrete walsh hadamard transform*) matrix.
        
        The hadamard matrix is defined as:

        .. math::

            H_2={\\begin{bmatrix} 1 & 1 \\\ 1 & -1 \end{bmatrix}}
            


            H_{2N}=H_N \otimes H_2
        


        *The dwht matrix is formed by sorting the rows of the hadamard matrix by
        the number of sign changes in each respective row, and normalising by N.*


        .. note::
            The side *N* must be a multiple of 2.


        
    """
    def __hadamard(N):
        if N == 2:
            return np.array([[1,1],[1,-1]])
        else:
            return np.kron(__hadamard(N//2), np.array([[1,1],[1,-1]]))

    x = __hadamard(N)

    # ORDER THE ROWS OF HADAMARD MATRIX BY THE NUMBER OF SIGN-CHANGES
    sequency = lambda x: sum([x[n]!=x[n-1] for n in range(1, N)])
    sequency = sorted([*enumerate([sequency(x) for x in x])], key = lambda x: x[1])
    
    # RETURN NORMALIZED WALSH HADAMARD TRANSFORM MATRIX
    return 1/np.sqrt(N) * np.array([x[i] for i,_ in sequency])


#class Training:
#    def __init__(self, transform, data_stream, description=""):
#        self.transform = Training.__dict__[transform](data_stream)
#        self.description = description


#   def klt(data_stream):
#        R = np.mean(
#                list(np.mean(list(np.outer(x,x) for x in data),0) for data in data_stream),
#                0
#        )
#        _, V = np.linalg.eigh(R)
#        V = np.fliplr(V).T
#        return V
#    
#    
#    def write(self, path):
#        with open(path, "wb") as fp:
#            pickle.dump(self, fp)

            
        


    
