
from typing import NamedTuple

import numpy as np

class Result(NamedTuple):
    mean:float
    variance:float

def histogram(x: np.array) -> list[tuple]:
    """ Return a list of tuples [(value, relative_frequency)]
        where count is the relative number of occurences of value in the input.
        
        .. note::

            If the input is a list[list] instead of np.array, histogram will
            return the relative number of occurences of each unique list and the output type
            will be list[tuple[list, int]]

    """
    # These lists are sorted by value
    values, counts = np.unique(x, return_counts=True)
    relative_frequencies = counts/sum(counts)
    return [(value, relative_frequency) for value, relative_frequency in zip(values, relative_frequencies)]


def entropy(x: np.array, is_histogram=False) -> float:
    """ 
        If x is a normalised histogram, set is_histogram=True and this function returns the entropy as defined by:

        .. math::

            - \sum_{\{x\}}{xlog_2x}
        

        .. tip::

            If input is and n-dimensional array of either np.ndarray or list[list[..list]], this function will return
            an array of dimension n-1 with the entropy in each n-1 element of the input.
            
        
        .. runblock:: pycon

            >>> import numpy as np
            >>> from block_coding_tools.evaluation import entropy
            >>> A=np.array([[0.1, 0.2, 0.7], [0.33, 0.37, 0.3]]).T
            >>> print("channel 1 frequencies:", *A[0:,0])
            >>> print("channel 2 frequencies:", *A[0:,1])
            >>> for i, ent in enumerate(entropy(A, is_histogram=True)):
            ...     print(f"channel {i+1} entropy: {ent}")



        
        .. warning::

            It is up to the user of this function to know if the input is
            a valid normalised histogram!
    
    """
    if is_histogram:
        return -sum(x*np.log2(x) for x in x if not np.any(x==0))
    else:
        x = [relative_frequency for _, relative_frequency in histogram(x)]
        return entropy(x, is_histogram=True)


def peak_signal_to_noise(x, y, data_range) -> float:
    """

        Given signal x, a signal y, and the maximum signal magnitude, this function returns the
        **peak signal to noise ratio**.
        For an 1<n<2 dimensional signal pair x, y the PSNR is defined:


        .. math::

            MSE = \\frac{1}{MN}(\sum_{i=0}^{M-1}\sum_{j=0}^{N-1} [x(i,j)-y(i,j)]^2)
        
        .. math::

            PSNR = 10log_{10}\\frac{MAX_x ^2}{MSE}


        Example:

        .. runblock:: pycon

            >>> import numpy as np
            >>> from block_coding_tools.evaluation import peak_signal_to_noise as psnr
            >>> noise = np.random.rand(5,5)
            >>> x = np.ones((5,5))
            >>> print(psnr(x, x+noise, np.max(x+noise)-np.min(x+noise)))
            >>> print(psnr(x, x+noise/2, np.max(x+noise)-np.min(x+noise)))


    """
    square_error = np.power(x-y,2)
    return 10*np.log10(data_range**2 / np.mean(square_error))


def transform_coding_gain(coeff_vector: list[list]) -> float:
    """
            
        Given a list of lists, this function computes the transform coding gain
        as defined by:

        .. math::

            G_{TC} = \\frac{\\frac{1}{N}sum_{i=0}^{N-1}\sigma_i^2}{(\prod_i=0^{N-1})^{\\frac{1}{N}}} 

    """

    # The zeroth dimension (np.var(x, 0)) calculates the variance in each column
    channel_variance = np.var(coeff_vector, 0) 
    return 20*np.log10(np.mean(channel_variance)/np.prod(channel_variance)**(1/len(channel_variance)))