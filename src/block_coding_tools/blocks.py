""" Tools for deconstructing arrays into streams of blocks"""
from typing import Generator
import numpy as np





def decompose(image: np.array, block_size: int) -> Generator:
    """
        Returns:
            generator of side block_size blocks from image 
    """
    row_size, col_size = image.shape
    for x in range(0, row_size-block_size + 1, block_size):
        for y in range(0, col_size-block_size + 1, block_size):
            yield image[x:x+block_size, y:y+block_size]


def decompose_flat(image: np.array, block_size: int):
    """
        Returns:
            generator of side block_size blocks of flattened image.
    """
    row_size, col_size = image.shape
    for x in range(0, row_size - block_size + 1, block_size):
        for y in range(0, col_size-block_size + 1):
            yield image[y][x:x+block_size]


def compose(image_shape: tuple, block_size, blocks):
    row_size, col_size = image_shape 
    image = np.empty(image_shape)
    for x in range(0, row_size-block_size + 1, block_size ):
        for y in range(0, col_size-block_size + 1, block_size):
            image[x:x+block_size, y:y+block_size] = next(blocks)
    return image



def decompose_color(image: np.array, block_size: int):
    row_size, col_size = image_shape
    yield from decompose(image[:,:,0], block_size)
    yield from decompose(image[:,:,1], block_size)
    yield from decompose(image[:,:,2], block_size)




    
    
    

