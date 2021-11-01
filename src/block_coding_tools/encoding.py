""" tools for entropy encoding

.. module:: encoding
   :platform: Unix, Windows
.. moduleauthor:: Thomas Berg Svendsen

"""
import numpy as np


Node = tuple
is_node = lambda x: type(x)==Node
left = lambda x, _: x
right = lambda _, x: x




def _make_node(fst, snd, probability):
    print(probability)
    return (probability, (fst,snd))


def generate_tree(x: list) -> list[list]:

    # sort key: the elements of x will either be float,
    # or tuple. If tuple, the combined probability will
    # be the first element of the tuple.
    probability = lambda x: x[0] if type(x)==tuple else x

    def _generate_huffman(x: list) -> list[list]:

        fst, *rest = sorted(x, key=probability)
        if rest==[]:
            return fst
        else:
            snd,*rest = rest
            combined_prob = probability(fst) + probability(snd)
            print
            node = _make_node(fst, snd, combined_prob)
            return _generate_huffman([node]+rest)
        
    return _generate_huffman(x)





def traverse(f, x):
    if is_node(x):
        x = f(*x)
        return traverse(f, x)
    return x

def _determine_compensation(i, grid):
    # in order for the branches of the binary tree to draw over each other
    # determine the offset needed to compensate
    compensation = 0
    for row in grid[i+1:]:
        if np.any(row):
            compensation += 1
        else:
            break
    return compensation


__syms = {0: " ", 2: "-", 3: '|'}
def view_tree(tree, i=0, j=0, grid=np.zeros((100,50))):

    node, tree = tree
    grid[i,j] = node
    if type(tree)==tuple:
        left, right = tree
        if type(left)==tuple:
            grid[i, j+1:j+4] = 2
            grid = view_tree(left, i=i, j=j+4, grid=grid)
        else:
            grid[i, j+1:j+4] = 2
            grid[i, j+4] = left


        if type(right)==tuple:
            i_step = _determine_compensation(i, grid)
            j_step = j if not j else j+3
            grid[i+1:(i+2+i_step), j_step] = 3
            grid[i+2+i_step, j_step:j_step+3] = 2
            grid = view_tree(right, i=i+2+i_step, j=j_step+3, grid=grid)
        else:
            grid[i+1, j+3] = 3
            grid[i+2, j+3] = right

    return grid


def print_grid(grid):
    for row in grid:
        if not np.any(row):
            return
        
        for col in row:
            if col in __syms.keys():
                print(__syms[col], sep="", end="")
            else: print(f'{col:.2f}', sep="", end="")
        print("\n",end="")




    
