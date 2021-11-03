""" tools for entropy encoding

.. module:: encoding
   :platform: Unix, Windows
.. moduleauthor:: Thomas Berg Svendsen

"""
import numpy as np


def _make_node(fst, snd, probability):

    return (probability, (fst,snd))



def generate_tree(x: list) -> list[list]:

    # sort key: the elements of x will either be float,
    # or tuple. If tuple, the combined probability will
    # be the first element of the tuple.
    probability = lambda x: x[0] if type(x)==tuple else x

    def _generate_huffman(x: list) -> tuple[tuple]:

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



def _intermediate_code_book(tree: tuple[tuple], code=[],code_book=[]) -> dict:
    
    _, tree = tree
    if type(tree)==tuple:
        left, right = tree
        if type(left)==tuple:
            code.append('1')
            code_book = _intermediate_code_book(left, code=code, code_book=code_book)
            code.pop()
        else:
            code.append('1')
            
            # account for multiple symbols having the same probability
            code_book.append(("".join(b for b in code), left))     
            code.pop()
            
        if type(right)==tuple:
            code.append('0')
            code_book = _intermediate_code_book(right, code=code, code_book=code_book)
            code.pop()
        else:
            code.append('0')
           
            code_book.append(("".join(b for b in code), right))
            
            code.pop()
    #_ = code or code.pop()        
    return code_book
      
        
def generate_code_book(hist: list[tuple]) -> dict:
    
    tree = generate_tree([prob for _, prob in hist])
    hist = sorted(hist, key=lambda x: x[1])
    
    tmp_code_book = _intermediate_code_book(tree, code=[], code_book=[])
    
    # Sort the code book so the correct code is assigned to each symbol
    tmp_code_book.sort(key=lambda x: x[1])
    
    return {code[0]: symbol for code, symbol in zip(tmp_code_book, [symbol for symbol, _ in hist])}
        
        
class Coder:

    def __init__(self, code_book):
        self.code_book = code_book
        self.histogram = []
    
    def __str__(self):
        tree = self.tree()
        n_codes = len(self.code_book)
        mbps = get_mbps(tree)
        longest_code, *_ = self.code_book.keys()
        estimated_header = self.estimate_header_size()
        return f"Coder with:\n\tnumber of codes: {n_codes}\n\taverage code length: {mbps:.3f}\n\tMax code length: {len(longest_code)}\n\tEstimated header size: {estimated_header}\n\tEstimated filesize: {estimated_header + 50 + mbps*np.prod(self.source_shape)} bits"
        
    def tree(self):
        return generate_tree([prob for _, prob in self.histogram])
    
    def estimate_header_size(self):
        length_field = np.ceil(np.log2(self.max_len()))
        return len(self.code_book)*length_field 
    
    def mbps(self):
        return get_mbps(self.tree())
        
    def max_len(self):
        longest_code,*_ = self.code_book.keys()
        return len(longest_code) 
        
    def set_source_shape(self,shape):
        self.source_shape = shape
        
    def show_tree(self, hist=None):
        if hist:
            tree = generate_tree([prob for _, prob in hist])
            view_tree(tree)
            
        elif self.histogram:
            tree = self.tree()
            view_tree(tree)

            
        else: print("No histogram: cant generate tree")
        
        return
     

            
    @classmethod
    def from_histogram(cls, hist: list[tuple], keep_histogram=False):
        code_book = generate_code_book(hist)
        coder = cls(code_book)
        if keep_histogram:
            coder.histogram = hist
        return coder
    

def get_nodes(tree, nodes=[]):
    node, tree = tree
    nodes.append(node)
    if type(tree)==tuple:
        left, right = tree
        if type(left)==tuple:
            nodes = get_nodes(left, nodes=nodes)
        if type(right)==tuple:
            nodes = get_nodes(right, nodes=nodes)
        
    return nodes
    

def get_mbps(tree, acc=0):
    node, tree = tree
    acc+=node
    if type(tree)==tuple:
        left, right = tree
        if type(left)==tuple:
            acc = get_mbps(left, acc=acc)
        if type(right)==tuple:
            acc = get_mbps(right, acc=acc)
    return acc
    




# Helper for _make_tree_grid
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
    
    
# Helper for view_tree
__syms = {0: " ", 2: "-", 3: '|'}
def _make_tree_grid(tree, i=0, j=0, grid=np.zeros((100,50))):
    # THIS THING IS FRAGILE.
    node, tree = tree
    grid[i,j] = node
    if type(tree)==tuple:
        left, right = tree
        if type(left)==tuple:
            grid[i, j+1:j+4] = 2
            grid = _make_tree_grid(left, i=i, j=j+4, grid=grid)
        else:
            grid[i, j+1:j+4] = 2
            grid[i, j+4] = left


        if type(right)==tuple:
            i_step = _determine_compensation(i, grid)
            j_step = 0 if not j else 3
            grid[i+1:(i+2+i_step), j_step] = 3
            grid[i+2+i_step, j_step:j_step+3] = 2
            grid = _make_tree_grid(right, i=i+2+i_step, j=j_step+4, grid=grid)
        else:
            grid[i+1, j+3] = 3
            grid[i+2, j+3] = right

    return grid


def view_tree(tree, grid=np.zeros((100,50)), show=None):
    if show is None:
        i_lim, j_lim = grid.shape
    else:
        i_lim, j_lim = show

    grid = _make_tree_grid(tree, grid=grid)
    for row in grid:
        if not np.any(row):
            return
        
        for col in row:
            if col in __syms.keys():
                print(__syms[col], sep="", end="")
            else: print(f'{col:.2f}', sep="", end="")
        print("\n",end="")




    
