






from block_coding_tools import encoding


# Encode source with 5 symbols:
coder = encoding.Coder.from_histogram([(0,0.4), (1,0.2), (2,0.15), (3,0.15), (4,0.10)], keep_histogram=True)

# Example image shape:
coder.set_source_shape((384,384))

print(coder.code_book)
print(coder)
coder.show_tree()
