import numpy as np
dim1, dim2 = 21, 18
table_table = np.random.randn(64, 6)
comb = np.random.randn(6, 6, dim)
white = np.random.randn(dim, dim2)
black = np.random.randn(dim, dim2)
bias = np.random.randn(dim2)
last = np.random.randn(dim2, 1)
# Castling really should be part of the white/black part, no?
cstl = np.random.randn(4, 1)
# Saving and loading should be done like this:
# https://numpy.org/doc/stable/reference/generated/numpy.savez.html
# Remember to pad to 12x10: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
