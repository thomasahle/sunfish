#cython: boundscheck=False,
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython

cpdef void testing(np.int32_t[:, :] board) nogil:
	with gil:
		print(board)
		print("ayy")
		print("HEYY")
	