#cython: boundscheck=False,
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython

cpdef void testing(unsigned char[:] data) nogil:
	with gil:
		print data
		print "HEYY"
	