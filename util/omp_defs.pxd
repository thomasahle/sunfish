from openmp cimport omp_lock_t, \
    omp_init_lock, omp_destroy_lock, \
    omp_set_lock, omp_unset_lock, omp_get_thread_num

cdef void acquire(omp_lock_t *l) nogil
cdef void release(omp_lock_t *l) nogil
cdef omp_lock_t *get_N_locks(int N) nogil
cdef void free_N_locks(int N, omp_lock_t *locks) nogil
