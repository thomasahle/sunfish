import time
import numpy as np

type = np.int8

fts = np.random.randn(10**5, 10).astype(type)
layer = np.random.randn(10, 10).astype(type)

try:
    from numba import guvectorize
    @guvectorize(['void(int8[:], int8[:])'], '(n)->(n)', nopython=True, fastmath=True)
    def add(x, res):
        res += x
    # Do some warmup first
    res = np.zeros(10, dtype=type)
    for v in fts:
        add(v, res)
    # Then do it for real
    start = time.time()
    res = np.zeros(10, dtype=type)
    for v in fts:
        add(v, res)
    print(res)
    print('Numba time:', time.time() - start)
except ImportError:
    print('No numba')

# Do some warmup first
res = np.zeros(10, dtype=type)
for v in fts:
    res = res + v
# Then do it for real
start = time.time()
res = np.zeros(10, dtype=type)
for v in fts:
    res = res + v
print(res)
print('Numpy time:', time.time() - start)

def pyadd(x, res):
    for i, xi in enumerate(x):
        res[i] += xi
pyfts = fts.tolist()
# Do some warmup first
res = [0] * 10
for v in pyfts:
    pyadd(v, res)
# Then do it for real
start = time.time()
res = [0] * 10
for v in pyfts:
    pyadd(v, res)
print(res)
print('Python time:', time.time() - start)

def pyadd2(x, res):
    return [xi + ri for xi, ri in zip(x, res)]
pyfts = fts.tolist()
# Do some warmup first
res = [0] * 10
for v in pyfts:
    res = pyadd2(v, res)
# Then do it for real
start = time.time()
res = [0] * 10
for v in pyfts:
    res = pyadd2(v, res)
print(res)
print('Python time:', time.time() - start)


res = np.zeros(10, dtype=type)
for v in pyfts:
    res += layer @ v
start = time.time()
res = np.zeros(10, dtype=type)
for v in pyfts:
    res += layer @ v
print('Nupy time, matmul:', time.time() - start)


def matmul(m, v, res):
    for i in range(10):
        for j in range(10):
            res[i] += m[i][j] * v[j]
    #for i, mr in enumerate(m):
        #res[i] += sum(mri * vi for mri, vi in zip(mr, v))
res = [0] * 10
for v in pyfts:
    #pyadd(matmul(layer, v), res)
    matmul(layer, v, res)
start = time.time()
res = [0] * 10
for v in pyfts:
    #pyadd(matmul(layer, v), res)
    matmul(layer, v, res)
print('Python time, matmul:', time.time() - start)
