#!/usr/bin/env python3
import pytest
import numpy as np
import time

@pytest.fixture
def test_data():
    """Create test data for speed tests"""
    np.random.seed(42)  # For reproducibility
    type = np.int8
    fts = np.random.randn(10**4, 10).astype(type)  # Smaller size for faster tests
    layer = np.random.randn(10, 10).astype(type)
    return fts, layer, type

def test_numpy_add(test_data):
    """Test numpy addition performance"""
    fts, _, type = test_data
    
    # Run a small warmup
    res = np.zeros(10, dtype=type)
    for v in fts[:100]:  # Use 100 iterations for warmup
        res = res + v
        
    # Actual test
    start = time.time()
    res = np.zeros(10, dtype=type)
    for v in fts:
        res = res + v
    elapsed = time.time() - start
    
    print(f"\nNumpy add time: {elapsed:.6f} seconds")
    assert len(res) == 10

def test_python_add(test_data):
    """Test pure Python addition performance"""
    fts, _, _ = test_data
    pyfts = fts.tolist()
    
    def pyadd(x, res):
        for i, xi in enumerate(x):
            res[i] += xi
    
    # Run a small warmup
    res = [0] * 10
    for v in pyfts[:100]:  # Use 100 iterations for warmup
        pyadd(v, res)
    
    # Actual test
    start = time.time()
    res = [0] * 10
    for v in pyfts:
        pyadd(v, res)
    elapsed = time.time() - start
    
    print(f"\nPython add time: {elapsed:.6f} seconds")
    assert len(res) == 10

def test_python_add_functional(test_data):
    """Test Python addition with functional style"""
    fts, _, _ = test_data
    pyfts = fts.tolist()
    
    def pyadd2(x, res):
        return [xi + ri for xi, ri in zip(x, res)]
    
    # Run a small warmup
    res = [0] * 10
    for v in pyfts[:100]:  # Use 100 iterations for warmup
        res = pyadd2(v, res)
    
    # Actual test
    start = time.time()
    res = [0] * 10
    for v in pyfts:
        res = pyadd2(v, res)
    elapsed = time.time() - start
    
    print(f"\nPython add functional time: {elapsed:.6f} seconds")
    assert len(res) == 10

def test_numpy_matmul(test_data):
    """Test numpy matrix multiplication performance"""
    fts, layer, type = test_data
    pyfts = fts.tolist()
    
    # Run a small warmup
    res = np.zeros(10, dtype=type)
    for v in pyfts[:100]:  # Use 100 iterations for warmup
        res += layer @ v
    
    # Actual test
    start = time.time()
    res = np.zeros(10, dtype=type)
    for v in pyfts:
        res += layer @ v
    elapsed = time.time() - start
    
    print(f"\nNumpy matmul time: {elapsed:.6f} seconds")
    assert len(res) == 10

def test_python_matmul(test_data):
    """Test pure Python matrix multiplication performance"""
    fts, layer, _ = test_data
    pyfts = fts.tolist()
    
    def matmul(m, v, res):
        for i in range(10):
            for j in range(10):
                res[i] += m[i][j] * v[j]
    
    # Run a small warmup
    res = [0] * 10
    for v in pyfts[:100]:  # Use 100 iterations for warmup
        matmul(layer, v, res)
    
    # Actual test
    start = time.time()
    res = [0] * 10
    for v in pyfts:
        matmul(layer, v, res)
    elapsed = time.time() - start
    
    print(f"\nPython matmul time: {elapsed:.6f} seconds")
    assert len(res) == 10

# Optionally test numba if available
@pytest.mark.skipif(True, reason="Optional test requiring numba")
def test_numba_add(test_data):
    """Test numba-accelerated addition if available"""
    try:
        from numba import guvectorize
        fts, _, type = test_data
        
        @guvectorize(['void(int8[:], int8[:])'], '(n)->(n)', nopython=True, fastmath=True)
        def add(x, res):
            res += x
        
        # Run a small warmup
        res = np.zeros(10, dtype=type)
        for v in fts[:100]:  # Use 100 iterations for warmup
            add(v, res)
        
        # Actual test
        start = time.time()
        res = np.zeros(10, dtype=type)
        for v in fts:
            add(v, res)
        elapsed = time.time() - start
        
        print(f"\nNumba add time: {elapsed:.6f} seconds")
        assert len(res) == 10
    except ImportError:
        pytest.skip("Numba not available")