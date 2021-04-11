#takes numpy array and calculates dct using sequential formula
#we are assuming its a square matrix as described in the project

import numpy as np
import math
import time
from scipy.fftpack import dct

z = 1/math.sqrt(2)

def seq_dct(mat):
    begin = time.time()

    y,x = mat.shape #x=y if square
    dct_mat = np.zeros(mat.shape)
    normalizer = math.sqrt(2/x) * math.sqrt(2/y)
    for k1 in range(0, x):
        for k2 in range(0, y):
            for n1 in range(0, x):
                for n2 in range(0, y):
                    dct_mat[k1, k2] += mat[n1, n2]*np.cos((np.pi/x)*(n1 + .5)*k1)*np.cos((np.pi/y)*(n2 + .5)*k2)
    dct_mat = dct_mat * normalizer

    end = time.time()
    t = end-begin
    return dct_mat, t

def matmul_dct(mat):
    return

#lambda, not used in our case
def l(x):
    if(x!=0):
        return 1
    return z

#to check correctness and also time refactorization method
def scipy_dct(mat):
    begin = time.time()

    dct_mat = dct(dct(mat, type=2, norm='ortho').T, type=2, norm='ortho').T

    end = time.time()
    t = end - begin
    return dct_mat, t