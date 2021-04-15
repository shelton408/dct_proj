#takes numpy array and calculates dct using sequential formula
#we are assuming its a square matrix as described in the project

import numpy as np
import math
import time
from scipy.fftpack import dct

z_norm = 1/math.sqrt(2)

def formula_dct(mat):
    begin = time.time()

    y,x = mat.shape #x=y if square
    dct_mat = np.zeros(mat.shape)
    normalizer = math.sqrt(2/x) * math.sqrt(2/y)
    for k1 in range(0, x):
        for k2 in range(0, y):
            for n1 in range(0, x):
                for n2 in range(0, y):
                    dct_mat[k1, k2] += mat[n1, n2]*np.cos((np.pi/x)*(n1 + .5)*k1)*np.cos((np.pi/y)*(n2 + .5)*k2)
    result = dct_mat * normalizer

    for k in range(0,x):
        result[0][k] *= z_norm
    for k in range(0,y):
        result[k][0] *= z_norm

    end = time.time()
    t = end-begin
    return result, t

def oned_dct(mat):
    x = mat.shape[0]
    dct_mat = np.zeros(mat.shape[0])
    for k in range(0,x):
        for n in range(0,x):
            dct_mat[k] +=mat[n]*np.cos((np.pi/x)*(n + .5)*k)
    for k in range(0,x):
        dct_mat[k] = dct_mat[k]*math.sqrt(2/x)
    dct_mat[0] = dct_mat[0]*z_norm
    print(dct(mat, type=2, norm='ortho'))
    print(dct_mat)

def double_1d_dct(mat):
    begin = time.time()
    
    y,x = mat.shape #x=y if square
    dct_mat = np.zeros(mat.shape)
    dct_mat2 = np.zeros(mat.shape)
    normalizer = math.sqrt(2/x) * math.sqrt(2/y)
    for y_ in range(0,y):
        for k in range(0,x):
            for n in range(0,x):
                dct_mat[y_, k] +=mat[y_, n]*np.cos((np.pi/x)*(n + .5)*k)
    for i in range(0,y):
        for j in range(0,x):
            dct_mat[i,j] = dct_mat[i,j] * math.sqrt(2/x)*l(j)

    for x_ in range(0,x):
        for k in range(0,y):
            for n in range(0,y):
                dct_mat2[k, x_] +=dct_mat[n, x_]*np.cos((np.pi/y)*(n + .5)*k)
    for i in range(0,y):
        for j in range(0,x):
            dct_mat2[i,j] = dct_mat2[i,j] * math.sqrt(2/y)*l(i)

    result = dct_mat2

    end = time.time()
    t = end - begin
    return result, t

def matmul_dct(mat):
    begin = time.time()
    y, x = mat.shape
    T = np.zeros(mat.shape)
    for j in range(x):
        T[0, j] = 1/math.sqrt(x)
    for i in range(1, y):
        for j in range(x):
            T[i,j] = math.sqrt(2/x)*np.cos((2*j+1)*i*np.pi/(2*x))
    
    #avoid np.matmul since it can be automatically parallelized
    #transpose may be optimized using built in but the cost of transpose shouldnt affect the order of computation
    #matmul in O(N^3)
    dct_mat = matmul(matmul(T, mat), T.T)

    end = time.time()
    t = end-begin
    return dct_mat, t

def matmul(a, b):
    C = np.zeros((a.shape[0], b.shape[1]))

    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(b.shape[0]):
                C[i,j] += a[i,k] * b[k,j]
    
    return C
    

#lambda, not used in our case
def l(x):
    if(x==0):
        return z_norm
    return 1

#to check correctness and also time refactorization method
def scipy_dct(mat):
    begin = time.time()

    dct_mat = dct(dct(mat, type=2, norm='ortho').T, type=2, norm='ortho').T

    end = time.time()
    t = end - begin
    return dct_mat, t