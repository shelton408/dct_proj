#imports just to check that they run on hpc, https://github.com/ContinuumIO/gtc2017-numba for tutorial on numba
#mpi at https://mpi4py.readthedocs.io/en/stable/tutorial.html
import numba
import numpy as np
from Sequential.seq_dct import *
import timeit
import pandas as pd


#list out different methods run in test_suite
#to check correctness, can compare anything to scipy dct
TEST_CASES = ['scipy (refac)', 'seq by formula']
def test_suite(shape):
    times = []
    random_mat = np.random.randint(10, size = shape)

    result, t = scipy_dct(random_mat)
    times.append(t)

    result, t = seq_dct(random_mat)
    times.append(t)

    return times

shapes = [[x,x] for x in range(8,56,16)]
results = [TEST_CASES]
for shape in shapes:
    results.append(test_suite(shape))
table = pd.DataFrame(results)
print(table)
