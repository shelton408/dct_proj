#imports just to check that they run on hpc, https://github.com/ContinuumIO/gtc2017-numba for tutorial on numba
#mpi at https://mpi4py.readthedocs.io/en/stable/tutorial.html
import numba
import numpy as np
from Sequential.seq_dct import *
import timeit
import pandas as pd
import matplotlib.pyplot as plt


#list out different methods run in test_suite
#to check correctness, can compare anything to scipy dct
TEST_CASES = ['N', 
#'formula', 
'double 1d-dct', 
'matmul dct']
def test_suite(shape):
    times = []
    times.append(shape[0])#append N to the times for vizualization/graphing purposes

    random_mat = np.random.randint(10, size = shape)

    if 'formula' in TEST_CASES:
        result, t = formula_dct(random_mat)
        times.append(t)

    if 'double 1d-dct' in TEST_CASES:
        result, t = double_1d_dct(random_mat)
        times.append(t)

    if 'matmul dct' in TEST_CASES:
        result, t = matmul_dct(random_mat)
        times.append(t)

    # result, t = scipy_dct(random_mat)
    # times.append(t)
    return times

shapes = [[x,x] for x in [8,16,32,64, 128, 256]]
results = []
for shape in shapes:
    results.append(test_suite(shape))
table = pd.DataFrame(results)
table.columns = TEST_CASES
print(table)
ax = table.plot(kind='line', x='N', y=TEST_CASES[1])
for case in TEST_CASES[2:]:
    table.plot(kind='line', x='N', y=case, ax=ax)
plt.show()

