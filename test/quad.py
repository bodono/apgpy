#import sys
#sys.path.append("..")
#del sys.path[0]
#del sys.path[0]

import numpy as np
import apgpy as apg
from apgwrapper import NumpyWrapper
import matplotlib.pyplot as plt

n = 300
m = 1000
A = np.random.randn(m, n)
b = np.random.randn(m)
np.linalg.cond(A)

U, s, V = np.linalg.svd(A, full_matrices=True)
S = np.zeros((m, n))
S[:n, :n] = np.diag(s)
S = S**2

A = np.dot(U, np.dot(S, V))
np.linalg.cond(A)

AtA = np.dot(A.T, A)
Atb = np.dot(A.T, b)

def quad_grad(y):
    return NumpyWrapper(np.dot(AtA, y.nparray) - Atb)

x = apg.solve(quad_grad, {}, np.zeros(n), eps=1e-10)