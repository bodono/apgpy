import numpy as np
import apgpy as apg

n = 2000
m = 5000
A = np.random.randn(m, n)
b = np.random.randn(m)
mu = 1.

#U, s, V = np.linalg.svd(A, full_matrices=True)
#S = np.zeros((m, n))
#S[:n, :n] = np.diag(s)
#S **= 2
#A = np.dot(U, np.dot(S, V))
#np.linalg.cond(A)

AtA = np.dot(A.T, A)
Atb = np.dot(A.T, b)

def quad_grad(y):
    return np.dot(AtA, y) - Atb

def soft_thresh(y, t):
    return np.sign(y) * np.maximum(abs(y) - t * mu, 0.)

x = apg.solve(quad_grad, soft_thresh, np.zeros(n), eps=1e-8, gen_plots=False, quiet=True)
#plt.show()