from __future__ import print_function
import numpy as np
from apgwrapper import NumpyWrapper
from functools import partial

def npwrap(x):
    if isinstance(x, np.ndarray):
        return NumpyWrapper(x)
    return x

def npwrapfunc(f, *args):
    return npwrap(f(*args))

def solve(grad_f, prox_h, x_init,
          max_iters=2500,
          eps=1e-6,
          alpha=1.01,
          beta=0.5,
          use_restart=True,
          gen_plots=False,
          quiet=False,
          use_gra=False,
          step_size=False,
          fixed_step_size=False,
          debug=False):

    df = partial(npwrapfunc, grad_f)
    ph = partial(npwrapfunc, prox_h)

    x_init = npwrap(x_init)

    x = x_init.copy()
    y = x.copy()
    g = df(y.data)
    theta = 1.

    if not step_size:
        # barzilai-borwein step-size initialization:
        t = 1. / g.norm()
        x_hat = x - t * g
        g_hat = df(x_hat.data)
        t = abs((x - x_hat).dot(g - g_hat) / (g - g_hat).norm() ** 2)
    else:
        t = step_size

    if gen_plots:
        errs = np.zeros(max_iters)

    k = 0
    err1 = np.nan
    iter_str = 'iter num %i, norm(Gk)/(1+norm(xk)): %1.2e, step-size: %1.2e'
    for k in range(max_iters):

        if not quiet and k % 100 == 0:
            print(iter_str % (k, err1, t))

        x_old = x.copy()
        y_old = y.copy()

        x = y - t * g

        if prox_h:
            x = ph(x.data, t)

        err1 = (y - x).norm() / (1 + x.norm()) / t

        if gen_plots:
            errs[k] = err1

        if err1 < eps:
            break

        if not use_gra:
            theta = 2. / (1 + np.sqrt(1 + 4 / (theta ** 2)))
        else:
            theta = 1.

        if not use_gra and use_restart and (y - x).dot(x - x_old) > 0:
            if debug:
                print('restart, dg = %1.2e' % (y - x).dot(x - x_old))
            x = x_old.copy()
            y = x.copy()
            theta = 1.
        else:
            y = x + (1 - theta) * (x - x_old)

        g_old = g.copy()
        g = df(y.data)

        # tfocs-style backtracking:
        if not fixed_step_size:
            t_old = t
            t_hat = 0.5 * ((y - y_old).norm() ** 2) / abs((y - y_old).dot(g_old - g))
            t = min(alpha * t, max(beta * t, t_hat))
            if debug:
                if t_old > t:
                    print('back-track, t = %1.2e, t_old = %1.2e, t_hat = %1.2e' % (t, t_old, t_hat))

    if not quiet:
        print(iter_str % (k, err1, t))
        print('terminated')
    if gen_plots:
        import matplotlib.pyplot as plt
        errs = errs[1:k]
        plt.figure()
        plt.semilogy(errs[1:k])
        plt.xlabel('iters')
        plt.title('||Gk||/(1+||xk||)')
        plt.draw()

    return x.data
