apgpy
=====================
(@author bodonoghue) python package 
Implements an Accelerated Proximal Gradient method
(Nesterov 2007, Beck and Teboulle 2009)

solves: 

    minimize f(x) + h(x)
    over x \in R^dim_x

where `f` is smooth, convex - user supplies function to evaluate gradient of `f`  
`h` is convex - user supplies function to evaluate the proximal operator of `h`

call as:

   ` x = apgpy.solve( grad_f, prox_h, dim_x ) `

`solve` has call signature:

```
def solve(grad_f, prox_h, dim_x,
          max_iters=2000,
          eps=1e-6,
          alpha=1.01,
          beta=0.5,
          gen_plots=True,
          use_restart=True,
          x_init=False,
          quiet=False,
          use_gra=False,
          step_size=False,
          fixed_step_size=False,
          debug=False)
```

this takes in two functions:

    grad_f(v) = df(v)/dv 
    (gradient of f at v)
    
    prox_h(v, t) = argmin_x ( t * h(x) + 1/2 * norm(x-v)^2 )
    where t is the (scalar, positive) step size at that iteration


each iteration of `apg` requires one gradient evaluation of `f` and one prox step with `h`

quits when:
    
    norm( y(k) - x(k+1) ) / t < eps * max( 1,norm( x(k+1) ) 

`apg` implements something similar to TFOCS step-size adaptation (Becker, Candes and Grant 2010)  
and gradient-scheme adaptive restarting ([O'Donoghue and Candes 2013](http://bodonoghue.org/publications/adap_restart.pdf))

see notebooks/ for usage instances

