import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import lsqr as sp_lsqr
from pylops import MatrixMult, Identity
from pylops.optimization.basic import lsqr
from pylops.utils.backend import get_array_module, get_module_name

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator

class L2_MC(ProxOperator):
    r"""
    This is customized pyproximal.L2 operator with modified method prox
    L2 Norm proximal operator to be able to calculate the following one:

    `f(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2 
    + 1/2\time_step ||\mathbf{x} - \mathbf{v}||^2`
    
    # !!!! in comparison to the original pyproximal.L2 they are added two more 
    # parameters:

    v : :obj:`numpy.ndarray`, optional
        Data vector
    time_step : `np.float`, optional

    """
    def __init__(self, Op=None, b=None, v=None, time_step=None, sigma=1., 
                 alpha=1., qgrad=True, niter=10, x0=None, warm=True,
                 densesolver=None, kwargs_solver=None):
        
        super().__init__(Op, True)
        self.b = b
        self.sigma = sigma
        self.alpha = alpha
        self.qgrad = qgrad
        self.niter = niter
        self.x0 = x0
        self.v = v
        self.time_step = time_step
        self.warm = warm
        self.densesolver = densesolver
        self.count = 0
        self.kwargs_solver = {} if kwargs_solver is None else kwargs_solver

        # when using factorize, store the first tau*sigma=0 so that the
        # first time it will be recomputed (as tau cannot be 0)
        if self.densesolver == 'factorize':
            self.tausigma = 0

        # create data term
        if self.Op is not None and self.b is not None:
            self.OpTb = self.sigma * self.Op.H @ self.b
            # create A.T A upfront for explicit operators

    def __call__(self, x):

        if self.Op is not None and self.b is not None \
            and self.v is not None and self.time_step is not None:
            f = (self.sigma / 2.) * (np.linalg.norm(self.Op * x - self.b) ** 2)
        #else 

        return f

    def _increment_count(func):
        """Increment counter
        """
        def wrapped(self, *args, **kwargs):
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x, tau):
        # define current number of iterations
        if isinstance(self.niter, int):
            niter = self.niter
        else:
            niter = self.niter(self.count)

        # solve proximal optimization
        if self.Op is not None and self.b is not None:

            y = x + tau * self.OpTb + tau/self.time_step * self.v

            Op1 = Identity(self.Op.shape[1], dtype=self.Op.dtype) * float(1+tau/self.time_step) + \
                    float(tau * self.sigma) * (self.Op.H * self.Op)
            if get_module_name(get_array_module(x)) == 'numpy':
                x = sp_lsqr(Op1, y, iter_lim=niter, x0=self.x0,
                            **self.kwargs_solver)[0]
            else:
                x = lsqr(Op1, y, niter=niter, x0=self.x0,
                            **self.kwargs_solver)[0].ravel()
                
            if self.warm:
                self.x0 = x
        elif self.b is not None:
            num = x + tau * self.sigma * self.b
            if self.q is not None:
                num -= tau * self.alpha * self.q
            x = num / (1. + tau * self.sigma)
        else:
            num = x
            if self.q is not None:
                num -= tau * self.alpha * self.q
            x = num / (1. + tau * self.sigma)
        return x

    def grad(self, x):
        if self.Op is not None and self.b is not None:
            g = self.sigma * self.Op.H @ (self.Op @ x - self.b)
        elif self.b is not None:
            g = self.sigma * (x - self.b)
        else:
            g = self.sigma * x
        if self.q is not None and self.qgrad:
            g += self.alpha * self.q
        return g