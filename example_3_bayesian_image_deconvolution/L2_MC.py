import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import lsqr as sp_lsqr
from pylops import MatrixMult, Identity
from pylops.optimization.basic import lsqr
from pylops.utils.backend import get_array_module, get_module_name

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator

class L2_MC(ProxOperator):
    r"""L2 Norm proximal operator.

    The Proximal operator of the :math:`\ell_2` norm is defined as: :math:`f(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2`
    and :math:`f_\alpha(\mathbf{x}) = f(\mathbf{x}) +
    \alpha \mathbf{q}^T\mathbf{x}`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    b : :obj:`numpy.ndarray`, optional
        Data vector
    q : :obj:`numpy.ndarray`, optional
        Dot vector
    sigma : :obj:`int`, optional
        Multiplicative coefficient of L2 norm
    alpha : :obj:`float`, optional
        Multiplicative coefficient of dot product
    qgrad : :obj:`bool`, optional
        Add q term to gradient (``True``) or not (``False``)
    niter : :obj:`int` or :obj:`func`, optional
        Number of iterations of iterative scheme used to compute the proximal.
        This can be a constant number or a function that is called passing a
        counter which keeps track of how many times the ``prox`` method has
        been invoked before and returns the ``niter`` to be used.
    x0 : :obj:`np.ndarray`, optional
        Initial vector
    warm : :obj:`bool`, optional
        Warm start (``True``) or not (``False``). Uses estimate from previous
        call of ``prox`` method.
    densesolver : :obj:`str`, optional
        Use ``numpy``, ``scipy``, or ``factorize`` when dealing with explicit
        operators. The former two rely on dense solvers from either library,
        whilst the last computes a factorization of the matrix to invert and
        avoids to do so unless the :math:`\tau` or :math:`\sigma` paramets
        have changed. Choose ``densesolver=None`` when using PyLops versions
        earlier than v1.18.1 or v2.0.0
    **kwargs_solver : :obj:`dict`, optional
        Dictionary containing extra arguments for
        :py:func:`scipy.sparse.linalg.lsqr` solver when using
        numpy data (or :py:func:`pylops.optimization.solver.lsqr` and
        when using cupy data)

    Notes
    -----
    The L2 proximal operator is defined as:

    .. math::

        prox_{\tau f_\alpha}(\mathbf{x}) =
        \left(\mathbf{I} + \tau \sigma \mathbf{Op}^T \mathbf{Op} \right)^{-1}
        \left( \mathbf{x} + \tau \sigma \mathbf{Op}^T \mathbf{b} -
        \tau \alpha \mathbf{q}\right)

    when both ``Op`` and ``b`` are provided. This formula shows that the
    proximal operator requires the solution of an inverse problem. If the
    operator ``Op`` is of kind ``explicit=True``, we can solve this problem
    directly. On the other hand if ``Op`` is of kind ``explicit=False``, an
    iterative solver is employed. In this case it is possible to provide a warm
    start via the ``x0`` input parameter.

    When only ``b`` is provided, ``Op`` is assumed to be an Identity operator
    and the proximal operator reduces to:

    .. math::

        \prox_{\tau f_\alpha}(\mathbf{x}) =
        \frac{\mathbf{x} + \tau \sigma \mathbf{b} - \tau \alpha \mathbf{q}}
        {1 + \tau \sigma}

    If ``b`` is not provided, the proximal operator reduces to:

    .. math::

        \prox_{\tau f_\alpha}(\mathbf{x}) =
        \frac{\mathbf{x} - \tau \alpha \mathbf{q}}{1 + \tau \sigma}

    Finally, note that the second term in :math:`f_\alpha(\mathbf{x})` is added
    because this combined expression appears in several problems where Bregman
    iterations are used alongside a proximal solver.

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