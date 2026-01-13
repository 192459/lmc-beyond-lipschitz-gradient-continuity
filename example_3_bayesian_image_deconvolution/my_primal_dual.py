import time
import numpy as np

from pylops.utils.backend import get_array_module, to_numpy

def AdaptivePrimalDual(proxf, proxg, A, x0, tau, mu,
                       alpha=0.5, eta=0.95, s=1., delta=1.5,
                       z=None, niter=10, tol=1e-10, callback=None, show=False):
    r"""Adaptive Primal-dual algorithm

    Solves the minimization problem in
    :func:`pyproximal.optimization.primaldual.PrimalDual`
    using an adaptive version of the first-order primal-dual algorithm of [1]_.
    The main advantage of this method is that step sizes :math:`\tau` and
    :math:`\mu` are changing through iterations, improving the overall speed
    of convergence of the algorithm.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    A : :obj:`pylops.LinearOperator`
        Linear operator of g
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`
        Stepsize of subgradient of :math:`f`
    mu : :obj:`float`
        Stepsize of subgradient of :math:`g^*`
    alpha : :obj:`float`, optional
        Initial adaptivity level (must be between 0 and 1)
    eta : :obj:`float`, optional
        Scaling of adaptivity level to be multipled to the current alpha every
        time the norm of the two residuals start to diverge (must be between
        0 and 1)
    s : :obj:`float`, optional
        Scaling of residual balancing principle
    delta : :obj:`float`, optional
        Balancing factor. Step sizes are updated only when their ratio exceeds
        this value.
    z : :obj:`numpy.ndarray`, optional
        Additional vector
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    tol : :obj:`int`, optional
        Tolerance on residual norms
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    steps : :obj:`tuple`
        Tau, mu and alpha evolution through iterations

    Notes
    -----
    The Adative Primal-dual algorithm share the the same iterations of the
    original :func:`pyproximal.optimization.primaldual.PrimalDual` solver.
    The main difference lies in the fact that the step sizes ``tau`` and ``mu``
    are adaptively changed at each iteration leading to faster converge.

    Changes are applied by tracking the norm of the primal and dual
    residuals. When their mutual ratio increases beyond a certain treshold
    ``delta`` the step lenghts are updated to balance the minimization and
    maximization part of the overall optimization process.

    .. [1] T., Goldstein, M., Li, X., Yuan, E., Esser, R., Baraniuk, "Adaptive
        Primal-Dual Hybrid Gradient Methods for Saddle-Point Problems",
        ArXiv, 2013.

    """
    if show:
        tstart = time.time()
        print('Adaptive Primal-dual: min_x f(Ax) + x^T z + g(x)\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'Linear operator (A): %s\n'
              'Additional vector (z): %s\n'
              'tau0 = %10e\tmu0 = %10e\n'
              'alpha0 = %10e\teta = %10e\n'
              's = %10e\tdelta = %10e\n'
              'niter = %d\t\ttol = %10e\n' %
              (type(proxf), type(proxg), type(A),
               None if z is None else 'vector', tau, mu,
               alpha, eta, s, delta, niter, tol))
        head = '   Itn       x[0]          f           g          z^x       J = f + g + z^x'
        print(head)

    # initialization
    x = x0.copy()
    y = np.zeros(A.shape[0], dtype=x.dtype)
    Ax = np.zeros(A.shape[0], dtype=x.dtype)
    ATy = np.zeros(A.shape[1], dtype=x.dtype)
    taus = np.zeros(niter + 1)
    mus =  np.zeros(niter + 1)
    alphas = np.zeros(niter + 1)
    taus[0], mus[0], alphas[0] = tau, mu, alpha
    p = d = tol + 1.

    iiter = 0
    while iiter < niter and p > tol and d > tol:

        # store old values
        xold = x.copy()
        yold = y.copy()
        Axold = Ax.copy()
        ATyold = ATy.copy()

        # proxf
        if z is not None:
            ATy += z
        x = proxf.prox(x - tau * ATy, tau)
        Ax = A.matvec(x)
        Axhat = 2 * Ax - Axold

        # proxg
        y = proxg.proxdual(y + mu * Axhat, mu)
        ATy = A.rmatvec(y)

        # update steps
        if z is not None:
            p = np.linalg.norm((xold - x) / tau - (ATyold - ATy) -
                               A.rmatvec(z) + z)
        else:
            p = np.linalg.norm((xold - x) / tau - (ATyold - ATy))
        d = np.linalg.norm((yold - y) / mu - (Axold - Ax))

        if p > s * d * delta:
            tau /= 1 - alpha
            mu *= 1 - alpha
            alpha *= eta
        elif p < s * d / delta:
            tau *= 1 - alpha
            mu /= 1 - alpha
            alpha *= eta

        # save history of steps
        taus[iiter + 1] = tau
        mus[iiter + 1] = mu
        alphas[iiter + 1] = alpha
        iiter += 1

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf, pg = proxf(x), proxg(A.matvec(x))
                pf = 0. if type(pf) == bool else pf
                pg = 0. if type(pg) == bool else pg
                zx = 0. if z is None else np.dot(z, x)
                msg = '%6g  %12.5e  %10.3e  %10.3e  %10.3e      %10.3e' % \
                      (iiter + 1, np.real(to_numpy(x[0])), pf, pg, zx, pf + pg + zx)
                print(msg)

    steps = (taus[:iiter + 1], mus[:iiter + 1], alphas[:iiter + 1])
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))

    # print(p)
    # print(d)

    return x, steps
