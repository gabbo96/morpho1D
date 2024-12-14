def bisection(func, a, b, args, tol=1e-7, max_iter=1000):
    """Find the root of a function using the bisection method.

    Parameters:
    ----------
    `func` : function
        The function for which we are trying to find a root.
    `a` : float
        The lower bound of the interval.
    `b` : float
        The upper bound of the interval.
    `args` : tuple
        Arguments of the function `func`, other than the first
    `tol` : float
        The tolerance for the root (default is 1e-7).
    `max_iter` : int
        The maximum number of iterations (default is 1000).

    Returns:
    -------
    float: The approximate root of the function.
    """
    if func(a, *args) * func(b, *args) >= 0:
        raise ValueError(
            "The function must have different signs at the endpoints a and b."
        )

    for _ in range(max_iter):
        c = (a + b) / 2
        if func(c, *args) == 0 or (b - a) / 2 < tol:
            return c
        if func(c, *args) * func(a, *args) < 0:
            b = c
        else:
            a = c

    raise ValueError("Maximum number of iterations reached without convergence.")
