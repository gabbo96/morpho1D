import numpy as np


def energy_bisec_f(Y, H_target, Q, B, eta):
    """Function to be given as input to the bisection() function.

    Parameters
    ----------
    Y : float
        water depth
    H_target : float
        _description_
    Q : float
        _description_
    B : float
        _description_
    eta : float
        _description_

    Returns
    -------
    res
        difference between the energy for the given water depth and the target energy
    """
    return energy(Y, Q, B, eta) - H_target


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


def froude(Q, B, Y, g=9.81):
    Fr = Q / (B * Y * (g * Y) ** 0.5)
    return Fr


def energy(Y, Q, B, eta, g=9.81):
    U = Q / (B * Y)
    H = eta + Y + U**2 / (2 * g)
    return H


def spinta(Q, B, Y, rho=1000, g=9.81):
    S = 0.5 * rho * g * B * Y**2 + rho * Q**2 / (B * Y)
    return S


def uniflow_depth(Q, B, S, C, g=9.81):
    Y_u = (Q / (B * C * (g * S) ** 0.5)) ** (2 / 3)
    return Y_u


def crit_depth(Q, B, g=9.81):
    return (Q**2 / (g * B**2)) ** (1 / 3)


def bed_IC(S, L, num_nodes, eta_ups=100):
    eta = np.linspace(eta_ups, eta_ups - S * L, num_nodes)
    return eta


def energy_slope(Q, B, C, Y, g=9.81):
    U = Q / (B * Y)
    j = U**2 / (C**2 * g * Y)
    return j


def shields_stress():
    return ...


def phis(theta, tf, C):
    """Computes the dimensionless transport capacity per unit width (phi)
    given the Shields stress (theta). Different transport formulas are implemented

    Parameters
    ----------
    `theta` : ndarray
        Shields stress
    `tf` : str
        String defining the transport formula adopted. Available options:
        "MPM" (Meyer-Peter & Mueller, 1947), "EH" (Engelund & Hansen, 1967),
        "P78" (Parker, 1978), "WP" (Wong&Parker, 2006).
    `C` : float
        Chézy coefficient. Used only if `tf`="EH"

    Returns
    -------
    `phi` : ndarray
        Dimensionless transport capacity per unit width.
    """
    phi = np.zeros(np.size(theta))
    if tf == "MPM":  # Meyer-Peter and Muller (1948)
        theta_cr = 0.047
        nst = theta < theta_cr
        phi[~nst] = 8 * (theta[~nst] - theta_cr) ** 1.5
    elif tf == "EH":  # Engelund & Hansen
        phi = 0.05 * C**2 * theta**2.5
    elif tf == "P90":  # Parker (1990)
        c1 = 0.0386
        c2 = 0.853
        c3 = 5474
        c4 = 0.00218
        x = theta / c1
        phi[x >= 1] = (
            c4
            * (theta[x >= 1] ** 1.5)
            * (np.exp(14.2 * (x[x >= 1] - 1) - 9.28 * (x[x >= 1] - 1) ** 2))
        )
        phi[x > 1.59] = c3 * c4 * theta[x > 1.59] ** 1.5 * (1 - c2 / x[x > 1.59]) ** 4.5
        phi[x < 1] = c4 * theta[x < 1] ** 1.5 * x[x < 1] ** 14.2
    elif tf == "P78":  # Parker (1978)
        theta_cr = 0.03
        nst = theta < theta_cr
        phi[~nst] = 11.2 * theta[~nst] ** 1.5 * (1 - theta_cr / theta[~nst]) ** 4.5
    elif tf == "WP":  # Wong&Parker (2006)
        theta_cr = 0.0495
        nst = theta < theta_cr
        phi[~nst] = 3.97 * (theta[~nst] - theta_cr) ** 1.5
    else:
        print("error: unknown transport formula")
        phi = None
    return phi
