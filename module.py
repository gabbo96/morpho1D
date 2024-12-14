import numpy as np
from morpholib.bedload import compute_qs


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
    j = energy_slope_U(U, C, Y, g=g)
    return j


def energy_slope_U(U, C, Y, g=9.81):
    j = U**2 / (C**2 * g * Y)
    return j


def shields_stress(j, Y, Delta, Ds):
    return j * Y / (Delta * Ds)


def C_eta(U, C, Y, Delta, Ds, tf, g=9.81, eps=1e-6):
    # Computes the celerity of the propagation of a perturbation of the bed elvel
    # (kinematic wave approximation)
    j_epsp = energy_slope_U(U + eps, C, Y)
    j_epsm = energy_slope_U(U - eps, C, Y)
    tau_epsp = shields_stress(j_epsp, Y, Delta, Ds)
    tau_epsm = shields_stress(j_epsm, Y, Delta, Ds)
    qs_epsp = compute_qs(tau_epsp, tf, Ds, Delta, C=C)
    qs_epsm = compute_qs(tau_epsm, tf, Ds, Delta, C=C)
    dqs_dU = (qs_epsp - qs_epsm) / (2 * eps)

    Fr = U / (g * Y)
    fc = Fr < 0.9 | Fr > 1.1  # far from critical conditions
    ncs = 0.9 <= Fr <= 1  # near critical conditions, larger than 1
    ncl = 1 < Fr <= 1.1  # near critical conditions, smaller than 1

    C_eta = np.zeros_like(Fr)
    C_eta[fc] = dqs_dU[fc] * U[fc] / Y[fc] / (1 - Fr[fc] ** 2)
    C_eta[ncs] = (
        U[ncl]
        * 0.5
        * (Fr[ncs] - 1 + np.sqrt((Fr[ncs] - 1) ** 2 + 2 * dqs_dU[ncs] / Y[ncs]))
    )
    C_eta[ncl] = (
        U[ncl]
        * 0.5
        * (Fr[ncl] - 1 - np.sqrt((Fr[ncl] - 1) ** 2 + 2 * dqs_dU[ncl] / Y[ncl]))
    )
    return C_eta


def compute_timestep(CFL, Q, B, C, Y, Delta, Ds, tf):
    dt = CFL * dx / C_eta_max
    return dt


def update_bed(qs, eta, cfl):
    eta_new = ...
    return eta_new
