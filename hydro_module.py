import numpy as np
from bisection import bisection


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


def compute_profile(M, h_ups0, h_ds0, Q, B, C, Y_cr, eta, x):
    Y = np.zeros(M)
    Y_sub = np.zeros(M)
    Y_super = np.zeros(M)
    H_sub = np.zeros(M)
    H_super = np.zeros(M)

    E_min = energy(Y_cr, Q, B, eta) - eta
    # ------------------ First spatial loop: upstream direction ------------------ #
    # Downstream BC
    Y_sub[-1] = h_ds0 - eta[-1]
    if froude(Q, B[-1], Y_sub[-1]) > 1:
        Y_sub[-1] = Y_cr[-1]
    H_sub[-1] = energy(Y_sub[-1], Q, B[-1], eta[-1])

    # Spatial loop in upstream direction
    for i in range(M - 2, -1, -1):
        dx = x[i + 1] - x[i]
        j = energy_slope(Q, B[i + 1], C, Y_sub[i + 1])
        H_sub[i] = H_sub[i + 1] + j * dx

        # Check if the specific energy is lower than the minimum for the given flow discharge
        if H_sub[i] - eta[i] < E_min[i]:
            Y_sub[i] = Y_cr[i] + 0.05
            H_sub[i] = energy(Y_sub[i], Q, B[i], eta[i])
        else:
            YL = Y_cr[i]
            YR = 10 * Y_cr[i]
            Y_sub[i] = bisection(energy_bisec_f, YL, YR, (H_sub[i], Q, B[i], eta[i]))

    # ----------------- Second spatial loop: downstream direction ---------------- #
    # Upstream BC
    Y_super[0] = h_ups0 - eta[0]
    if froude(Q, B[0], Y_super[0]) < 1:
        Y_super[0] = Y_cr[0]
    H_super[0] = energy(Y_super[0], Q, B[0], eta[0])

    # Spatial loop in downstream direction
    for i in range(M - 1):
        dx = x[i + 1] - x[i]
        j = energy_slope(Q, B[i], C, Y_super[i])
        H_super[i + 1] = H_super[i] - j * dx
        # Check if the specific energy is lower than the minimum for the given flow discharge
        if H_super[i + 1] - eta[i + 1] < E_min[i + 1]:
            Y_super[i + 1] = Y_cr[i + 1] - 0.05
            H_super[i + 1] = energy(Y_super[i + 1], Q, B[i + 1], eta[i + 1])
        else:
            YL = 1e-6
            YR = Y_cr[i + 1]
            Y_super[i + 1] = bisection(
                energy_bisec_f, YL, YR, (H_super[i + 1], Q, B[i + 1], eta[i + 1])
            )

    # -------------------------- Final profile retrieval ------------------------- #
    S_sub = spinta(Q, B, Y_sub)
    S_super = spinta(Q, B, Y_super)
    Y[S_sub >= S_super] = Y_sub[S_sub >= S_super]
    Y[S_sub < S_super] = Y_super[S_sub < S_super]
    return Y
