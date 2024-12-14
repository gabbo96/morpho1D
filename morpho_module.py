import numpy as np
from hydro_module import energy_slope_U
from morpholib.bedload import compute_qs


def shields_stress(j, Y, Delta, Ds):
    return j * Y / (Delta * Ds)


def bed_celerity(U, C, Y, Delta, Ds, tf, g=9.81, eps=1e-6):
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


def compute_timestep(C_eta, CFL, dx):
    C_eta_max = np.max(np.abs(C_eta))
    dt = CFL * dx / C_eta_max
    return dt


def integrate_exner(eta, qs, qs0, dx, dt, C_eta, p):
    M = eta.size
    eta_new = np.zeros_like(eta)
    eta_new[0] = eta[0] - dt / dx * (qs[0] - qs0) / (1 - p)
    for i in range(1, M):
        if C_eta[i] > 0:
            eta_new[i] = eta[i] - dt / dx * (qs[i] - qs[i - 1]) / (1 - p)
        else:
            eta_new[i] = eta[i] - dt / dx * (qs[i + 1] - qs[i]) / (1 - p)
    return eta_new
