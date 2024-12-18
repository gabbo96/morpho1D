import numpy as np
from hydro_module import energy_slope_U
from morpholib.bedload import compute_qs


def shields_stress(j, Y, Delta, Ds):
    return j * Y / (Delta * Ds)


def bed_celerity(U, C, Y, Delta, Ds, tf, g=9.81, eps=1e-8):
    # Computes the celerity of the propagation of a perturbation of the bed elvel
    # (kinematic wave approximation)
    j_epsp = energy_slope_U(U + eps, C, Y)
    j_epsm = energy_slope_U(U - eps, C, Y)
    tau_epsp = shields_stress(j_epsp, Y, Delta, Ds)
    tau_epsm = shields_stress(j_epsm, Y, Delta, Ds)
    qs_epsp = compute_qs(tau_epsp, tf, Ds, Delta, C=C)
    qs_epsm = compute_qs(tau_epsm, tf, Ds, Delta, C=C)
    dqs_dU = (qs_epsp - qs_epsm) / (2 * eps)

    Fr = U / np.sqrt(g * Y)
    fc = (Fr < 0.9) | (Fr > 1.1)  # far from critical conditions
    ncs = (Fr >= 0.9) & (Fr <= 1)  # near critical conditions, larger than 1
    ncl = (Fr > 1) & (Fr <= 1.1)  # near critical conditions, smaller than 1

    C_eta = np.zeros_like(Fr)
    C_eta[fc] = dqs_dU[fc] * U[fc] / Y[fc] / (1 - Fr[fc] ** 2)
    C_eta[ncs] = (
        U[ncs]
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


def integrate_exner(eta, q_s, q_s0, dx, dt, C_eta, p):
    # Initialize eta_new with the same shape as eta
    eta_new = np.zeros_like(eta)
    eta_new[0] = eta[0] - dt / dx * (q_s[0] - q_s0) / (1 - p)
    eta_new[-1] = eta[-1] - dt / dx * (q_s[-1] - q_s[-2]) / (1 - p)

    # Create boolean arrays for the conditions
    Cpos = C_eta > 0
    Cneg = C_eta <= 0
    Cpos[0] = Cneg[0] = False
    Cpos[-1] = Cneg[-1] = False

    # Update eta_new based on the conditions
    eta_new[Cpos] = eta[Cpos] - dt / dx * (q_s[Cpos] - q_s[np.roll(Cpos, -1)]) / (1 - p)
    eta_new[Cneg] = eta[Cneg] - dt / dx * (q_s[np.roll(Cneg, 1)] - q_s[Cneg]) / (1 - p)
    return eta_new


def update_bed(
    eta, tf, Q, B, Y, C, q_s0, Delta, Ds, p, dx, CFL, dt_max=20, use_cfl=False
):
    U = Q / (B * Y)
    j = energy_slope_U(U, C, Y)
    taus = shields_stress(j, Y, Delta, Ds)
    q_s = compute_qs(taus, tf, Ds, Delta, C=C)
    C_eta = bed_celerity(U, C, Y, Delta, Ds, tf)
    dt_comp = compute_timestep(C_eta, CFL, dx)
    if use_cfl:
        dt = dt_comp
    else:
        dt = dt_max
    eta_new = integrate_exner(eta, q_s, q_s0, dx, dt, C_eta, p)
    return dt_comp, eta_new
