import numpy as np
from hydro_module import energy_slope_U


def compute_phis(tau, tf, C=None):
    """Computes the dimensionless transport capacity per unit width and its
    derivatives with respect to water depth (phiD) and Shields stress (phiT)
    given the Shields stress theta. Different transport formulas are implemented

    Parameters
    ----------
    tau : float or ndarray
        Shields stress
    tf : str
        String defining the transport formula adopted. Available options:
        "MPM" (Meyer-Peter & Mueller, 1947), "EH" (Engelund & Hansen, 1967),
        "P78" (Parker, 1978), "WP" (Wong&Parker, 2006).
    C : float, optional
        Chézy coefficient. Used only if `tf`="EH", by default None.

    Returns
    -------
    A tuple containing phi, phiD, and phiT, all in the form of a numpy array
    """
    if isinstance(tau, float) or isinstance(tau, int):
        tau = np.array([tau])  # to avoid errors with the boolean arrays
        return_floats = True
    else:
        return_floats = False

    phi = np.zeros(np.size(tau))
    phiD = np.zeros(np.size(tau))
    phiT = np.zeros(np.size(tau))
    if tf == "MPM":  # Meyer-Peter and Muller (1948)
        theta_cr = 0.047
        nst = tau < theta_cr
        phi[~nst] = 8 * (tau[~nst] - theta_cr) ** 1.5
        phiT[nst] = None
        phiT[~nst] = 1.5 * tau[~nst] / (tau[~nst] - theta_cr)
    elif tf == "EH":  # Engelund & Hansen
        if C is None:
            raise ValueError(
                "C must be provided when using the transport formula by Engelund & Hansen"
            )
        phi = 0.05 * C**2 * tau**2.5
        phiD[:] = 2 * 2.5 / C
        phiT[:] = 2.5
    elif tf == "P90":  # Parker (1990)
        c1 = 0.0386
        c2 = 0.853
        c3 = 5474
        c4 = 0.00218
        x = tau / c1
        phi[x >= 1] = (
            c4
            * (tau[x >= 1] ** 1.5)
            * (np.exp(14.2 * (x[x >= 1] - 1) - 9.28 * (x[x >= 1] - 1) ** 2))
        )
        phiT[x >= 1] = (
            tau[x >= 1]
            / phi[x >= 1]
            * (
                1 / c1 * phi[x >= 1] * (14.2 - 9.28 * 2 * (x[x >= 1] - 1))
                + 1.5 * phi[x >= 1] / tau[x >= 1]
            )
        )
        phi[x > 1.59] = c3 * c4 * tau[x > 1.59] ** 1.5 * (1 - c2 / x[x > 1.59]) ** 4.5
        phiT[x > 1.59] = (
            tau[x > 1.59]
            / phi[x > 1.59]
            * (
                1.5 * tau[x > 1.59] ** 0.5 * (1 - c2 / x[x > 1.59]) ** 4.5 * c3 * c4
                + 4.5
                * c1
                * c2
                * (1 - c2 / x[x > 1.59]) ** 3.5
                * c3
                * c4
                / tau[x > 1.59] ** 0.5
            )
        )
        phi[x < 1] = c4 * tau[x < 1] ** 1.5 * x[x < 1] ** 14.2
        phiT[x < 1] = (
            tau[x < 1]
            / phi[x < 1]
            * (
                14.2 / c1 * c4 * tau[x < 1] ** 1.5 * x[x < 1] ** 13.2
                + c4 * x[x < 1] ** 14.2 * 1.5 * tau[x < 1] ** 0.5
            )
        )
    elif tf == "P78":  # Parker (1978)
        theta_cr = 0.03
        nst = tau < theta_cr
        phi[~nst] = 11.2 * tau[~nst] ** 1.5 * (1 - theta_cr / tau[~nst]) ** 4.5
        phiT[~nst] = 1.5 + 4.5 * theta_cr / (tau[~nst] - theta_cr)
        phiT[nst] = None
    elif tf == "WP":  # Wong&Parker (2006)
        theta_cr = 0.0495
        nst = tau < theta_cr
        phi[~nst] = 3.97 * (tau[~nst] - theta_cr) ** 1.5
        phiT[nst] = None
        phiT[~nst] = 1.5 * tau[~nst] / (tau[~nst] - theta_cr)
    else:
        raise ValueError("Unknown transport formula")
    if return_floats:
        return phi[0], phiD[0], phiT[0]
    else:
        return phi, phiD, phiT


def compute_qs(tau, tf, Ds, Delta, C=None, g=9.81):
    """Computes the transport capacity per unit width.

    Parameters
    ----------
    tau : float or ndarray
        Shields stress.
    tf : str
        transport formula. See compute_phis for the list of available formulas.
    Ds : float
        sediment grain size
    Delta : float
        relative sediment density, typically set to 1.65
    C : float, optional
        Chézy coefficient. Used only if `tf`="EH", by default None.
    g : float, optional
        gravity acceleration, by default 9.81

    Returns
    -------
    ndarray
        transport capacity per unit width
    """
    return np.sqrt(g * Delta * Ds**3) * compute_phis(tau, tf, C=C)[0]


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
