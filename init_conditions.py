import numpy as np


def bed_IC(S, L, num_nodes, eta_ups=100):
    eta = np.linspace(eta_ups, eta_ups - S * L, num_nodes)
    return eta


def initial_conditions(in_d):
    # Build width and bed elevation arrays combining the input values
    # of the different reaches
    x = np.array([0])
    eta = np.array([0])
    B = np.array(in_d["reaches"][0]["B"])
    for reach in in_d["reaches"]:
        M_reach = int(reach["L"] / in_d["dx"])
        B = np.append(B, np.ones(M_reach - 1) * reach["B"])
        x = np.append(
            x, np.linspace(x[-1], x[-1] + (M_reach - 1) * in_d["dx"], num=M_reach)[1:]
        )
        eta = np.append(
            eta, bed_IC(reach["S"], reach["L"], M_reach, eta_ups=eta[-1])[1:]
        )

    S = -(eta[1:] - eta[:-1]) / (x[1:] - x[:-1])
    S = np.append(S, S[-1])
    return x, eta, B, S
