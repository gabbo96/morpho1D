import numpy as np


def initial_conditions(in_d):
    # Build width and bed elevation arrays combining the input values
    # of the different reaches
    x = np.array([0])
    eta = np.array([0])
    B = np.array([in_d["reaches"][0]["B"]])
    for reach in in_d["reaches"]:
        M_reach = int(reach["L"] / in_d["dx"]) + 1
        x = np.append(
            x[:-1], np.linspace(x[-1], x[-1] + (M_reach - 1) * in_d["dx"], M_reach)
        )
        eta = np.append(
            eta[:-1], np.linspace(eta[-1], eta[-1] - reach["S"] * reach["L"], M_reach)
        )
        if isinstance(reach["B"], list):
            B = np.append(B[:-1], np.linspace(reach["B"][0], reach["B"][1], M_reach))
        else:
            B = np.append(B[:-1], np.ones(M_reach) * reach["B"])

    S = -(eta[1:] - eta[:-1]) / (x[1:] - x[:-1])
    S = np.append(S, S[-1])
    return x, eta, B, S
