import os
import json
import matplotlib.pyplot as plt
from module import *

inputs_dir = "inputs_files"
inputs_filename = "slope_decrease"

bed_color = "goldenrod"
water_color = "dodgerblue"

# Import input dictionary
with open(os.path.join(inputs_dir, f"{inputs_filename}.json"), "r") as f:
    in_d = json.load(f)
Q = in_d["Q"]
C = in_d["C"]

# Build width and bed elevation arrays combining the input values
# of the different reaches
x = np.array([0])
eta0 = np.array([0])
B = np.array(in_d["reaches"][0]["B"])
for i, reach in enumerate(in_d["reaches"]):
    M_reach = int(reach["L"] / reach["dx"])

    B = np.append(B, np.ones(M_reach - 1) * reach["B"])
    x = np.append(
        x, np.linspace(x[-1], x[-1] + (M_reach - 1) * reach["dx"], num=M_reach)[1:]
    )
    eta0 = np.append(
        eta0, bed_IC(reach["S"], reach["L"], M_reach, eta_ups=eta0[-1])[1:]
    )
eta = eta0[:]
S = -(eta[1:] - eta[:-1]) / (x[1:] - x[:-1])
S = np.append(S, S[-1])

# Reference flow
Y_cr = crit_depth(Q, B)
Y_u = uniflow_depth(Q, B, S, C)

# Boundary conditions on the free surface level
h_ups0 = eta[0] + Y_u[0]
h_ds0 = eta[-1] + Y_u[-1]
print(f"{h_ups0 = :.2f}, {h_ds0 = :.2f}")

# Initialize arrays
M = np.size(x)
Y = np.zeros(M)
Y_cr = np.zeros(M)
Y_sub = np.zeros(M)
Y_super = np.zeros(M)
H_sub = np.zeros(M)
H_super = np.zeros(M)

for n in range(in_d["num_iter"]):
    Y_cr[:] = crit_depth(Q, B)
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

    # Shields update and bed evolution

for i in range(len(Y) - 1):
    if Y[i] == Y_super[i] and Y[i + 1] == Y_sub[i + 1]:
        print(f"Hydraulic jump at index {i}")

# ----------------------------------- Plots ---------------------------------- #
plt.plot(x, eta, color=bed_color, label="Bed elevation")
plt.plot(x, eta + Y, color=water_color, label="Water level")
# plt.plot(x, eta + Y_sub, label="sub")
# plt.plot(x, eta + Y_super, label="super")
plt.plot(x, eta + Y_cr, label="Critical", color="black", linestyle="dashed", alpha=0.7)
plt.plot(x, eta + Y_u, label="Uniform flow", color="black", alpha=0.7)
plt.xlim([np.min(x), np.max(x)])
plt.xlabel("Longitudinal coordinate $x$ [m]")
plt.ylabel("Vertical coordinate $z$ [m]")
plt.grid()
plt.legend()
plt.show()
