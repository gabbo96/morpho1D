import json
import matplotlib.pyplot as plt
from module import *

bed_color = "goldenrod"
water_color = "dodgerblue"

# Retrieve inputs
with open("inputs.json", "r") as f:
    in_d = json.load(f)
dx = in_d["L"] / (in_d["num_nodes"] - 1)
B = in_d["B"]
C = in_d["C"]

# Reference flow
Y_cr = crit_depth(in_d["Q"], in_d["B"])
Y_u = uniflow_depth(in_d["Q"], in_d["B"], in_d["S"], in_d["C"])

# Initial conditions
eta0 = bed_IC(in_d["S"], in_d["L"], in_d["num_nodes"])
eta = eta0[:]

# Boundary conditions on the free surface level
h_ups0 = eta[0] + Y_u + 0.5
h_ds0 = eta[-1] + Y_u + 2

# Initialize arrays
Y = np.zeros(in_d["num_nodes"])
Y_cr = np.zeros(in_d["num_nodes"])
Y_sub = np.zeros(in_d["num_nodes"])
Y_super = np.zeros(in_d["num_nodes"])
H_sub = np.zeros(in_d["num_nodes"])
H_super = np.zeros(in_d["num_nodes"])

for n in range(in_d["num_iter"]):
    Q = in_d["Q"]
    Y_cr[:] = crit_depth(Q, B)
    E_min = energy(Y_cr, Q, B, eta) - eta

    # ------------------ First spatial loop: upstream direction ------------------ #
    # Downstream BC
    Y_sub[-1] = h_ds0 - eta[-1]
    if froude(Q, B, Y_sub[-1]) > 1:
        Y_sub[-1] = Y_cr[-1]
    H_sub[-1] = energy(Y_sub[-1], Q, B, eta[-1])

    # Spatial loop
    for i in range(in_d["num_nodes"] - 2, -1, -1):
        j = energy_slope(Q, B, C, Y_sub[i + 1])
        H_sub[i] = H_sub[i + 1] + j * dx

        # Check if the specific energy is lower than the minimum for the given flow discharge
        if H_sub[i] - eta[i] < E_min[i]:
            Y_sub[i] = Y_cr[i]
        else:
            YL = Y_cr[i]
            YR = 10 * Y_cr[i]
            Y_sub[i] = bisection(energy_bisec_f, YL, YR, (H_sub[i], Q, B, eta[i]))

    # ----------------- Second spatial loop: downstream direction ---------------- #
    # Upstream BC
    Y_super[0] = h_ups0 - eta[0]
    if froude(Q, B, Y_super[0]) < 1:
        Y_super[0] = Y_cr[0]
    H_super[0] = energy(Y_super[0], Q, B, eta[0])

    # Spatial loop
    for i in range(in_d["num_nodes"] - 1):
        j = energy_slope(Q, B, C, Y_super[i])
        H_super[i + 1] = H_super[i] - j * dx
        # Check if the specific energy is lower than the minimum for the given flow discharge
        if H_super[i + 1] - eta[i + 1] < E_min[i + 1]:
            Y_super[i + 1] = Y_cr[i + 1]
        else:
            YL = 1e-6
            YR = Y_cr[i + 1]
            Y_super[i + 1] = bisection(
                energy_bisec_f, YL, YR, (H_super[i + 1], Q, B, eta[i + 1])
            )

    # -------------------------- Final profile retrieval ------------------------- #
    S_sub = spinta(Q, B, Y_sub)
    S_super = spinta(Q, B, Y_super)
    Y[S_sub > S_super] = Y_sub[S_sub > S_super]
    Y[S_sub < S_super] = Y_super[S_sub < S_super]

    # Shields update and bed evolution

for i in range(len(Y)):
    if Y[i] == Y_super[i] and Y[i + 1] == Y_sub[i + 1]:
        print(f"Hydraulic jump at index {i}")

# ----------------------------------- Plots ---------------------------------- #
x_plot = np.linspace(0, in_d["L"], num=in_d["num_nodes"])
plt.plot(x_plot, eta, color=bed_color, label="Bed elevation")
plt.plot(x_plot, eta + Y, color=water_color, label="Water level")
plt.plot(
    x_plot, eta + Y_cr, label="Critical", color="black", linestyle="dashed", alpha=0.7
)
plt.plot(x_plot, eta + Y_u, label="Uniform flow", color="black", alpha=0.7)
plt.xlim([np.min(x_plot), np.max(x_plot)])
plt.xlabel("Longitudinal coordinate $x$ [m]")
plt.ylabel("Vertical coordinate $z$ [m]")
plt.grid()
plt.legend()
plt.show()
