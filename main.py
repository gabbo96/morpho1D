import os
import json
import matplotlib.pyplot as plt
from hydro_module import *

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
Y_cr = np.zeros(M)

for n in range(in_d["num_iter"]):
    Y_cr[:] = crit_depth(Q, B)
    Y = compute_profile(M, h_ups0, h_ds0, Q, B, C, Y_cr, eta, x)

    # Shields update and bed evolution

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
