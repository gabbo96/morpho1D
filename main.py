import os
import json
import matplotlib.pyplot as plt
from hydro_module import *
from morpho_module import *
from init_conditions import *

inputs_dir = "inputs_files"
inputs_filename = "mild_reach_bridge"

bed_color = "goldenrod"
water_color = "dodgerblue"

# ---------------------- Initial and boundary conditions --------------------- #
# Import input dictionary
with open(os.path.join(inputs_dir, f"{inputs_filename}.json"), "r") as f:
    d = json.load(f)
Q = d["Q"]
C = d["C"]

x, eta, B, S = initial_conditions(d)

eta_out = np.zeros((d["num_bed_profiles"], eta.size))
eta_out[0, :] = eta[:]

# Reference flow
Y_cr = crit_depth(Q, B)
Y_u = uniflow_depth(Q, B, S, C)

# Boundary conditions on the free surface level
h_ups0 = eta[0] + Y_u[0]
h_ds0 = eta[-1] + Y_u[-1]
taus_0 = energy_slope(Q, B[0], C, Y_u[0]) * Y_u[0] / (d["Delta"] * d["Ds"])
q_s0 = compute_qs(taus_0, d["tf"], d["Ds"], d["Delta"])

# ------------------------------ Time evolution ------------------------------ #
dts = []
for n in range(d["num_iter"]):
    Y = compute_profile(h_ups0, h_ds0, Q, B, C, Y_cr, eta, x)
    dt, eta_new = update_bed(
        eta,
        d["tf"],
        Q,
        B,
        Y,
        C,
        q_s0,
        d["Delta"],
        d["Ds"],
        d["p"],
        d["dx"],
        d["CFL"],
        dt_max=d["dt_max"],
    )
    eta = eta_new[:]
    dts.append(dt)

# ----------------------------------- Plots ---------------------------------- #
plt.plot(dts)

plt.figure()
plt.plot(x, eta_out[0], ":", color=bed_color, label="Initial bed elevation")
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
