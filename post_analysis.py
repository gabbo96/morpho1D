import os
import json
import numpy as np
from plots import *
from init_conditions import initial_conditions
from hydro_module import crit_depth, uniflow_depth, compute_profile

plt.style.use("stylesheet.mplstyle")

show_plots = True
ylims = None
outputs_dir = "outputs"
sim_dirname = "local_widening_mild_Lw10"
sim_dirpath = os.path.join(outputs_dir, sim_dirname)
input_filepath = os.path.join(sim_dirpath, "inputs.json")

x = np.loadtxt(os.path.join(sim_dirpath, "x.csv"))
eta_out = np.loadtxt(os.path.join(sim_dirpath, "eta.csv"))
eta0_scale = np.loadtxt(os.path.join(sim_dirpath, "eta0_scale.csv"))
h_out = np.loadtxt(os.path.join(sim_dirpath, "h.csv"))
timestamps = np.loadtxt(os.path.join(sim_dirpath, "timestamps.csv"))

with open(input_filepath, "r") as f:
    d = json.load(f)

Q = d["Q"]
C = d["C"]
x, eta, B, S = initial_conditions(d)
x_bounds_reaches = np.cumsum([dL["L"] for dL in d["reaches"]])[
    :-1
]  # for vertical dashed lines in plots

# Reference flow
Y_cr = crit_depth(Q, B)
Y_u = uniflow_depth(Q, B, S, C)

# Boundary conditions on the free surface level
h_ups0 = eta[0] + Y_u[0]
h_ds0 = eta[-1] + Y_u[-1]

# Plot initial bed and water level profiles
Y = compute_profile(h_ups0, h_ds0, Q, B, C, Y_cr, eta, x)
plot_eta_h(
    x,
    eta_out[0, :],
    eta_out[0, :] + Y,
    Y_u=Y_u,
    Y_cr=Y_cr,
    eta0_scale=eta0_scale,
    x_bounds=x_bounds_reaches,
    plot_axis_labels=False,
    ylims=ylims,
)
plt.savefig(
    os.path.join(sim_dirpath, "Initial conditions.png"), dpi=600, bbox_inches="tight"
)

# ---------------------------- Final configuration --------------------------- #
eta = eta_out[-1, :]
S = -(eta[1:] - eta[:-1]) / (x[1:] - x[:-1])
S = np.append(S, S[-1])
Y_u = np.zeros_like(S)
Y_u[S > 0] = uniflow_depth(Q, B[S > 0], S[S > 0], C)
Y_u[S < 0] = np.nan
plot_eta_h(
    x,
    eta_out[-1, :],
    h_out[-1, :],
    Y_cr=Y_cr,
    x_bounds=x_bounds_reaches,
    eta0_scale=eta0_scale,
    plot_axis_labels=False,
    ylims=ylims,
    ic=[eta_out[0, :], eta_out[0, :] + Y],
)

plt.savefig(
    os.path.join(sim_dirpath, "Final configuration.png"), dpi=600, bbox_inches="tight"
)

# Create an instance of the SliderPlot class and show the plot
slider_plot = SliderPlot(timestamps, x, eta_out, h_out, eta0_scale, x_bounds_reaches)
if show_plots:
    slider_plot.show()
