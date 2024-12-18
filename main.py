# %%
import os
import json
import matplotlib.pyplot as plt
from hydro_module import *
from morpho_module import *
from init_conditions import *
from plots import *

plt.style.use("stylesheet.mplstyle")

inputs_filename = "local_widening_mild.json"
sim_name = "local_widening_mild_Lw10"

# Setup I/O directories
inputs_dir = "inputs_files"  # folder where the input .json file is stored
outputs_dir = "outputs"  # folder where the simulation subfolder will be created

# ---------------------- Initial and boundary conditions --------------------- #
# Import input dictionary and set initial conditions
with open(os.path.join(inputs_dir, inputs_filename), "r") as f:
    d = json.load(f)
Q = d["Q"]
C = d["C"]
x, eta, B, S = initial_conditions(d)
x_bounds_reaches = np.cumsum([dL["L"] for dL in d["reaches"]])[
    :-1
]  # for vertical dashed lines in plots

# Reference plane to scale the plots
S0 = d["reaches"][0]["S"]
Ltot = np.sum([dL["L"] for dL in d["reaches"]])
eta0_scale = np.linspace(eta[0], eta[0] - S0 * Ltot, num=eta.size)
print(f"Fr_0 = {C*S0**0.5:.2f}")

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
    eta,
    eta + Y,
    Y_u=Y_u,
    Y_cr=Y_cr,
    title="Initial conditions",
    eta0_scale=eta0_scale,
    x_bounds=x_bounds_reaches,
)
# Compute reference transport capacity and Exner timescale
taus_0 = energy_slope(Q, B[0], C, Y_u[0]) * Y_u[0] / (d["Delta"] * d["Ds"])
q_s0 = compute_qs(taus_0, d["tf"], d["Ds"], d["Delta"])
T_E = (1 - d["p"]) * B[0] * Y_u[0] / q_s0
print(f"{taus_0 = :.2f}")

# %%
# ------------------------------ Time evolution ------------------------------ #
n = 0  # iterations counter
t_star = 0  # dimensionless time
dts = []  # dimensional timestep

# Setup variables to store outputs
num_plotstep = int(d["t_star_max"] / d["t_star_plotstep"])
timestamps = np.linspace(d["t_star_plotstep"], d["t_star_max"], num=num_plotstep)
actual_timestamps = [0]
plot_idx = 0
h_out = np.zeros((timestamps.size + 1, eta.size))
eta_out = np.zeros((timestamps.size + 1, eta.size))
eta_out[0, :] = eta[:]
h_out[0, :] = eta_out[0, :] + Y[:]

# For loop in time
while t_star < d["t_star_max"] and d["mobile_bed"] == True:
    h_ups0 = eta[0] + Y_u[0]
    # h_ds0 = eta[-1] + Y[-1]
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
        use_cfl=True,
    )
    eta[:] = eta_new[:]

    dts.append(dt)
    t_star += dt / T_E
    if t_star >= timestamps[plot_idx]:
        print(f"{t_star = :.2f} ({t_star/d['t_star_max']*100:.0f}%)")
        actual_timestamps.append(t_star)
        plot_idx += 1
        eta_out[plot_idx, :] = eta_new[:]
        h_out[plot_idx, :] = eta_new[:] + Y[:]
    n += 1

if d["mobile_bed"]:
    # Save final configuration
    eta_out[-1, :] = eta_new[:]
    h_out[-1, :] = eta_new[:] + Y[:]

# Create simulation directory
out_dirpath = os.path.join(outputs_dir, sim_name)
os.makedirs(out_dirpath, exist_ok=True)

# Save input dictionary
with open(os.path.join(out_dirpath, "inputs.json"), "w") as f:
    json.dump(d, f, indent=4)

# Save output arrays to csv files
np.savetxt(os.path.join(out_dirpath, "timestamps.csv"), actual_timestamps)
np.savetxt(os.path.join(out_dirpath, "x.csv"), x)
np.savetxt(os.path.join(out_dirpath, "eta0_scale.csv"), eta0_scale)
np.savetxt(os.path.join(out_dirpath, "eta.csv"), eta_out)
np.savetxt(os.path.join(out_dirpath, "h.csv"), h_out)

# %%
# ----------------------------------- Plots ---------------------------------- #
if d["mobile_bed"]:
    # Time steps returned by the CFL condition
    plt.figure()
    plt.plot(dts)

    # Final configuration
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
        Y_u=Y_u,
        ic=[eta_out[0, :], h_out[0, :]],
        title="Final configuration",
        eta0_scale=eta0_scale,
        x_bounds=x_bounds_reaches,
    )

    # Slider
    slider_plot = SliderPlot(
        actual_timestamps, x, eta_out, h_out, eta0_scale, x_bounds_reaches
    )
    slider_plot.show()

plt.show()

# %%
