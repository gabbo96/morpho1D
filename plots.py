import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def plot_eta_h(
    x,
    eta,
    h,
    Y_u=None,
    Y_cr=None,
    bed_color="goldenrod",
    water_color="dodgerblue",
    ic=None,
    title=None,
    eta0_scale=0,
    x_bounds=None,
    plot_axis_labels=True,
    ylims=None,
):
    fig, ax = plt.subplots()
    if eta is not None and h is not None:
        ax.plot(x, h - eta0_scale, label="Water level", color=water_color, zorder=3)
        ax.plot(x, eta - eta0_scale, label="Bed elevation", color=bed_color, zorder=3)
    if Y_u is not None:
        ax.plot(
            x,
            eta + Y_u - eta0_scale,
            label="Uniform flow",
            color="slategrey",
            alpha=0.7,
            zorder=2,
        )
    if Y_cr is not None:
        ax.plot(
            x,
            eta + Y_cr - eta0_scale,
            "--",
            label="Critical",
            color="slategrey",
            alpha=0.7,
            zorder=2,
        )
    if ic is not None:
        ax.plot(x, ic[0] - eta0_scale, ":", color=bed_color, zorder=3)
        ax.plot(x, ic[1] - eta0_scale, ":", color=water_color, zorder=3)
        ax.plot([], ":", color="slategray", label="Initial conditions")
    if x_bounds is not None:
        for x_bound in x_bounds:
            ax.axvline(
                x_bound, color="black", linestyle="dotted", linewidth=1.5, zorder=2
            )
    if plot_axis_labels:
        ax.set_xlabel("Longitudinal coordinate $x$ [m]")
        ax.set_ylabel("Change in vertical coordinate $z-z_0$ [m]")
    if title is not None:
        ax.set_title(title)

    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_xlim([np.min(x), np.max(x)])
    ax.grid(linewidth=0.5, zorder=1)
    ax.legend()
    return fig, ax


class SliderPlot:
    def __init__(
        self,
        plot_timestamps,
        x,
        eta_out,
        h_out,
        eta0_scale,
        x_bounds,
        bed_color="goldenrod",
        water_color="dodgerblue",
    ):
        self.timestamps = plot_timestamps
        self.eta = eta_out
        self.h = h_out
        self.yscale = eta0_scale
        self.fig, self.axs = plot_eta_h(
            x,
            None,
            None,
            ic=[eta_out[0, :], h_out[0, :]],
            eta0_scale=self.yscale,
            x_bounds=x_bounds,
        )
        self.axs.set_ylim(
            [np.min(eta_out - eta0_scale), np.max(h_out[-1, :] - eta0_scale) + 0.5]
        )
        self.fig.subplots_adjust(bottom=0.25)
        self.ax_t = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])  # Add slider axes
        self.s_t = Slider(
            self.ax_t,
            r"Time $t/T_E$",
            np.min(self.timestamps),
            np.max(self.timestamps),
            valinit=self.timestamps[0],
            valstep=self.timestamps,
            color="green",
        )

        # Initial plot and slider update
        t_stamp = self.timestamps[0]
        t_idx = np.argwhere(np.array(self.timestamps) == t_stamp)[0][0]
        eta = eta_out[t_idx, :]
        h = h_out[t_idx, :]
        (self.l_eta,) = self.axs.plot(
            x, eta - self.yscale, label="Bed elevation", color=bed_color
        )
        (self.l_h,) = self.axs.plot(
            x, h - self.yscale, label="Water level", color=water_color
        )
        self.axs.legend()
        self.s_t.on_changed(self.update)

    def update(self, val):
        t_stamp = self.s_t.val
        t_idx = np.argwhere(np.array(self.timestamps) == t_stamp)[0][0]
        eta = self.eta[t_idx, :]
        h = self.h[t_idx, :]
        self.l_eta.set_ydata(eta - self.yscale)
        self.l_h.set_ydata(h - self.yscale)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()
