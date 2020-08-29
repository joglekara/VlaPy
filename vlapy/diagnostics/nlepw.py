# MIT License
#
# Copyright (c) 2020 Archis Joglekar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
from scipy import fft

from vlapy.diagnostics import low_level_helpers as llh
from vlapy.diagnostics import base


class NLEPW(base.BaseDiagnostic):
    def __init__(self, vph, wepw):
        super().__init__()
        self.rules_to_store_f = {
            "time": "first-last",
            "space": ["k" + str(ik) for ik in range(0, 2)],
        }

        self.vph = vph
        self.wepw = wepw

    def __call__(self, storage_manager):
        super().pre_custom_diagnostics(storage_manager=storage_manager)

        metrics = self.get_metrics(storage_manager=storage_manager)
        self.make_plots(storage_manager=storage_manager)

        super().log_custom_metrics(metrics=metrics)
        super().post_custom_diagnostics(storage_manager=storage_manager)

    def get_metrics(self, storage_manager):
        metrics = {}

        density_k = (
            2.0
            / storage_manager.fields_dataset["n"].coords["space"].data.size
            * np.abs(
                fft.fft(
                    storage_manager.fields_dataset["n"].data,
                    axis=1,
                    workers=-1,
                )
            )
        )

        for ik in range(1, len(self.rules_to_store_f["space"])):
            density_ik = density_k[:, ik]
            metrics["sum_n_" + str(ik)] = np.sum(density_ik) / density_ik.size

        return metrics

    def make_plots(self, storage_manager):
        super().make_plots(storage_manager=storage_manager)
        tax = storage_manager.fields_dataset["e"].coords["time"].data
        xax = storage_manager.fields_dataset["e"].coords["space"].data
        ek = np.abs(
            2.0
            / xax.size
            * fft.fft(storage_manager.fields_dataset["e"].data, axis=1, workers=-1)
        )

        ek_mag = [
            np.squeeze(ek[:, ik])
            for ik in range(1, len(self.rules_to_store_f["space"]))
        ]

        for log in [True, False]:
            base.plot_e_vs_t(
                plots_dir=self.plots_dir,
                t=tax,
                e=ek_mag,
                title="Electric Field Modes vs Time",
                log=log,
            )

        self.make_f_plots(storage_manager=storage_manager)

    def make_f_plots(self, storage_manager):
        current_time = storage_manager.fields_dataset["e"].coords["time"].data[-1]

        iv_to_plot = (
            np.abs(
                storage_manager.dist_dataset["distribution_function"]
                .coords["velocity"]
                .data
                - self.vph
            )
            < 1.5
        )

        v_to_plot = (
            storage_manager.dist_dataset["distribution_function"]
            .coords["velocity"]
            .data[iv_to_plot]
        )

        base.plot_f_vs_x_and_v(
            plots_dir=self.plots_dir,
            f=storage_manager.current_f[:, iv_to_plot],
            x=storage_manager.fields_dataset["e"].coords["space"].data,
            v=v_to_plot,
            title="f(x,v) @ t = "
            + str(np.round(current_time, 2))
            + r" $\omega_p^{-1}$",
            filename="f(x,v).png",
        )

        for ik in range(
            storage_manager.dist_dataset["distribution_function"]
            .coords["fourier_mode"]
            .data.size
        ):
            f_to_plot = np.abs(
                storage_manager.dist_dataset["distribution_function"].loc[
                    {"fourier_mode": ik, "velocity": v_to_plot}
                ]
            )

            base.plot_f_vs_v(
                plots_dir=self.plots_dir,
                f=f_to_plot,
                v=v_to_plot,
                ylabel=r"$\hat{f}$",
                title=str(ik) + "-Mode of Distribution Function",
                filename="abs(f^(k=" + str(ik) + ")).png",
            )
