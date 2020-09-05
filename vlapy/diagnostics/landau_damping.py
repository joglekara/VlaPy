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


class LandauDamping(base.BaseDiagnostic):
    def __init__(self, vph, wepw):
        super().__init__()
        self.rules_to_store_f = {
            "time": "first-last",
            "space": ["k0", "k1"],
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
        e_amp, e_phase = llh.get_e_ss(efield_arr=storage_manager.fields_dataset["e"])
        metrics = {
            "damping_rate": llh.get_damping_rate(
                efield_arr=storage_manager.fields_dataset["e"]
            ),
            "E_ss_amp": e_amp,
            "E_ss_phase": e_phase / np.pi,
        }
        return metrics

    def make_plots(self, storage_manager):
        super().make_plots(storage_manager=storage_manager)

        wax = llh.get_w_ax(storage_manager.fields_dataset["e"])
        tax = storage_manager.fields_dataset["e"].coords["time"].data
        ek_rec = llh.get_nth_mode(storage_manager.fields_dataset["e"], 1)

        ek_mag = np.array([np.abs(ek_rec[it, 1]) for it in range(tax.size)])
        ekw_mag = np.abs(fft.fft(np.array([ek_rec[it, 1] for it in range(tax.size)])))

        ek1_shift = llh.get_nlfs(storage_manager.fields_dataset["e"], self.wepw)

        base.plot_e_vs_t(
            plots_dir=self.plots_dir,
            t=tax,
            e=ek_mag,
            title="Electric Field Amplitude vs Time",
        )
        base.plot_e_vs_t(
            plots_dir=self.plots_dir,
            t=tax,
            e=ek_mag,
            title="Electric Field Amplitude vs Time",
            log=False
        )
        base.plot_e_vs_w(
            plots_dir=self.plots_dir,
            w=wax,
            e=ekw_mag,
            title="Electric Field Amplitude vs Frequency",
        )
        base.plot_dw_vs_t(
            plots_dir=self.plots_dir,
            t=tax,
            ek1_shift=ek1_shift,
            title="Non-linear Frequency Shift vs Time",
        )

        self.make_f_plots(storage_manager=storage_manager)

    def make_f_plots(self, storage_manager):
        iv_to_plot = (
            storage_manager.dist_dataset["distribution_function"]
            .coords["velocity"]
            .data
            > 0
        )

        v_to_plot = (
            storage_manager.dist_dataset["distribution_function"]
            .coords["velocity"]
            .data[iv_to_plot]
        )

        for ik in range(
            storage_manager.dist_dataset["distribution_function"]
            .coords["fourier_mode"]
            .data.size
        ):
            f_to_plot = np.abs(
                storage_manager.dist_dataset["distribution_function"].data[
                    :, ik, iv_to_plot
                ]
            )

            base.plot_f_vs_v(
                plots_dir=self.plots_dir,
                f=np.abs(f_to_plot),
                v=v_to_plot,
                ylabel=r"$\hat{f}$",
                title=str(ik) + "-Mode of Distribution Function",
                filename="abs(f^(k=" + str(ik) + ")).png",
            )
