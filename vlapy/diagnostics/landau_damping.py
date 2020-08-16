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

import os

import numpy as np
import mlflow
from matplotlib import pyplot as plt

from vlapy.diagnostics import low_level_helpers as llh
from vlapy.diagnostics import base


class LandauDamping(base.BaseDiagnostic):
    def __init__(self, params_to_log, vph, wepw):
        super().__init__()
        self.params_to_log = params_to_log
        self.rules_to_store_f = {
            "time": "first-last",
            "space": ["k0", "k1"],
        }

        self.vph = vph
        self.wepw = wepw

    def __call__(self, storage_manager):
        super()._make_dirs_(storage_manager)

        storage_manager.load_data_over_all_timesteps()
        e_amp, e_phase = llh.get_e_ss(efield_arr=storage_manager.overall_arrs["e"])

        metrics = {
            "damping_rate": llh.get_damping_rate(
                efield_arr=storage_manager.overall_arrs["e"]
            ),
            "normalized_slope": llh.get_normalized_slope(
                f_arr=storage_manager.overall_arrs["distribution"], vph=self.vph
            ),
            "E_ss_amp": e_amp,
            "E_ss_phase": e_phase / np.pi,
            "wB": np.sqrt(e_amp * self.wepw / self.vph),
            "dw_ss": llh.get_nlfs(storage_manager.overall_arrs["e"], wepw=self.wepw)[
                int(0.7 * storage_manager.overall_arrs["e"].coords["time"].data.size)
            ],
        }

        health_metrics = super().update_health_metrics_dict(
            storage_manager=storage_manager
        )

        mlflow.log_metrics(metrics=metrics)
        mlflow.log_metrics(metrics=health_metrics)

        self.make_plots(storage_manager)

        storage_manager.unload_data_over_all_timesteps()

    def make_plots(self, storage_manager):

        wax = llh.get_w_ax(storage_manager.overall_arrs["e"])
        tax = storage_manager.overall_arrs["e"].coords["time"].data
        ek_rec = llh.get_nth_mode(storage_manager.overall_arrs["e"], 1)

        ek_mag = np.array([np.abs(ek_rec[it, 1]) for it in range(tax.size)])
        ekw_mag = np.abs(
            np.fft.fft(np.array([ek_rec[it, 1] for it in range(tax.size)]))
        )

        ek1_shift = llh.get_nlfs(storage_manager.overall_arrs["e"], self.wepw)

        v_to_plot = np.abs(
            storage_manager.overall_arrs["distribution"].coords["velocity"].data
            - self.vph
        ) < 8 * np.sqrt(self.wepw / self.vph * ek_mag[-1])
        f_to_plot = np.abs(
            storage_manager.overall_arrs["distribution"]["distribution_function"].data[
                :, 0, v_to_plot
            ]
        )

        base.plot_e_vs_t(
            plots_dir=self.plots_dir,
            t=tax,
            e=ek_mag,
            title="Electric Field Amplitude vs Time",
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
        base.plot_fhat0(
            plots_dir=self.plots_dir,
            f=np.abs(f_to_plot),
            v=storage_manager.overall_arrs["distribution"]
            .coords["velocity"]
            .data[v_to_plot],
            title="Zeroth Mode of Distribution Function",
            filename="fk0_zi.png",
        )

        super().plot_health(storage_manager=storage_manager)

        v_to_plot = storage_manager.overall_arrs["distribution"].coords["velocity"].data
        f_to_plot = storage_manager.overall_arrs["distribution"][
            "distribution_function"
        ].data[:, 0,]

        base.plot_fhat0(
            plots_dir=self.plots_dir,
            f=np.abs(f_to_plot),
            v=v_to_plot,
            title="Zeroth Mode of Distribution Function",
            filename="fk0-zo.png",
        )

        f_to_plot = storage_manager.overall_arrs["distribution"][
            "distribution_function"
        ].data[:, 1,]
        base.plot_fhat0(
            plots_dir=self.plots_dir,
            f=np.abs(f_to_plot),
            v=v_to_plot,
            title="First Mode of Distribution Function",
            filename="fk1-zo.png",
        )

        # fk1_kv_t = np.abs(
        #     np.fft.fft(
        #         np.squeeze(storage_manager.f_arr["distribution_function"][::100, 1,]),
        #         axis=-1,
        #     )
        # )
        #
        # self.__plot_f_k1(
        #     fk1_kv_t,
        #     storage_manager.f_arr.coords["time"].data[::100],
        #     np.fft.fftfreq(
        #         storage_manager.f_arr.coords["velocity"].data.size,
        #         d=storage_manager.f_arr.coords["velocity"].data[2]
        #         - storage_manager.f_arr.coords["velocity"].data[1],
        #     ),
        #     r"$\hat{\hat{f}}^1(k_v, t)$",
        # )
        #
        # del fk1_kv_t

    def __plot_f_k1(self, fk1_kv_t, time, kv, title):

        kv = np.fft.fftshift(kv)
        fk1_kv_t = np.fft.fftshift(fk1_kv_t, axes=-1)

        max_fk1_kv = np.amax(fk1_kv_t)
        largest_wavenumber_index_vs_time = [
            np.amax(
                np.where(fk1_kv_t[it, : fk1_kv_t.shape[1] // 2] < max_fk1_kv * 1e-2)
            )
            for it in range(fk1_kv_t.shape[0])
        ]
        t1 = int(0.3 * fk1_kv_t.shape[0])
        t2 = int(0.7 * fk1_kv_t.shape[0])

        wavenumber_propagation_speed = abs(
            kv[largest_wavenumber_index_vs_time[t2]]
        ) - abs(kv[largest_wavenumber_index_vs_time[t1]])

        wavenumber_propagation_speed /= time[t2] - time[t1]

        this_fig = plt.figure(figsize=(8, 4))
        this_plt = this_fig.add_subplot(111)
        levels = np.linspace(-8, -2, 19)
        cb = this_plt.contourf(kv, time, np.log10(fk1_kv_t), levels=levels)
        this_fig.colorbar(cb)
        this_plt.plot(
            kv[largest_wavenumber_index_vs_time[t2] : kv.size // 2],
            -kv[largest_wavenumber_index_vs_time[t2] : kv.size // 2]
            / wavenumber_propagation_speed,
            "r",
        )
        this_plt.axhline(y=time[t2], ls="--")
        this_plt.axhline(y=time[t1], ls="--")
        this_plt.axvline(x=kv[largest_wavenumber_index_vs_time[t2]], ls="-.")
        this_plt.axvline(x=kv[largest_wavenumber_index_vs_time[t1]], ls="-.")
        this_plt.set_xlabel(r"$k_v$", fontsize=12)
        this_plt.set_ylabel(r"$1/\omega_p$", fontsize=12)
        this_plt.set_title(
            title
            + ", wavenumber propagation speed = "
            + str(round(wavenumber_propagation_speed, 6)),
            fontsize=14,
        )
        this_fig.savefig(
            os.path.join(self.plots_dir, "fk1.png"), bbox_inches="tight",
        )
        plt.close(this_fig)
