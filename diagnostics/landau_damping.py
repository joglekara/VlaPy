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

from diagnostics import low_level_helpers as llh


class LandauDamping:
    def __init__(self, params_to_log):
        self.params_to_log = params_to_log

        self.f_rules = "k0k1"

    def __call__(self, storage_manager):
        self.plots_dir = os.path.join(storage_manager.base_path, "plots")
        os.makedirs(self.plots_dir)

        metrics = {
            "damping_rate": llh.get_damping_rate(efield_arr=storage_manager.efield_arr),
            # "frequency": llh.get_oscillation_frequency(
            #     efield_arr=storage_manager.efield_arr
            # ),
            "E_max": llh.get_e_max(efield_arr=storage_manager.efield_arr),
        }

        mlflow.log_metrics(metrics=metrics)

        self.make_plots(storage_manager)

    def make_plots(self, storage_manager):
        tax = storage_manager.efield_arr.coords["time"].data
        ek_rec = llh.get_first_mode(storage_manager.efield_arr)

        ek_mag = np.array([np.abs(ek_rec[it, 1]) for it in range(tax.size)])

        self.__plot_e_vs_t(tax, ek_mag, "Electric Field Amplitude vs Time")

    def __plot_e_vs_t(self, t, e, title):
        this_fig = plt.figure(figsize=(8, 4))
        this_plt = this_fig.add_subplot(111)
        this_plt.plot(t, e)
        this_plt.set_xlabel(r"Time ($\omega_p^{-1}$)", fontsize=12)
        this_plt.set_ylabel(r"$\hat{E}_{k=1}$", fontsize=12)
        this_plt.set_title(title, fontsize=14)
        this_plt.grid()
        this_fig.savefig(
            os.path.join(self.plots_dir, "E_vs_time.png"), bbox_inches="tight"
        )
        plt.close(this_fig)

    def __plot_f(self, f, x, v, title):
        pass
