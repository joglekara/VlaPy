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


class LandauDamping:
    def __init__(self):
        self.params_to_log = [
            "k0",
            "w0",
            "a0",
        ]

    def __call__(self, storage_manager):
        self.plots_dir = os.path.join(storage_manager.base_path, "plots")
        os.makedirs(self.plots_dir)

        metrics = {
            "damping_rate": self.get_damping_rate(storage_manager),
            "frequency": self.get_oscillation_frequency(storage_manager),
        }

        mlflow.log_metrics(metrics=metrics)

        self.make_plots(storage_manager)

    def get_damping_rate(self, storage_manager):
        efield_arr = storage_manager.efield_arr
        tax = efield_arr.coords["time"].data

        t_ind = tax.size // 4

        ek = np.fft.fft(efield_arr.data, axis=1)
        ek_mag = np.array([np.abs(ek[it, 1]) for it in range(tax.size)])[t_ind:]

        dEdt = np.gradient(np.log(ek_mag), tax[2] - tax[1])

        return np.mean(dEdt)

    def get_oscillation_frequency(self, storage_manager):
        efield_arr = storage_manager.efield_arr
        tax = efield_arr.coords["time"].data
        nt = tax.size
        dt = tax[1] - tax[0]

        ekw = np.fft.fft2(efield_arr.data[nt // 4 :,])
        ek1w = np.abs(ekw[:, 1])

        wax = np.fft.fftfreq(ekw.shape[0], d=dt) * 2 * np.pi

        return wax[ek1w.argmax()]

    def make_plots(self, storage_manager):
        efield_arr = storage_manager.efield_arr
        tax = efield_arr.coords["time"].data
        dt = tax[1] - tax[0]
        wax = np.fft.fftfreq(tax.size, d=dt) * 2 * np.pi

        ek = np.fft.fft(efield_arr.data, axis=1)
        ek_mag = np.array([np.abs(ek[it, 1]) for it in range(tax.size)])
        ekw_mag = np.abs(np.fft.fft(np.array([ek[it, 1] for it in range(tax.size)])))

        self.__plot_e_vs_t(tax, ek_mag, "Electric Field Amplitude vs Time")
        self.__plot_e_vs_w(wax, ekw_mag, "Electric Field Amplitude vs Frequency")

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

    def __plot_e_vs_w(self, w, e, title):
        this_fig = plt.figure(figsize=(8, 4))
        this_plt = this_fig.add_subplot(111)
        this_plt.semilogy(np.fft.fftshift(w), np.fft.fftshift(e), "-x")
        this_plt.set_xlabel(r"Frequency ($\omega_p$)", fontsize=12)
        this_plt.set_ylabel(r"$\hat{\hat{E}}_{k=1}$", fontsize=12)
        this_plt.set_title(title, fontsize=14)
        this_plt.set_xlim(1, 1.5)
        this_plt.grid()
        this_plt.set_ylim(0.001 * np.amax(e), 1.5 * np.amax(e))
        this_fig.savefig(
            os.path.join(self.plots_dir, "E_vs_frequency.png"), bbox_inches="tight"
        )

    def __plot_f(self, f, x, v, title):
        pass
