import numpy as np
import mlflow
from matplotlib import pyplot as plt
import os


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
