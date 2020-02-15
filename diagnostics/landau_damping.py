import numpy as np
import mlflow


class LandauDamping:
    def __init__(self):
        pass

    def __call__(self, storage_manager):
        # pass
        damping_rate = self.get_damping_rate(storage_manager)
        frequency = self.get_oscillation_frequency(storage_manager)

        metrics = {
            "damping_rate": damping_rate,
            "frequency": frequency,
        }

        mlflow.log_metrics(metrics=metrics)

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
        pass

    def __plot_e_vs_t(self, w, e, title):
        pass

    def __plot_e_vs_w(self, w, e, title):
        pass

    def __plot_f(self, f, x, v, title):
        pass
