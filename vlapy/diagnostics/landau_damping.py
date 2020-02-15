import numpy as np


class LandauDamping:
    def __init__(self):
        pass

    def __call__(self, storage_manager):
        pass
        # damping_rate = self.get_damping_rate(storage_manager)
        # frequency = self.get_oscillation_frequency(storage_manager)
        #
        # metrics = {
        #     "damping_rate": damping_rate,
        #     "frequency": frequency,
        # }
        #
        # mlflow.log_metrics(metrics=metrics)

    def get_damping_rate(self, storage_manager):
        efield_arr = storage_manager.efield_arr
        tax = efield_arr.coords["t"]

        t_ind = tax.size // 4

        print("t_ind: " + str(t_ind))

        ek = np.fft.fft(efield_arr.data, axis=1)
        ek_mag = np.array([np.abs(ek[it, 1]) for it in range(tax.size)])[t_ind:]

        print("ek_mag shape: " + str(ek_mag.shape))
        print(ek_mag)
        dEdt = np.gradient(np.log(ek_mag))  # , tax[2] - tax[1])

        print(dEdt.shape)

        return np.mean(dEdt)

    def get_oscillation_frequency(self, storage_manager):
        efield_arr = storage_manager.efield_arr
        tax = efield_arr.coords["t"]
        nt = tax.size
        dt = tax[1] - tax[0]

        ekw = np.fft.fft2(efield_arr.data[nt // 2 :,])
        ek1w = np.abs(ekw[:, 1])
        wax = np.fft.fftfreq(ek1w.shape[0], d=dt) * 2 * np.pi

        return wax[ek1w.argmax()]

    def make_plots(self, storage_manager):
        pass

    def __plot_e_vs_t(self, w, e, title):
        pass

    def __plot_e_vs_w(self, w, e, title):
        pass

    def __plot_f(self, f, x, v, title):
        pass
