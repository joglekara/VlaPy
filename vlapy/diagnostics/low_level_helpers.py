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
from scipy import interpolate, signal


def get_nth_mode(efield_arr, mode_number):
    """
    Filters the electric field for only the first Fourier mode and reconstructs it back to real space

    :param efield_arr:
    :param mode_number:
    :return:
    """
    ek = np.fft.fft(efield_arr.data, axis=1, norm="ortho")
    ek_rec = np.zeros(efield_arr.shape, dtype=np.complex128)
    ek_rec[:, mode_number] = ek[:, mode_number]
    ek_rec = 2 * np.fft.ifft(ek_rec, axis=1, norm="ortho")

    return ek_rec


def get_w_ax(efield_arr):
    """
    Just gets the frequency axis

    :param efield_arr:
    :return:
    """
    tax = efield_arr.coords["time"].data
    dt = tax[1] - tax[0]
    wax = np.fft.fftfreq(tax.size, d=dt) * 2 * np.pi
    return wax


def get_e_ss(efield_arr):
    """
    Gets E_max

    :param efield_arr:
    :return:
    """

    ek1_ss = 2 * np.fft.fft(efield_arr.data, axis=1, norm="ortho")[-1, 1]

    return np.abs(ek1_ss), np.angle(ek1_ss)


def get_damping_rate(efield_arr):
    """
    Gets the gradient of the last 75% of the simulation.

    TODO AJ - remove the hardcoding here

    :param efield_arr:
    :return:
    """
    tax = efield_arr.coords["time"].data

    t_ind = tax.size // 4

    ek = np.fft.fft(efield_arr.data, axis=1)
    ek_mag = np.array([np.abs(ek[it, 1]) for it in range(tax.size)])[t_ind:]

    dedt = np.gradient(np.log(ek_mag), tax[2] - tax[1])

    return np.mean(dedt)


def get_nlfs(ef, wepw):
    """
    Calculate the shift in frequency with respect to a reference

    This can be done by subtracting a signal at the reference frequency from the
    given signal

    :param ef:
    :param wepw:
    :return:
    """
    ek1 = np.fft.fft(ef.data, axis=1)[:, 1]

    # ek1.shape
    dt = ef.coords["time"].data[2] - ef.coords["time"].data[1]
    midpt = int(ek1.shape[0] / 2)

    window = 1
    # Calculate hilbert transform
    analytic_signal = signal.hilbert(window * np.real(ek1))
    # Determine envelope
    amplitude_envelope = np.abs(analytic_signal)
    # Phase = angle(signal)    ---- needs unwrapping because of periodicity
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # f(t) = dphase/dt
    instantaneous_frequency = np.diff(instantaneous_phase) / dt  ### Sampling rate!
    # delta_f(t) = f(t) - driver_freq
    freq_shift = (instantaneous_frequency - wepw) / wepw

    # Smooth the answer
    b, a = signal.butter(8, 0.125)
    freq_shift_smooth = signal.filtfilt(b, a, freq_shift, padlen=midpt)

    return freq_shift_smooth


def get_normalized_slope(f_arr, vph):
    """
    Get current slope normalized to initial slope.
    Uses splines for better approximations.

    :param f_arr:
    :param vph:
    :return:
    """
    vax = f_arr.coords["velocity"].data
    dfk0_now = np.gradient(np.squeeze(np.abs(f_arr["distribution_function"][-1, 0, :])))
    dfk0_initial = np.gradient(
        np.squeeze(np.abs(f_arr["distribution_function"][0, 0, :]))
    )

    spline_now = interpolate.interp1d(vax, dfk0_now, kind="cubic")
    spline_initial = interpolate.interp1d(vax, dfk0_initial, kind="cubic")

    return spline_now(vph) / spline_initial(vph)


def get_oscillation_frequency(efield_arr):
    """
    Get oscillation frequency of electric field array

    :param efield_arr:
    :return:
    """
    tax = efield_arr.coords["time"].data

    # TODO AJ - need to come up with a better way to specify this index
    t_ind = tax.size // 4
    ekw = np.fft.fft2(efield_arr.data[t_ind:, :])
    ek1w = np.abs(ekw[:, 1])

    wax = get_w_ax(efield_arr)

    return wax[ek1w.argmax()]
