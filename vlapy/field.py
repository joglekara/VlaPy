import numpy as np


def compute_charges(f, dv):
    """
    Computes a simple moment of the distribution function along
    the velocity axis

    :param f:
    :param dv:
    :return:
    """
    return np.sum(f, axis=1) * dv


def __fft_solve__(net_charge_density, kx):
    """
    del^2 phi = -rho
    del e = - integral[rho] = - integral[fdv]

    :param net_charge_density:
    :param kx:
    :return:
    """
    rhok = np.fft.fft(net_charge_density)
    rhok[0] = 0
    rhok[1:] = rhok[1:] / (-1j * kx[1:])

    return np.real(np.fft.ifft(rhok))


def solve_for_field(charge_density, kx):
    """
    Solves for the net electric field after subtracting ion charge

    :param charge_density:
    :param kx:

    :return:
    """
    return __fft_solve__(net_charge_density=1.0 - charge_density, kx=kx)


def get_total_electric_field(driver_field, f, dv, kx):
    """
    Allows adding a driver field

    :param driver_field:
    :param f:
    :param dv:
    :param kx:
    :return:
    """
    return driver_field + solve_for_field(charge_density=compute_charges(f, dv), kx=kx)
