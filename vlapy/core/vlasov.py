import numpy as np


def __get_k__(ax):
    """
    get axis of transformed quantity

    :param ax:
    :return:
    """
    return np.fft.fftfreq(ax.size, d=ax[1] - ax[0])


def update_spatial_adv_spectral(f, kx, v, dt):
    """
    evolution of df/dt = v df/dx

    :param f:
    :param kx:
    :param v:
    :param dt:
    :return:
    """

    return np.real(np.fft.ifft(__vdfdx__(np.fft.fft(f, axis=0), v, kx, dt), axis=0))


def update_velocity_adv_spectral(f, kv, e, dt):
    """
    evolution of df/dt = e df/dv

    :param f:
    :param kv:
    :param e:
    :param dt:
    :return:
    """

    return np.real(np.fft.ifft(__edfdv__(np.fft.fft(f, axis=1), e, kv, dt), axis=1))


def __edfdv__(fp, e, kv, dt):
    """

    :param fp:
    :param e:
    :param kv:
    :param dt:
    :return:
    """
    return np.exp(-1j * kv * dt * e[:, None]) * fp


def __vdfdx__(fp, v, kx, dt):
    """

    :param fp:
    :param v:
    :param kx:
    :param dt:
    :return:
    """
    return np.exp(-1j * kx[:, None] * dt * v) * fp
