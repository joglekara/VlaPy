import numpy as np
from numba import njit


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


@njit
def __edfdv__(fp, e, kv, dt):
    for ix in range(e.size):
        fp[ix,] = (
            np.exp(-1j * kv * dt * e[ix]) * fp[ix,]
        )

    return fp


@njit
def __vdfdx__(fp, v, kx, dt):
    for ix in range(kx.size):
        fp[ix,] = (
            np.exp(-1j * kx[ix] * dt * v) * fp[ix,]
        )

    return fp
