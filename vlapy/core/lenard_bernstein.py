import numpy as np


def make_philharmonic_matrix(vax, nv, nu, dt, dv, v0):
    """
    This matrix is composed of the linear operator that must be inverted with respect
    to the right side, which'll be the distribution function

    :param vax:
    :param nv:
    :param nu:
    :param dt:
    :param dv:
    :param v0:
    :return:
    """

    a = nu * dt * np.ones(nv - 1) * (-(v0 ** 2.0) / dv ** 2.0 + vax[:-1] / 2 / dv)
    b = nu * dt * np.ones(nv) * (2 * v0 ** 2.0 / dv ** 2.0 + 0.0 / 2 / dv)
    c = nu * dt * np.ones(nv - 1) * (-(v0 ** 2.0) / dv ** 2.0 - vax[1:] / 2 / dv)
    leftside = np.diag(a, -1) + np.diag(1 + b, 0) + np.diag(c, 1)

    return leftside


def take_collision_step(leftside, f):
    """
    Just solves a tridiagonal system here.

    :param leftside:
    :param f_t:
    :return:
    """
    return np.linalg.solve(leftside, f)
