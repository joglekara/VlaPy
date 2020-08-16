import numpy
import scipy
from scipy import optimize, special


def plasma_dispersion(value):
    """
    This function leverages the Fadeeva function in scipy to calculate the Z function

    :param value:
    :return:
    """
    return scipy.special.wofz(value) * numpy.sqrt(numpy.pi) * 1j


def plasma_dispersion_prime(value):
    """
    This is a simple relation for Z-prime, which happens to be directly proportional to Z

    :param value:
    :return:
    """
    return -2.0 * (1.0 + value * plasma_dispersion(value))


def get_roots_to_electrostatic_dispersion(
    wp_e, vth_e, k0, maxwellian_convention_factor=2.0, initial_root_guess=None
):
    """
    This function calculates the root of the plasma dispersion relation

    :param wp_e:
    :param vth_e:
    :param k0:
    :param maxwellian_convention_factor:
    :param initial_root_guess:
    :return:
    """
    if initial_root_guess is None:
        initial_root_guess = numpy.sqrt(wp_e ** 2.0 + 3 * (k0 * vth_e) ** 2.0)

    chi_e = numpy.power((wp_e / (vth_e * k0)), 2.0) / maxwellian_convention_factor

    def plasma_epsilon1(x):
        val = 1.0 - chi_e * plasma_dispersion_prime(x)
        return val

    epsilon_root = scipy.optimize.newton(plasma_epsilon1, initial_root_guess)

    return epsilon_root * k0 * vth_e * numpy.sqrt(maxwellian_convention_factor)
