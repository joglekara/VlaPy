import numpy as np

from tests.extras import canosa_values
from vlapy.diagnostics.z_function import get_roots_to_electrostatic_dispersion


def test_z_solver():
    k0 = canosa_values.k0
    w_real_actual = canosa_values.w_real
    w_imag_actual = canosa_values.w_imag

    for kk, wr, wi in zip(k0, w_real_actual, w_imag_actual):
        answer = get_roots_to_electrostatic_dispersion(wp_e=1.0, vth_e=1.0, k0=kk)

        np.testing.assert_almost_equal(wr + 1j * wi, answer, decimal=4)
