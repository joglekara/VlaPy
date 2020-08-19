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

EPSILON = np.float64(1e-6)
gamma = np.array([0.1, 0.6, 0.3])


def get_triple_thing(coeffs, start_indices, stop_indices, f):
    return np.sum(
        f[None, :, start_indices:stop_indices] * coeffs[:, None, None], axis=0
    )


def get_beta_plus_matrices(nv):
    # first index is beta plus/minus
    # 2nd - beta plus 0/1/2
    # 3rd - beta plus 0 1st term/2nd term
    # 4th - beta plus 0 1st term 1st/2nd/3rd coefficient
    beta_coeffs = np.zeros((2, 3, 2, 3))

    # first index - beta plus/minus
    # 2nd - beta plus, subscript 0/1/2
    # 3rd - beta plus, 0, 1st term/2nd term
    # 4th - beta plus, 0, 1st term, 1st/2nd/3rd coefficient
    # 5th - beta plus, 0, 1st term, 1st coefficient, start index
    beta_inds = np.zeros_like((2, 3, 2, 3, 2))

    beta_coeffs[:, :, 0, :] = np.array([1.0, -2.0, 1.0])[None, None, None, :]
    beta_coeffs[0, 0, 1, :] = np.array([1.0, -4.0, 3.0])
    beta_coeffs[0, 1, 1, :] = np.array([1.0, 0.0, -1])
    beta_coeffs[0, 2, 1, :] = np.array([3.0, -4.0, 1])
    beta_coeffs[1, 0, 1, :] = beta_coeffs[0, 2, 1, :]
    beta_coeffs[1, 1, 1, :] = beta_coeffs[0, 1, 1, :]
    beta_coeffs[1, 2, 1, :] = beta_coeffs[0, 0, 1, :]

    ###
    beta_inds[0, :] = np.array([0, 1, 2])
    beta_inds[1, :] = np.array([0, 1, 2])
    beta_inds[0, 1, 0, :] = np.array([0, 1, 2])
    ###

    return (
        get_beta_plus_0_matrices(),
        get_beta_plus_1_matrices(),
        get_beta_plus_2_matrices(),
    )


def get_beta_minus_matrices(nv):
    def get_beta_minus_0_matrices():
        first_term = (
            np.diag(np.ones(nv - 1), k=1)
            + np.diag(-2.0 * np.ones(nv - 2), k=2)
            + np.diag(np.ones(nv - 3), k=3)
            + np.diag(np.zeros(nv), k=0)
        )
        second_term = (
            np.diag(3.0 * np.ones(nv - 1), k=1)
            + np.diag(-4.0 * np.ones(nv - 2), k=2)
            + np.diag(np.ones(nv - 3), k=3)
            + np.diag(np.zeros(nv), k=0)
        )

        return first_term, second_term

    def get_beta_minus_1_matrices():
        first_term = (
            np.diag(np.ones(nv), k=0)
            + np.diag(-2.0 * np.ones(nv - 1), k=1)
            + np.diag(np.ones(nv - 2), k=2)
        )
        second_term = np.diag(np.ones(nv), k=0) + np.diag(np.ones(nv - 2), k=2)

        return first_term, second_term

    def get_beta_minus_2_matrices():
        first_term = (
            np.diag(np.ones(nv - 1), k=-1)
            + np.diag(-2.0 * np.ones(nv), k=0)
            + np.diag(np.ones(nv - 1), k=1)
        )
        second_term = (
            np.diag(np.ones(nv - 1), k=-1)
            + np.diag(-4.0 * np.ones(nv), k=0)
            + np.diag(3.0 * np.ones(nv - 1), k=1)
        )

        return first_term, second_term

    return (
        get_beta_minus_0_matrices(),
        get_beta_minus_1_matrices(),
        get_beta_minus_2_matrices(),
    )


def get_overall_stencils(nv):
    def get_first_term_plus_matrix():
        return (
            np.diag(2.0 / 6.0 * np.ones(nv - 2), k=-2)
            + np.diag(-7.0 / 6.0 * np.ones(nv - 1), k=-1)
            + np.diag(11.0 / 6.0 * np.ones(nv), k=0)
        )

    def get_second_term_plus_matrix():
        return (
            np.diag(-1.0 / 6.0 * np.ones(nv - 1), k=-1)
            + np.diag(5.0 / 6.0 * np.ones(nv), k=0)
            + np.diag(2.0 / 6.0 * np.ones(nv - 1), k=1)
        )

    def get_third_term_plus_matrix():
        return (
            np.diag(2.0 / 6.0 * np.ones(nv), k=0)
            + np.diag(5.0 / 6.0 * np.ones(nv - 1), k=1)
            + np.diag(-1.0 / 6.0 * np.ones(nv - 2), k=2)
        )

    def get_first_term_minus_matrix():
        return (
            np.diag(-1.0 / 6.0 * np.ones(nv - 1), k=-1)
            + np.diag(-5.0 / 6.0 * np.ones(nv), k=0)
            + np.diag(2.0 / 6.0 * np.ones(nv - 1), k=-1)
        )

    def get_second_term_minus_matrix():
        return (
            np.diag(2.0 / 6.0 * np.ones(nv), k=0)
            + np.diag(5.0 / 6.0 * np.ones(nv - 1), k=1)
            + np.diag(-1.0 / 6.0 * np.ones(nv - 2), k=2)
        )

    def get_third_term_minus_matrix():
        return (
            np.diag(11.0 / 6.0 * np.ones(nv - 1), k=1)
            + np.diag(-7.0 / 6.0 * np.ones(nv - 2), k=2)
            + np.diag(2.0 / 6.0 * np.ones(nv - 3), k=3)
            + np.diag(np.zeros(nv), k=0)
        )

    return (
        get_first_term_plus_matrix(),
        get_second_term_plus_matrix(),
        get_third_term_plus_matrix(),
        get_first_term_minus_matrix(),
        get_second_term_minus_matrix(),
        get_third_term_minus_matrix(),
    )


def get_weights(alpha):
    return alpha / np.sum(alpha)


def get_edfdv_linearized_weno5(nv, dv):

    (
        beta_plus_0_matrices,
        beta_plus_1_matrices,
        beta_plus_2_matrices,
    ) = get_beta_plus_matrices(nv=nv)

    (
        beta_minus_0_matrices,
        beta_minus_1_matrices,
        beta_minus_2_matrices,
    ) = get_beta_minus_matrices(nv=nv)

    (
        first_term_plus_matrix,
        second_term_plus_matrix,
        third_term_plus_matrix,
        first_term_minus_matrix,
        second_term_minus_matrix,
        third_term_minus_matrix,
    ) = get_overall_stencils(nv=nv)

    def _beta_(f):
        beta_0_plus = (
            13.0 / 12.0 * (beta_plus_0_matrices[0][None,] * f[:, None, :]) ** 2.0
            + 0.25 * (beta_plus_0_matrices[1][None,] * f[:, None, :]) ** 2.0
        )

        beta_1_plus = (
            13.0 / 12.0 * (beta_plus_1_matrices[0][None,] * f[:, None, :]) ** 2.0
            + 0.25 * (beta_plus_1_matrices[1][None,] * f[:, None, :]) ** 2.0
        )

        beta_2_plus = (
            13.0 / 12.0 * (beta_plus_2_matrices[0][None,] * f[:, None, :]) ** 2.0
            + 0.25 * (beta_plus_2_matrices[1][None,] * f[:, None, :]) ** 2.0
        )

        beta_0_minus = (
            13.0 / 12.0 * (beta_minus_0_matrices[0][None,] * f[:, None, :]) ** 2.0
            + 0.25 * (beta_minus_0_matrices[1][None,] * f[:, None, :]) ** 2.0
        )

        beta_1_minus = (
            13.0 / 12.0 * (beta_minus_1_matrices[0][None,] * f[:, None, :]) ** 2.0
            + 0.25 * (beta_minus_1_matrices[1][None,] * f[:, None, :]) ** 2.0
        )

        beta_2_minus = (
            13.0 / 12.0 * (beta_minus_2_matrices[0][None,] * f[:, None, :]) ** 2.0
            + 0.25 * (beta_minus_2_matrices[1][None,] * f[:, None, :]) ** 2.0
        )

        return (
            beta_0_plus,
            beta_1_plus,
            beta_2_plus,
            beta_0_minus,
            beta_1_minus,
            beta_2_minus,
        )

    def update_edfdv_lw5(f, e, dt):
        beta = _beta_(f)
        alpha_plus = [
            gamma_i / (EPSILON + beta_i) ** 2.0
            for gamma_i, beta_i in zip(gamma, beta[:3])
        ]
        alpha_minus = [
            gamma_i / (EPSILON + beta_i) ** 2.0
            for gamma_i, beta_i in zip(gamma, beta[3:])
        ]
        weights_plus = [get_weights(alpha=alpha_i) for alpha_i in alpha_plus]
        weights_minus = [get_weights(alpha=alpha_i) for alpha_i in alpha_minus]

        f_plus = (
            weights_plus[0] * np.dot(first_term_plus_matrix[None,], f[:, None, :])
            + weights_plus[1] * np.dot(second_term_plus_matrix[None,], f[:, None, :])
            + weights_plus[2] * np.dot(third_term_plus_matrix[None,], f[:, None, :])
        )

        f_minus = (
            weights_minus[0] * np.dot(first_term_minus_matrix[None,], f[:, None, :])
            + weights_minus[1] * np.dot(second_term_minus_matrix[None,], f[:, None, :])
            + weights_minus[2] * np.dot(third_term_minus_matrix[None,], f[:, None, :])
        )

        e_plus = np.maximum(e, 0)
        e_minus = np.minimum(e, 0)

        plus_out = np.zeros_like(f)
        plus_out[:, :-1] = np.diff(f_plus, axis=1) / dv

        minus_out = np.zeros_like(f)
        minus_out[:, :-1] = np.diff(f_minus, axis=1) / dv

        return dt * (e_plus[:, None] * plus_out + e_minus[:, None] * minus_out)

    return update_edfdv_lw5
