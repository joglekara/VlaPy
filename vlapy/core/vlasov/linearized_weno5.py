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


def get_beta_plus_method():
    def get_beta_plus_0(f):
        first_term = np.zeros_like(f)
        second_term = np.zeros_like(f)

        first_term[:, 0] = 13.0 / 12.0 * (f[:, 0]) ** 2.0
        first_term[:, 1] = 13.0 / 12.0 * (-2.0 * f[:, 0] + f[:, 1]) ** 2.0
        first_term[:, 2:] = (
            13.0 / 12.0 * (f[:, :-2] - 2.0 * f[:, 1:-1] + f[:, 2:]) ** 2.0
        )

        second_term[:, 0] = 1.0 / 4.0 * (3.0 * f[:, 0]) ** 2.0
        second_term[:, 1] = 1.0 / 4.0 * (-4.0 * f[:, 0] + 3.0 * f[:, 1]) ** 2.0
        second_term[:, 2:] = (
            1.0 / 4.0 * (f[:, :-2] - 4.0 * f[:, 1:-1] + 3.0 * f[:, 2:]) ** 2.0
        )
        return first_term + second_term

    def get_beta_plus_1(f):

        first_term = np.zeros_like(f)
        second_term = np.zeros_like(f)

        first_term[:, 0] = 13.0 / 12.0 * (-2.0 * f[:, 0] + f[:, 1]) ** 2.0
        first_term[:, 1:-1] = (
            13.0 / 12.0 * (f[:, :-2] - 2.0 * f[:, 1:-1] + f[:, 2:]) ** 2.0
        )
        first_term[:, -1] = 13.0 / 12.0 * (f[:, -2] - 2.0 * f[:, -1]) ** 2.0

        second_term[:, 0] = 1.0 / 4.0 * (-f[:, 1]) ** 2.0
        second_term[:, 1:-1] = 1.0 / 4.0 * (f[:, :-2] - f[:, 2:]) ** 2.0
        second_term[:, -1] = 1.0 / 4.0 * (f[:, -2]) ** 2.0

        return first_term + second_term

    def get_beta_plus_2(f):
        first_term[:, :-2] = (
            13.0 / 12.0 * (f[:, :-2] - 2.0 * f[:, 1:-1] + f[:, 2:]) ** 2.0
        )
        second_term[:, :-2] = (
            1.0 / 4.0 * (3.0 * f[:, :-2] - 4.0 * f[:, 1:-1] + f[:, 2:]) ** 2.0
        )

        # TODO Boundaries
        pass
        # return first_term + second_term

    return get_beta_plus_0, get_beta_plus_1, get_beta_plus_2


def get_edfdv_linearized_weno5(dv):
    def _alpha_(gamma, beta):
        return gamma / (EPSILON + beta) ** 2.0

    def update_edfdv_lw5(f, e, dt):
        pass

    pass
