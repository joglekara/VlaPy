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


def get_driver_function(x, pulse_dictionary, np):
    def driver_function(current_time):
        total_field = np.zeros(x.size)

        for this_pulse in list(pulse_dictionary.keys()):
            kk = pulse_dictionary[this_pulse]["k0"]
            ww = pulse_dictionary[this_pulse]["w0"]
            t_L = pulse_dictionary[this_pulse]["t_L"]
            t_R = pulse_dictionary[this_pulse]["t_R"]
            t_wL = pulse_dictionary[this_pulse]["t_wL"]
            t_wR = pulse_dictionary[this_pulse]["t_wR"]

            envelope = 0.5 * (
                np.tanh((current_time - t_L) / t_wL)
                - np.tanh((current_time - t_R) / t_wR)
            )

            total_field += (
                envelope
                * kk
                * pulse_dictionary[this_pulse]["a0"]
                * np.sin(kk * x - ww * current_time)
            )

        return total_field

    return driver_function


def make_driver_array(function, x_axis, time_axis):
    import numpy as np

    driver_array = np.zeros(time_axis.shape + x_axis.shape)
    for i in range(time_axis.size):
        driver_array[
            i,
        ] = function(time_axis[i])

    return driver_array


def get_driver_array_using_function(x, t, pulse_dictionary, this_np):
    """

    :param driver_function:
    :param x:
    :param t:
    :return:
    """

    driver_function = get_driver_function(x, pulse_dictionary, this_np)

    return make_driver_array(driver_function, x, t)
