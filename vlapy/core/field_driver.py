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


def _get_rise_fall_coeff_(normalized_time):
    """
    This is a smooth function that goes from 0 to 1. It is a 5th order polynomial because it satisfies the following
    5 conditions

    value = 0 at t=0 and t=1 (2 conditions)
    first derivative = 0 at t=0 and t=1 (2 conditions)
    value = 0.5 in the middle

    :param normalized_time:
    :return:
    """
    coeff = (
        6.0 * pow(normalized_time, 5.0)
        - 15.0 * pow(normalized_time, 4.0)
        + 10.0 * pow(normalized_time, 3.0)
    )
    return coeff


def get_pulse_coefficient(pulse_profile_dictionary, tt):
    """
    This function generates an envelope that smoothly goes from 0 to 1, and back down to 0.
    It follows the nomenclature introduced to me by working with oscilloscopes.

    The pulse profile dictionary will contain a rise time, flat time, and fall time.
    The rise time is the time over which the pulse goes from 0 to 1
    The flat time is the duration of the flat-top of the pulse
    The fall time is the time over which the pulse goes from 1 to 0.

    :param pulse_profile_dictionary:
    :param tt:
    :return:
    """
    start_time = pulse_profile_dictionary["start_time"]
    end_time = (
        start_time
        + pulse_profile_dictionary["rise_time"]
        + pulse_profile_dictionary["flat_time"]
        + pulse_profile_dictionary["fall_time"]
    )

    this_pulse = 0

    if (tt > start_time) and (tt < end_time):
        rise_time = pulse_profile_dictionary["rise_time"]
        flat_time = pulse_profile_dictionary["flat_time"]
        fall_time = pulse_profile_dictionary["fall_time"]
        end_rise = start_time + rise_time
        end_flat = start_time + rise_time + flat_time

        if tt <= end_rise:
            normalized_time = (tt - start_time) / rise_time
            this_pulse = _get_rise_fall_coeff_(normalized_time)

        elif (tt > end_rise) and (tt < end_flat):
            this_pulse = 1.0

        elif tt >= end_flat:
            normalized_time = (tt - end_flat) / fall_time
            this_pulse = 1.0 - _get_rise_fall_coeff_(normalized_time)

        this_pulse *= pulse_profile_dictionary["a0"]

    return this_pulse


def get_driver_function(x, pulse_dictionary):
    def driver_function(current_time):
        total_field = np.zeros(x.size)
        envelope = 0.0
        kk = 0.0
        ww = 0.0

        for this_pulse in list(pulse_dictionary.keys()):
            kk = pulse_dictionary[this_pulse]["k0"]
            ww = pulse_dictionary[this_pulse]["w0"]

            envelope = get_pulse_coefficient(
                pulse_profile_dictionary=pulse_dictionary[this_pulse], tt=current_time
            )

        if np.abs(envelope) > 0.0:
            total_field += envelope * np.cos(kk * x - ww * current_time)

        return total_field

    return driver_function


def make_driver_array(function, x_axis, time_axis):
    driver_array = np.zeros(time_axis.shape + x_axis.shape)
    for i in range(time_axis.size):
        driver_array[
            i,
        ] = function(x_axis, time_axis[i])

    return driver_array


def get_driver_array_using_function(x, t, pulse_dictionary):
    """

    :param driver_function:
    :param x:
    :param t:
    :return:
    """

    def driver_function(xx, current_time):
        total_field = np.zeros(xx.size)
        envelope = 0.0
        kk = 0.0
        ww = 0.0

        for this_pulse in list(pulse_dictionary.keys()):
            kk = pulse_dictionary[this_pulse]["k0"]
            ww = pulse_dictionary[this_pulse]["w0"]

            envelope = get_pulse_coefficient(
                pulse_profile_dictionary=pulse_dictionary[this_pulse], tt=current_time
            )

        if np.abs(envelope) > 0.0:
            total_field += envelope * np.cos(kk * xx - ww * current_time)

        return total_field

    return make_driver_array(driver_function, x, t)
