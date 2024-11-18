# -*- coding: utf-8 -*-
"""
This file contains Qudi methods for handling real-world values with units.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-core/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

__all__ = ('create_formatted_output', 'get_relevant_digit', 'get_si_norm', 'get_unit_prefix_dict',
           'round_value_to_error', 'ScaledFloat')

import math
import numpy as np
try:
    import pyqtgraph.functions as fn
except ImportError:
    fn = None


def get_unit_prefix_dict():
    """Return the dictionary, which assigns the prefix of a unit to its proper order of magnitude.

    Parameters
    ----------
    None

    Returns
    -------
    dict 
        keys are string prefix and values are magnitude values.
    """
    unit_prefix_dict = {
        'y': 1e-24,
        'z': 1e-21,
        'a': 1e-18,
        'f': 1e-15,
        'p': 1e-12,
        'n': 1e-9,
        'µ': 1e-6,
        'm': 1e-3,
        '': 1,
        'k': 1e3,
        'M': 1e6,
        'G': 1e9,
        'T': 1e12,
        'P': 1e15,
        'E': 1e18,
        'Z': 1e21,
        'Y': 1e24
    }
    return unit_prefix_dict


class ScaledFloat(float):
    """
    Format code 'r' for scaled output.

    Examples
    --------
    '{:.0r}A'.format(ScaledFloat(50))       --> 50 A
    '{:.1r}A'.format(ScaledFloat(1.5e3))    --> 1.5 kA
    '{:.1r}A'.format(ScaledFloat(2e-3))     --> 2.0 mA
    '{:rg}A'.format(ScaledFloat(2e-3))      --> 2 mA
    '{:rf}A'.format(ScaledFloat(2e-3))      --> 2.000000 mA
    """

    @property
    def scale(self):
        """
        Returns the scale. (No prefix if 0)

        Examples
        --------
        1e-3: m
        1e6: M
        """

        # Zero makes the log crash and should not have a prefix
        if self == 0:
            return ''

        exponent = math.floor(math.log10(abs(self)) / 3)
        if exponent < -8:
            exponent = -8
        if exponent > 8:
            exponent = 8
        prefix = 'yzafpnµm kMGTPEZY'
        return prefix[8 + exponent].strip()

    @property
    def scale_val(self):
        """ Returns the scale value which can be used to devide the actual value

        Examples
        --------
        m: 1e-3
        M: 1e6
        """
        scale_str = self.scale
        return get_unit_prefix_dict()[scale_str]

    def __format__(self, fmt):
        """
        Fromats the string using format fmt.

        r for scaled output.

        Parameters
        ----------
        fmt : str 
            format string
        """
        autoscale = False
        if len(fmt) >= 2:
            if fmt[-2] == 'r':
                autoscale = True
                fmt = fmt[:-2] + fmt[-1]
            elif fmt[-1] == 'r':
                autoscale = True
                fmt = fmt[:-1] + 'f'
        elif fmt[-1] == 'r':
            autoscale = True
            fmt = fmt[:-1] + 'f'
        if autoscale:
            scale = self.scale
            if scale == 'u':
                index = 'micro'
            else:
                index = scale
            value = self / get_unit_prefix_dict()[index]
            return '{:s} {:s}'.format(value.__format__(fmt), scale)
        else:
            return super().__format__(fmt)


def create_formatted_output(param_dict, num_sig_digits=5):
    """ Display a parameter set nicely in SI units.

    Parameters
    ----------
    param_dict : dict
        Dictionary with entries being dictionaries with two needed keywords 'value' and 'unit' and one
        optional keyword 'error'. Add the proper items to the specified keywords.
        Note that if no error is specified, no proper rounding (and therefore displaying) can be
        guaranteed.
    num_sig_digits : int, optional
        The number of significant digits will be taken if the rounding procedure was not
        successful at all. Default is 5.

    Returns
    -------
    str
        A nicely formatted string.
        
    Notes
    -----
    The absolute tolerance to a zero is set to 1e-18.
        
    Examples
    --------

    Example of a param dict:
    
    param_dict = {'Rabi frequency': {'value': 123.43, 'error': 0.321, 'unit': 'Hz'},
                  'ODMR contrast':  {'value': 2.563423, 'error': 0.523, 'unit': '%'},
                  'Fidelity':       {'value': 0.783, 'error': 0.2222, 'unit': ''}}
    
    If you want to access the value of the Fidelity, then you can do that via:
    
    >>> param_dict['Fidelity']['value']
    
    or on the error of the ODMR contrast:
    
    >>> param_dict['ODMR contrast']['error']
    """
    if fn is None:
        raise RuntimeError('Function "create_formatted_output" requires pyqtgraph.')

    output_str = ''
    atol = 1e-18    # absolute tolerance for the detection of zero.

    for entry in param_dict:
        if param_dict[entry].get('error') is not None:

            value, error, digit = round_value_to_error(
                param_dict[entry]['value'], param_dict[entry]['error'])

            if (np.isclose(value, 0.0, atol=atol)
                    or np.isnan(error)
                    or np.isclose(error, 0.0, atol=atol)
                    or np.isinf(error)):
                sc_fact, unit_prefix = fn.siScale(param_dict[entry]['value'])
                str_val = '{0:.{1}e}'.format(
                    param_dict[entry]['value'], num_sig_digits - 1)
                if np.isnan(float(str_val)):
                    value = np.NAN
                elif np.isinf(float(str_val)):
                    value = np.inf
                else:
                    value = float('{0:.{1}e}'.format(
                        param_dict[entry]['value'], num_sig_digits - 1))

            else:
                # the factor 10 moves the displayed digit by one to the right,
                # so that the values from 100 to 0.1 are displayed within one
                # range, rather then from the value 1000 to 1, which is
                # default.
                sc_fact, unit_prefix = fn.siScale(error * 10)
            output_str += '{0}: ({1} \u00B1 {2}) {3}{4} \n'.format(entry, round(value * sc_fact,
                                                                                num_sig_digits - 1),
                                                                   round(error * sc_fact,
                                                                         num_sig_digits - 1),
                                                                   unit_prefix,
                                                                   param_dict[entry]['unit'])
        else:
            output_str += '{0}: '.format(entry) + fn.siFormat(param_dict[entry]['value'],
                                                              precision=num_sig_digits,
                                                              suffix=param_dict[entry][
                                                                  'unit']) + ' (fixed) \n'
    return output_str


def round_value_to_error(value, error):
    """ The scientifically correct way of rounding a value according to an error.

    Parameters
    ----------
    value : float or int 
        the measurement value
    error : float or int
        the error for that measurement value

    Returns
    -------
    tuple
        A tuple containing the following elements:
        
        float
            The rounded value according to the error.
        float
            The rounded error.
        int
            The digit to which the rounding procedure was performed. A positive
            number indicates the position of the digit right from the comma, zero means
            the first digit left from the comma, and negative numbers are the digits left
            from the comma. This follows the convention used in the native `round` method
            and `numpy.round`.

    Notes
    -----
    - The input type of `value` or `error` will not be changed. If `float` is the input, `float` will be the output; the same applies to `integer`.
    - This method does not return strings, as each display method might want to display the rounded values in a different way (in exponential representation, in a different magnitude, etc.).
    - This function can handle an invalid error, i.e., if the error is zero, NaN, or infinite. The absolute tolerance to detect a number as zero is set to 1e-18.

    Procedure explanation:
    The scientific way of displaying a measurement result in the presence of an error is applied here. It follows this procedure:
    Take the first leading non-zero number in the error value and check whether the number is a digit within 3 to 9. If so, the rounding value
    is the specified digit. Otherwise, if the first leading digit is 1 or 2, then the next right digit is the rounding value.
    The error is rounded according to that digit, and the same applies to the value.

    Examples
    --------

    Example 1:

    >>> x_meas = 2.05650234
    >>> delta_x = 0.0634
    >>> result = some_function(x_meas, delta_x)
    >>> print(result)
    (2.06, 0.06, 2)

    Example 2:

    >>> x_meas = 0.34545
    >>> delta_x = 0.19145
    >>> result = some_function(x_meas, delta_x)
    >>> print(result)
    (0.35, 0.19, 2)

    Example 3:

    >>> x_meas = 239579.23
    >>> delta_x = 1289.234
    >>> result = some_function(x_meas, delta_x)
    >>> print(result)
    (239600.0, 1300.0, -2)

    Example 4:

    >>> x_meas = 961453
    >>> delta_x = 3789
    >>> result = some_function(x_meas, delta_x)
    >>> print(result)
    (961000, 4000, -3)
    """

    atol = 1e-18    # absolute tolerance for the detection of zero.

    # check if error is zero, since that is an invalid input!
    if np.isclose(error, 0.0, atol=atol) or np.isnan(error) or np.isinf(error):
        # self.log.error('Cannot round to the error, since either a zero error ')
        # logger.warning('Cannot round to the error, since either a zero error '
        #            'value was passed for the number {0}, or the error is '
        #            'NaN: Error value: {1}. '.format(value, error))

        # set the round digit to float precision
        round_digit = -12

        return value, error, round_digit

    # error can only be positive!
    log_val = np.log10(abs(error))

    if log_val < 0:
        round_digit = -(int(log_val) - 1)
    else:
        round_digit = -(int(log_val))

    first_err_digit = '{:e}'.format(error)[0]

    if first_err_digit in ('1', '2'):
        round_digit += 1

    # Use the python round function, since np.round uses the __repr__ conversion
    # function which shows enough digits to unambiguously identify the number.
    # But the __str__ conversion should round the number to a reasonable number
    # of digits, which is the standard output of the python round function.
    # Therefore take the python round function.

    return round(value, round_digit), round(error, round_digit), round_digit


def get_relevant_digit(entry):
    """ By using log10, abs and int operations, the proper relevant digit is
        obtained.

    Parameters
    ----------
    entry : float
        
    Returns
    -------
    int
        the leading relevant exponent
    """
    # the log10 can only be calculated of a positive number.
    entry = np.abs(entry)

    # the log of zero crashes, so return 0
    if entry == 0:
        return 0

    if np.log10(entry) >= 0:
        return int(np.log10(entry))
    else:
        # catch the asymmetric behaviour of the log and int operation.
        return int(int(np.abs(np.log10(entry))) + 1 + np.log10(entry)) - (
                    int(np.abs(np.log10(entry))) + 1)


def get_si_norm(entry):
    """ A rather different way to display the value in SI notation.

    Parameters
    ----------
    entry : float 
        the float number from which normalization factor should
        be obtained.
        

    Returns
    -------
    tuple
        A tuple containing:
        
        norm_val : float
            The value in a normalized representation.
        
        normalization : float
            The factor by which to divide the number.
    """
    val = get_relevant_digit(entry)
    fact = int(val / 3)
    power = int(3 * fact)
    norm = 10 ** power

    return entry / norm, norm
