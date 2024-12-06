import numpy as np

from scipy.linalg import lstsq
from scipy._lib._util import float_factorial
from scipy.ndimage import convolve1d
from scipy.signal._arraytools import axis_slice
from scipy.signal._savitzky_golay import _polyder, _fit_edge, _fit_edges_polyfit

def savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0, pos=None, use="conv"):
    """
    Compute the Savitzky-Golay filter coefficients for a causal filter.

    Parameters
    ----------
    window_length : int
        The length of the filter window (must be a positive odd integer).
    polyorder : int
        The order of the polynomial used to fit the samples.
    deriv : int, optional
        The order of the derivative to compute (default is 0).
    delta : float, optional
        The spacing of the samples (default is 1.0).
    pos : int, optional
        Position of the point to compute the filter coefficients for. In
        the causal case, this is the last point of the window.
    use : str, optional
        Determines the output format:
        - "conv": Coefficients are ordered for convolution.
        - "dot": Coefficients are ordered for dot product.

    Returns
    -------
    coeffs : ndarray
        The filter coefficients.
    """
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    # In the causal version, the position defaults to the end of the window.
    if pos is None:
        pos = window_length - 1

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than " "window_length.")

    if use not in ["conv", "dot"]:
        raise ValueError("`use` must be 'conv' or 'dot'")

    if deriv > polyorder:
        coeffs = np.zeros(window_length)
        return coeffs

    # Form the design matrix A. The columns of A are powers of integers
    # from -pos to window_length - pos - 1 (but causal starts from 0).
    x = np.arange(-pos, window_length - pos, dtype=float)

    if use == "conv":
        # Reverse so that result can be used in a convolution.
        x = x[::-1]

    order = np.arange(polyorder + 1).reshape(-1, 1)
    A = x**order

    # y determines which order derivative is returned.
    y = np.zeros(polyorder + 1)
    y[deriv] = float_factorial(deriv) / (delta**deriv)

    # Find the least-squares solution of A*c = y
    coeffs, _, _, _ = lstsq(A, y)

    return coeffs


def causal_savgol_filter(
    x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode="interp", cval=0.0
):
    """
    Apply a Savitzky-Golay filter to the input signal with causal filtering.

    Parameters
    ----------
    x : array_like
        The input data.
    window_length : int
        The length of the filter window (must be a positive odd integer).
    polyorder : int
        The order of the polynomial used to fit the samples.
    deriv : int, optional
        The order of the derivative to compute (default is 0).
    delta : float, optional
        The spacing of the samples (default is 1.0).
    axis : int, optional
        The axis along which to apply the filter (default is -1).
    mode : str, optional
        The mode to handle the boundaries. Supported modes are:
        'mirror', 'constant', 'nearest', 'interp', 'wrap'.
    cval : scalar, optional
        Value to fill past the edges of the array when mode is 'constant'.

    Returns
    -------
    y : ndarray
        The filtered data.
    """
    if mode not in ["mirror", "constant", "nearest", "interp", "wrap"]:
        raise ValueError(
            "mode must be 'mirror', 'constant', 'nearest' " "'wrap' or 'interp'."
        )

    x = np.asarray(x)
    # Ensure that x is either single or double precision floating point.
    if x.dtype != np.float64 and x.dtype != np.float32:
        x = x.astype(np.float64)

    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

    if mode == "interp":
        if window_length > x.shape[axis]:
            raise ValueError(
                "If mode is 'interp', window_length must be less "
                "than or equal to the size of x."
            )

        # Do not pad. Instead, for the elements within `window_length` of
        # the ends of the sequence, use the polynomial that is fitted to
        # the last `window_length` elements.
        y = convolve1d(x, coeffs, axis=axis, mode="constant")
        _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y)
    else:
        # Any mode other than 'interp' is passed on to ndimage.convolve1d.
        y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=cval)

    return y