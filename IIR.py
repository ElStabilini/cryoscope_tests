import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter, lfilter_zi


def step(t: np.array, start):
    return np.where(t < start, 0, 1)


def direct_model_IIR(params, t, start):
    g, tau, A = params
    return g * (1 + A * np.exp(-(t - start) / tau)) * step(t, start)


def filter(params, t, start):
    g, tau, A = params
    return g * (1 + A * np.exp(-(t - start) / tau))


def inverse_model_IIR(params, t, start, data):
    g, tau, A = params
    return data / (g * (1 + A * np.exp(-(t - start) / tau)))


def multi_exponential_IIR(params, t, start, data):
    g_1, tau_1, A_1, g_2, tau_2, A_2, g_3, tau_3, A_3 = params
    return data / (
        (g_1 * (1 + A_1 * np.exp(-(t - start) / tau_1)))
        * (g_2 * (1 + A_2 * np.exp(-(t - start) / tau_2)))
        * (g_3 * (1 + A_3 * np.exp(-(t - start) / tau_3)))
    )


def residuals_direct_IIR(params, t, start, data):
    return direct_model_IIR(params, t, start) - data


def residuals_inverse_IIR(params, t, start, data):
    return inverse_model_IIR(params, t, start, data) - step(t, start)


def residuals_multi_exponential(params, t, start, data):
    return multi_exponential_IIR(params, t, start, data) - step(t, start)


def iter_filter_application(
    direct: bool, iterations, t, start, init_guess, step_response
):
    results = []
    responses = []

    residuals = residuals_direct_IIR if direct else residuals_inverse_IIR
    model = direct_model_IIR if direct else inverse_model_IIR

    for _ in range(iterations):
        result = least_squares(residuals, init_guess, args=(t, start, step_response))
        results.append(result)
        step_response = model(
            result.x, t, start, step_response
        )  # reference data update
        responses.append(step_response)  # store data at each iteration
    return responses, results


def IIR_filter(coefficients: list[float], signal: list[float], use_zi: bool):
    a0, a1, b0, b1 = coefficients
    a = np.array([a0, a1])
    b = np.array([b0, b1])

    if use_zi:
        zi = lfilter_zi(b, a) * signal[0]
        filtered_signal, _ = lfilter(b, a, signal, zi=zi)

    else:
        filtered_signal = lfilter(b, a, signal)

    return filtered_signal


def residuals_coefficients(coefficients, data, use_zi, t, start):
    return IIR_filter(coefficients, data, use_zi) - step(t, start)


def single_exp_params(params, sampling_rate):
    g, tau, A = params
    alpha = 1 - np.exp(-1 / (sampling_rate * tau * (1 + A)))
    k = A / ((1 + A) * (1 - alpha)) if A < 0 else A / (1 + A - alpha)
    b0 = 1 - k + k * alpha
    b1 = -(1 - k) * (1 - alpha)
    a0 = 1
    a1 = -(1 - alpha)

    a = np.array([a0, a1]) * g
    b = np.array([b0, b1]) * g
    return a, b


"""TODO:
*  I do not need to fit g at each iteration, fitting once is enough (all codes need to be adjusted accordingly)
* 
"""
