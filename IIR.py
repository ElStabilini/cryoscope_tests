import numpy as np


def step(t: np.array, start):
    return np.where(t < start, 0, 1)


def direct_model_IIR(params, t, start):
    g, tau, A = params
    return g * (1 + A * np.exp(-(t - start) / tau)) * step(t)


def filter(params, t, start):
    g, tau, A = params
    return g * (1 + A * np.exp(-(t - start) / tau))


def inverse_model_IIR(params, t, start, data):
    g, tau, A = params
    return data / (g * (1 + A * np.exp(-(t - start) / tau)))


def residuals_direct_IIR(params, t, start, data):
    return direct_model_IIR(params, t, start) - data


def residuals_inverse_IIR(params, t, start, data):
    return inverse_model_IIR(params, t, start, data) - step(t, start)
