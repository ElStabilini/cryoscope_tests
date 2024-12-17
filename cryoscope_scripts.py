import numpy as np
import os
import json
import pickle

from qibocal.protocols.two_qubit_interaction.cryoscope import (
    CryoscopeData,
    CryoscopeResults,
)
from qibocal.protocols.ramsey.utils import fitting
from scipy.signal import savgol_filter
from causal_savgol import causal_savgol_filter
from dataclasses import dataclass


def load_cryoscope_data(file_path: str, flux_pulse_amplitude: float) -> CryoscopeData:

    cryoscope_data = CryoscopeData(flux_pulse_amplitude=flux_pulse_amplitude)

    data_npz = np.load(file_path)
    D1MX = np.rec.array(data_npz['["D1", "MX"]'])
    D1MY = np.rec.array(data_npz['["D1", "MY"]'])

    data_dict = {("D1", "MX"): D1MX, ("D1", "MY"): D1MY}

    cryoscope_data.data = data_dict

    return cryoscope_data


def _fit(
    data: CryoscopeData, savgol: bool, demod: bool, window_length: int, causal: bool
) -> CryoscopeResults:

    nyquist_order = 0

    fitted_parameters = {}
    detuning = {}
    amplitude = {}
    step_response = {}
    for qubit, setup in data.data:
        qubit_data = data[qubit, setup]
        x = qubit_data.duration
        y = 1 - 2 * qubit_data.prob_1

        popt, _ = fitting(x, y)

        fitted_parameters[qubit, setup] = popt

    qubits = np.unique([i[0] for i in data.data]).tolist()

    for qubit in qubits:

        sampling_rate = 1 / (x[1] - x[0])
        X_exp = 1 - 2 * data[(qubit, "MX")].prob_1
        Y_exp = 1 - 2 * data[(qubit, "MY")].prob_1

        norm_data = X_exp + 1j * Y_exp

        # demodulation frequency found by fitting sinusoidal
        demod_freq = -fitted_parameters[qubit, "MY"][2] / 2 / np.pi * sampling_rate

        # to be used in savgol_filter
        derivative_window_length = window_length / sampling_rate
        derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
        derivative_window_size += (derivative_window_size + 1) % 2

        # find demodulatation frequency
        if demod:
            demod_data = np.exp(2 * np.pi * 1j * x * demod_freq) * (norm_data)
        else:
            demod_data = norm_data

        # compute phase
        phase = np.unwrap(np.angle(demod_data))
        real_phase = phase / (2 * np.pi)

        # compute detuning

        if savgol:
            if causal:
                phase = causal_savgol_filter(
                    phase / (2 * np.pi),
                    window_length=derivative_window_size,
                    polyorder=2,
                    deriv=1,
                    # mode="nearest",
                )
            else:
                phase = savgol_filter(
                    phase / (2 * np.pi),
                    window_length=derivative_window_size,
                    polyorder=2,
                    deriv=1,
                    # mode="nearest",
                )
            raw_detuning = phase * sampling_rate
        else:
            phase = phase / (2 * np.pi)
            raw_detuning = phase * sampling_rate

        # real detuning (reintroducing demod_freq)
        if demod:
            detuning[qubit] = (
                raw_detuning - demod_freq + sampling_rate * nyquist_order
            ).tolist()
        else:
            detuning[qubit] = (raw_detuning + sampling_rate * nyquist_order).tolist()

        # params from flux_amplitude_frequency_protocol
        params = [1.9412681243469971, -0.012534948170662627, 0.0005454772278201887]

        # invert frequency amplitude formula
        p = np.poly1d(params)
        amplitude[qubit] = [max((p - freq).roots).real for freq in detuning[qubit]]

        # compute step response
        step_response[qubit] = (
            np.array(amplitude[qubit]) / data.flux_pulse_amplitude
        ).tolist()

    return (
        CryoscopeResults(
            amplitude=amplitude,
            detuning=detuning,
            step_response=step_response,
            fitted_parameters=fitted_parameters,
        ),
        raw_detuning,
        real_phase,
    )


def compute_window_length(data: CryoscopeData):

    fitted_parameters = {}
    for qubit, setup in data.data:
        qubit_data = data[qubit, setup]
        x = qubit_data.duration
        y = 1 - 2 * qubit_data.prob_1

        popt, _ = fitting(x, y)

        fitted_parameters[qubit, setup] = popt

    qubits = np.unique([i[0] for i in data.data]).tolist()

    for qubit in qubits:

        sampling_rate = 1 / (x[1] - x[0])

        derivative_window_length = 3 / sampling_rate
        derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
        derivative_window_size += (derivative_window_size + 1) % 2

    return (derivative_window_size, sampling_rate)


def save_fit_data(
    results: CryoscopeResults,
    raw_detuning,
    phase,
    name: str,
    savgol: bool,
    demod: bool,
    window_length: int,
    causal: bool,
):

    os.makedirs(name, exist_ok=True)

    metadata = {
        "savgol": savgol,
        "demodulation": demod,
        "window_length": window_length,
        "causal_filter": causal,
    }

    with open(os.path.join(name, "metadata.json"), "wb") as json_file:
        json.dump(metadata, json_file, indent=4)

    data = {
        "results": results,
        "raw_detuning": raw_detuning,
        "phase": phase,
    }

    with open(os.path.join(name, "data.pkl"), "wb") as pkl_file:
        pickle.dump(data, pkl_file)


def load_fit_data(name: str):

    if not os.path.exists(name):
        raise FileNotFoundError(f"Directory '{name}' does not exist.")

    with open(os.path.join(name, "metadata.json"), "rb") as json_file:
        metadata = json.load(json_file)

    with open(os.path.join(name, "data.pkl"), "wb") as pkl_file:
        data = pickle.load(pkl_file)

    return metadata, data


def build_phase(time, c, start, sigma):
    phase = []
    for t in time:

        phi = c if t < start else 0.015 * (t - start) + c
        phi += np.random.normal(loc=0, scale=sigma)
        phase.append(phi)

    return phase


@dataclass
class FitResults:
    """Similar to CryoscopeResults but for fictious data (no QubitId ecc..)"""

    detuning: list[float]
    """Expected detuning."""
    amplitude: list[float]
    """Flux amplitude computed from detuning."""
    step_response: list[float]
    """Waveform normalized to 1."""


def pseudo_fit(
    phase_data,
    flux_pulse_amplitude,
    savgol: bool,
    demod: bool,
    window_length: int,
    causal: bool,
) -> FitResults:

    sampling_rate = 1
    # sampling rate, at least for quantum machines should be 1

    # demodulation frequency found by fitting sinusoidal
    # TODO: capire come fare la demodulazione su dati sintetici
    """demod_freq = -fitted_parameters[qubit, "MY"][2] / 2 / np.pi * sampling_rate"""
    # to be used in savgol_filter

    derivative_window_length = window_length / sampling_rate
    derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
    derivative_window_size += (derivative_window_size + 1) % 2
    # find demodulatation frequency
    """if demod:
        demod_data = np.exp(2 * np.pi * 1j * x * demod_freq) * (norm_data)
    else:
        demod_data = norm_data
    # compute phase
    phase = np.unwrap(np.angle(demod_data))"""

    phase = phase_data
    # compute detuning
    if savgol:
        if causal:
            phase = causal_savgol_filter(
                phase / (2 * np.pi),
                window_length=derivative_window_size,
                polyorder=2,
                deriv=1,
                mode="nearest",
            )
        else:
            phase = savgol_filter(
                phase / (2 * np.pi),
                window_length=derivative_window_size,
                polyorder=2,
                deriv=1,
                mode="nearest",
            )
        raw_detuning = phase * sampling_rate
    else:
        phase = phase / (2 * np.pi)
        raw_detuning = phase * sampling_rate

    # real detuning (reintroducing demod_freq)
    """if demod:
        detuning[qubit] = (
            raw_detuning - demod_freq + sampling_rate * nyquist_order
        ).tolist()
    else:"""

    nyquist_order = 0
    detuning = (raw_detuning + sampling_rate * nyquist_order).tolist()

    # params from flux_amplitude_frequency_protocol
    params = [1.9412681243469971, -0.012534948170662627, 0.0005454772278201887]
    # invert frequency amplitude formula
    p = np.poly1d(params)
    amplitude = [max((p - freq).roots).real for freq in detuning]
    # compute step response
    step_response = (np.array(amplitude) / flux_pulse_amplitude).tolist()

    return (
        FitResults(
            amplitude=amplitude,
            detuning=detuning,
            step_response=step_response,
        ),
        raw_detuning,
        phase,
    )
