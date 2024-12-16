import matplotlib.pyplot as plt
import numpy as np

from IIR import step
from dataclasses import dataclass
from qibocal.protocols.two_qubit_interaction.cryoscope import CryoscopeResults


@dataclass
class Signal:
    signal: list[float]
    """list of samplings of the signal"""
    label: str
    """Name of the signal"""


def plot_signal_filtered(
    filtered_response: Signal, step_response: Signal, t: np.ndarray, start: float
):
    plt.figure(figsize=(12, 6))
    plt.plot(
        filtered_response.signal,
        color="blue",
        label=filtered_response.label,
    )
    plt.plot(step_response.signal, color="red", label=step_response.label)
    plt.plot(t, step(t, start), color="orange", label="ideal signal")
    plt.xlabel("time [ns]")
    plt.ylabel("Signal")
    plt.title("Step response")
    plt.legend()
    plt.grid()
    plt.show()


def plot_signal_filtered_iter(
    filtered_responses: list[Signal], step_response: Signal, t: np.ndarray, start: float
):
    colors = plt.cm.viridis(np.linspace(0, 1, len(filtered_responses)))

    plt.figure(figsize=(12, 6))

    for i, response in enumerate(filtered_responses):
        plt.plot(response.signal, color=colors[i], label=f"response + {i+1} filter")
    plt.plot(step_response.signal, color="red", label=step_response.label)
    plt.plot(t, step(t, start), color="orange", label="ideal signal")
    plt.xlabel("time [ns]")
    plt.ylabel("Signal")
    plt.title("Step response")
    plt.legend()
    plt.grid()
    plt.show()


def plot_reconstructed_data(
    cryoscope_results: CryoscopeResults,
    qubit: str,
    data_start: int,
    t: np.ndarray,
    start: float,
):
    plt.figure(figsize=(12, 6))
    plt.plot(
        cryoscope_results.step_response[qubit][data_start:],
        color="red",
        label="reconstructed signal",
    )
    plt.plot(step(t, start), color="blue", label="ideal signal")
    plt.xlabel("time [ns]")
    plt.ylabel("Signal")
    plt.title("Reconstructed vs ideal signal")
    plt.legend()
    plt.grid()
    plt.show()
