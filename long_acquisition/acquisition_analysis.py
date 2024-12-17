import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import cma

from scipy.optimize import least_squares, minimize, Bounds
from pathlib import Path

from cryoscope_scripts import _fit, load_cryoscope_data

from qibocal.protocols.two_qubit_interaction.cryoscope import CryoscopeResults

from IIR import (
    step,
    inverse_model_IIR,
    residuals_inverse_IIR,
    multi_exponential_IIR,
    residuals_multi_exponential,
    iter_filter_application,
)

from tools import (
    plot_reconstructed_data,
    plot_signal_filtered,
    plot_signal_filtered_iter,
    Signal,
)

DIRECT = False
DATA_START = 0
POLYORDER = 2
PHASE_SCALING = 2 * np.pi
MODE = "nearest"


def buildplot(
    data: list[float], color: str, label: str, title: str, save: bool, folder: str
):
    plt.figure(figsize=(12, 6))
    plt.plot(data, color=color, label=label)
    plt.xlabel("time [ns]")
    plt.ylabel("Signal")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
    if save:
        plt.savefig(f"{folder}/{title}.png")
    plt.close()


def plotWL(data: list[CryoscopeResults], target: str, title: str, save: bool):

    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))

    plt.figure(figsize=(12, 6))

    for i, cryo_result in enumerate(data):
        plt.plot(cryo_result.step_response[target], color=colors[i], label=f"{i+3}")
    plt.xlabel("time [ns]")
    plt.ylabel("Signal")
    plt.title("Reconstructed signal")
    plt.legend()
    plt.grid()
    plt.show()
    if save:
        plt.savefig(f"WL_study.png")
    plt.close()


def main():
    data_path = (
        Path.cwd().parent.parent
        / "cryo_material"
        / "long_acquisition"
        / "data"
        / "cryoscope-0"
    )
    data_json = data_path / "data.json"

    with open(data_json, "rb") as file:
        data = json.load(file)

    flux_amplitude = data['"flux_pulse_amplitude"']
    cryoscope_data = load_cryoscope_data(data_path / "data.npz", flux_amplitude)

    t = np.arange(0, 99, 1)
    start = 10

    # making plots for different window length
    cryo_results = []
    for i in range(3, 10):
        result, _, _ = _fit(cryoscope_data, True, True, i, False)
        cryo_results.append(result)

    with open("CryoscopeResults.pkl", "wb") as pkl_file:
        pickle.dump(cryo_results, pkl_file)

    plotWL(cryo_results, "D1", "WL study - step response", True)


if __name__ == "__main__":
    main()
