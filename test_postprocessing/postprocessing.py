import numpy as np
import json
import pickle
import matplotlib.pyplot as plt

from qibocal.protocols.two_qubit_interaction.cryoscope import (
    CryoscopeParameters,
    CryoscopeResults,
    CryoscopeData,
    #    residuals,
    #    exponential_params,
    filter_calc,
    _fit,
    _plot,
)
from pathlib import Path


def load_cryoscope_data(file_path: str, flux_pulse_amplitude: float) -> CryoscopeData:

    cryoscope_data = CryoscopeData(flux_pulse_amplitude=flux_pulse_amplitude)

    data_npz = np.load(file_path)
    D1MX = np.rec.array(data_npz['["D1", "MX"]'])
    D1MY = np.rec.array(data_npz['["D1", "MY"]'])

    data_dict = {("D1", "MX"): D1MX, ("D1", "MY"): D1MY}

    cryoscope_data.data = data_dict

    return cryoscope_data


def main():
    data_path = (
        Path.cwd().parent.parent
        / "cryo_material"
        / "long_acquisition"
        / "data"
        / "cryoscope-0"
    )
    data_json = data_path / "data.json"
    result_path = Path.cwd().parent / "single_IIR_step.pkl"

    with open(data_json, "rb") as file:
        data = json.load(file)

    # fmt: off
    flux_pulse_amplitude = data["\"flux_pulse_amplitude\""]
    # fmt: on
    cryoscope_data = load_cryoscope_data(data_path / "data.npz", flux_pulse_amplitude)

    cryoscope_results = _fit(cryoscope_data)
    print(
        f"feedback taps: {cryoscope_results.fir}\nfeedforward taps: {cryoscope_results.iir}"
    )
    print(
        f"Amplitude exponential: {cryoscope_results.A}\nTime decay constant: {cryoscope_results.tau}"
    )

    # test that I implemented filter calculation correctly
    # for this optimization results where already stored in a pickle file so I upload them from there

    with open(result_path, "rb") as pickle_file:
        result = pickle.load(pickle_file)

    print(
        f"PREVIOUS RESULTS\nAmplitude exponential: {result.x[2]}\nTime decay constant: {result.x[1]}"
    )


if __name__ == "__main__":
    main()
