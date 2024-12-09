import numpy as np
import os
import json
import pickle

from qibocal.protocols.two_qubit_interaction.cryoscope import (
    CryoscopeData,
    CryoscopeResults,
)

from cryoscope_scripts import build_phase

sigmas = np.arange(0.0, 0.007, 0.0005)
time = np.arange(0, 70, 1)
c = -np.pi / 8
start = 10

phases = [np.array(build_phase(time, c, start, sigma)) * 2 * np.pi for sigma in sigmas]


## voglio testare l'applicazione del savgol su questi con una serie di combinazioni diverse

## voglio che i risultati siano salvati in una serie di folder

## voglio poterci accedere per poter fare i plot anche sui notebook
