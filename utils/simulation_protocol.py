from enum import Enum

import numpy as np


## adjustable temporal resolution
# samples = time * sampling_factor
sampling_factor = 1000
samples_total = 310 * sampling_factor


class P1ModelIdentifier(Enum):
    MODEL_1 = "P1.dual"
    MODEL_2 = "P1.dual-vm"
    MODEL_3 = "P1.mc-vm"
    DUMMY_MODEL = "Dummy Model"


def create_piezo1_simulation_protocol(protocol_name: P1ModelIdentifier) -> dict:
    ## Mechanical stimuli (default: Dummy Model)
    if protocol_name == P1ModelIdentifier.MODEL_1:
        mech_stimuli = (2e-6, 4e-6, 6e-6)
    elif protocol_name == P1ModelIdentifier.MODEL_2:
        mech_stimuli = (0.4, 0.6, 0.8)
    elif protocol_name == P1ModelIdentifier.MODEL_3 or protocol_name == P1ModelIdentifier.DUMMY_MODEL:
        mech_stimuli = (10, 40, 70)
    else:
        raise ValueError(f"Unknown protocol name: {protocol_name}")
    
    piezo_signal = np.zeros(samples_total)
    piezo_signal[44 * sampling_factor : int(44.25 * sampling_factor)] = mech_stimuli[0]
    piezo_signal[69 * sampling_factor : int(69.25 * sampling_factor)] = mech_stimuli[1]
    piezo_signal[94 * sampling_factor : int(94.25 * sampling_factor)] = mech_stimuli[2]
    piezo_signal[144 * sampling_factor : int(144.25 * sampling_factor)] = mech_stimuli[0]
    piezo_signal[169 * sampling_factor : int(169.25 * sampling_factor)] = mech_stimuli[1]
    piezo_signal[194 * sampling_factor : int(194.25 * sampling_factor)] = mech_stimuli[2]
    piezo_signal[244 * sampling_factor : int(244.25 * sampling_factor)] = mech_stimuli[0]
    piezo_signal[269 * sampling_factor : int(269.25 * sampling_factor)] = mech_stimuli[1]
    piezo_signal[294 * sampling_factor : int(294.25 * sampling_factor)] = mech_stimuli[2]

    ## Glutamate stimuli (same for every model)
    spike_train = 2e-6 * np.ones(samples_total)
    spike_train[13 * sampling_factor : 25 * sampling_factor] = 1e-3 #[M]
    spike_train[38 * sampling_factor : 50 * sampling_factor] = 1e-3 #[M]
    spike_train[63 * sampling_factor : 75 * sampling_factor] = 1e-3 #[M]
    spike_train[88 * sampling_factor : 100 * sampling_factor] = 1e-3 #[M]
    spike_train[113 * sampling_factor : 125 * sampling_factor] = 6e-3 #[M]
    spike_train[138 * sampling_factor : 150 * sampling_factor] = 6e-3 #[M]
    spike_train[163 * sampling_factor : 175 * sampling_factor] = 6e-3 #[M]
    spike_train[188 * sampling_factor : 200 * sampling_factor] = 6e-3 #[M]

    return {"spike_train": spike_train, "mech_stimuli": piezo_signal}
