import matplotlib.pyplot as plt
from numba.core.types import np_uint16
import numpy as np
from typing import List
from ..Contact import Contact
from ..GroundTruthUnitsObject import GTUnitsObject

def plot_Spikes(
    contacts: List[Contact],
    gt_units: GTUnitsObject,
):

    n_contact = len(contacts)
    n_unit = gt_units.get_units_num()

    for i in range(n_contact):
        fig, ax = plt.subplots(n_unit // 4, 4, figsize=(12, 8))
        spikes = []
        for j in range(n_unit):
            unit = gt_units.units[j]
            contact = contacts[i]
            signal = contact.get_signal("filtered", 30000)
            spike_times = unit.get_spike_times()
            for spike_time in spike_times:
                spikes.append(signal[spike_time-60:spike_time+60])
            ax[j // 4, j % 4].plot(np.array(spikes).T)
    fig.tight_layout()
    plt.show()
    return fig, ax
