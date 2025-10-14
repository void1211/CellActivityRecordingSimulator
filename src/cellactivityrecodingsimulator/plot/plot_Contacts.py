from typing import List
import matplotlib.pyplot as plt
from ..Contact import Contact
from probeinterface import Probe
from probeinterface.plotting import plot_probe

def plot_Contacts(
    contacts: List[Contact],
    ax: plt.Axes = None,
    with_id: bool = False,
    with_device_index: bool = False,
    with_probe: bool = False,
    probe: Probe = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()

    if with_probe:
        plot_probe(probe, ax=ax)

    for contact in contacts:
        ax.scatter(contact.x, contact.y, marker='o', color='blue')
        if with_id:
            ax.text(contact.x, contact.y, f'{contact.id}', color='red')
        if with_device_index:
            ax.text(contact.x, contact.y, f'{contact.device_index}', color='green')

    fig.tight_layout()

