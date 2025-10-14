import matplotlib.pyplot as plt
from ..Contact import Contact
from typing import List



def plot_Signals(
    contacts: List[Contact],
    ax: plt.Axes = None,
    is_show: bool = True,
):
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    ax = fig.add_subplot(6, 1, 1)
    ax.plot(contacts[0].get_signal("raw"))
    ax.set_title("Raw Signal")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (s)")

    fig.tight_layout()