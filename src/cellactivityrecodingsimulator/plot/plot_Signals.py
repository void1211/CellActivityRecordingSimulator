import matplotlib.pyplot as plt
from ..Contact import Contact
from typing import List


def plot_Signals(
    contacts: List[Contact],
    ax: plt.Axes = None,
    time: List[float] = [0, 1],
    signal_type: str = "raw",
    fs: float = 30_000,
    is_title: bool = True,
    title: str = "Signal",
    is_xlabel: bool = True,
    is_ylabel: bool = True,
    ylabel: str = "Amplitude",
    xlabel: str = "Time (samples)",
):
    if ax is None:
        fig, axes = plt.subplots(len(contacts), 1, figsize=(12, 8), sharex=True)
        if len(contacts) == 1:
            axes = [axes]
    else:
        fig = ax.get_figure()
        axes = [ax]

    for i, contact in enumerate(contacts):
        if ax is None:
            current_ax = axes[i]
        else:
            current_ax = ax
            
        signal = contact.get_signal(signal_type, fs)
        start_idx = int(time[0] * fs)
        end_idx = int(time[1] * fs)
        
        if len(signal) > end_idx:
            signal_segment = signal[start_idx:end_idx]
        else:
            signal_segment = signal[start_idx:]
            
        current_ax.plot(signal_segment)
        if is_title:
            current_ax.set_title(title)
        if is_ylabel:
            current_ax.set_ylabel(ylabel)
        
    if ax is None:
        if is_xlabel:
            axes[-1].set_xlabel(xlabel)
        fig.tight_layout()
    else:
        if is_xlabel:
            ax.set_xlabel(xlabel)



