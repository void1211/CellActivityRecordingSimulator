import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..GroundTruthUnitsObject import GTUnitsObject
from probeinterface import Probe
from probeinterface.plotting import plot_probe

def plot_GTUnits(
    GTUnits: GTUnitsObject, 
    ax: plt.Axes = None,
    with_id: bool = False, 
    with_group: bool = False, 
    with_probe: bool = False,
    probe: Probe = None,
    dimension: str = "2D",
    is_show: bool = True,
    ):
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    if dimension == "2D":
        ax = fig.add_subplot(1,1,1)
    elif dimension == "3D":
        ax = fig.add_subplot(1,1,1, projection='3d')

    if with_probe:
        plot_probe(probe, ax=ax)
    
    if dimension == "2D":
        for unit in GTUnits.get_units():
            ax.scatter(unit.x, unit.y, marker='o', color='blue')
            if with_id:
                ax.text(unit.x, unit.y, f'{unit.id}', color='red')
            if with_group:
                ax.text(unit.x, unit.y, f'{unit.group}', color='green')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
    elif dimension == "3D":
        for unit in GTUnits.get_units():
            ax.scatter(unit.x, unit.y, unit.z, marker='o', color='blue')
            if with_id:
                ax.text(unit.x, unit.y, unit.z, f'{unit.id}', color='red')
            if with_group:
                ax.text(unit.x, unit.y, unit.z, f'{unit.group}', color='green')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    else:
        raise ValueError(f"Invalid dimension: {dimension}")

    ax.set_title('GTUnits')
    fig.tight_layout()

    if is_show:
        plt.show()
        return None
    else:
        plt.close()
        return fig
