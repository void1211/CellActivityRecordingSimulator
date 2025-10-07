import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..GroundTruthUnitsObject import GTUnitObject
from probeinterface import Probe
from probeinterface.plotting import plot_probe

def plot_GTUnit(
    GTUnit: GTUnitObject, 
    ax: plt.Axes = None,
    with_id: bool = False, 
    with_group: bool = False, 
    with_probe: bool = False,
    probe: Probe = None,
    dimension: str = "2D",
    is_show: bool = True,
    ):
    fig = plt.figure(figsize=(6, 6))
    
    if dimension == "2D":
        if ax is None:
            ax = fig.add_subplot(1,1,1)
        for cell in GTUnit.cells:
            ax.scatter(cell.x, cell.y, marker='o', color='blue')
            if with_id:
                ax.text(cell.x, cell.y, f'{cell.id}', color='red')
            if with_group:
                ax.text(cell.x, cell.y, f'{cell.group}', color='green')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
    elif dimension == "3D":
        if ax is None:
            ax = fig.add_subplot(1,1,1, projection='3d')
        for cell in GTUnit.cells:
            ax.scatter(cell.x, cell.y, cell.z, marker='o', color='blue')
            if with_id:
                ax.text(cell.x, cell.y, cell.z, f'{cell.id}', color='red')
            if with_group:
                ax.text(cell.x, cell.y, cell.z, f'{cell.group}', color='green')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    else:
        raise ValueError(f"Invalid dimension: {dimension}")
    
    ax.set_title('GTUnit')

    if is_show:
        plt.show()
        return None
    else:
        plt.close()
        return fig
