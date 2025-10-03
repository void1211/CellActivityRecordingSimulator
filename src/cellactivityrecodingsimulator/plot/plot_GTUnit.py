import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..GroundTruthUnitObject import GTUnitObject

def plot_GTUnit(
    GTUnit: GTUnitObject, 
    with_id: bool = False, 
    with_group: bool = False, 
    dimension: str = "2D",
    ):
    fig = plt.figure(figsize=(12, 8))
    
    if dimension == "2D":
        ax = fig.add_subplot(111)
        for cell in GTUnit.cells:
            ax.scatter(cell.x, cell.y, marker='o', color='blue')
            if with_id:
                ax.text(cell.x, cell.y, f'{cell.id}', color='red')
            if with_group:
                ax.text(cell.x, cell.y, f'{cell.group}', color='green')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
    elif dimension == "3D":
        ax = fig.add_subplot(111, projection='3d')
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
    return fig