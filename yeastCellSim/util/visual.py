from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from models.cell import Cell
from models.config import Configuration
import matplotlib as mpl

mpl.use('macosx')

resolution = 20


def draw_cell(fig, ax, cells: np.ndarray, rotations: np.ndarray, buds: np.ndarray, gens: np.ndarray,
              config: Configuration):
    coefs = (1, 1, 1)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1
    # Radii corresponding to the coefficients:
    rx, ry, rz = config.cell_properties.radius

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    colors = ['blue', 'green', 'red', 'pink', 'yellow',
              'cyan', 'salmon', 'tomato', 'slategrey', 'orange']

    for i in range(0, len(cells)):
        cell = cells[i]
        budsite = buds[i]
        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        x = rx * np.outer(np.cos(u), np.sin(v))
        x = x.reshape(-1)
        y = ry * np.outer(np.sin(u), np.sin(v))
        y = y.reshape(-1)
        z = rz * np.outer(np.ones_like(u), np.cos(v))
        z = z.reshape(-1)

        arr = (rotations[i] @ np.array([x, y, z])).reshape(3, resolution, resolution)
        x, y, z = arr[0], arr[1], arr[2]
        x += cell[0]
        y += cell[1]
        z += cell[2]

        # Plot:
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=colors[gens[i] % len(colors)])
        # ax.quiver(budsite[0], budsite[1], budsite[2], cell[0], cell[1], cell[2], color=colors[gens[i] % len(colors)])

    # Adjustment of the axes, so that they all have the same span:
    max_radius = cells.max(axis=0) + 3
    min_radius = cells.min(axis=0) - 3
    getattr(ax, 'set_{}lim'.format('x'))((min_radius[0], max_radius[0]))
    getattr(ax, 'set_{}lim'.format('y'))((min_radius[1], max_radius[1]))
    getattr(ax, 'set_{}lim'.format('z'))((min_radius[2], max_radius[2]))

    print(f'Number of cells {len(cells)}')
