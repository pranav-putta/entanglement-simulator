from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from models.cell import Cell
from models.config import Configuration
import matplotlib as mpl

mpl.use('macosx')


def draw_cell(fig, ax, cells: np.ndarray, rotations: np.ndarray, buds: np.ndarray, gens: np.ndarray,
              config: Configuration):
    coefs = (1, 1, 1)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1
    # Radii corresponding to the coefficients:
    rx, ry, rz = config.cell_properties.radius

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    colors = ['blue', 'green', 'red', 'pink', 'yellow',
              'cyan', 'salmon', 'tomato', 'slategrey', 'orange']

    mxx, mxy, mxz = 0, 0, 0
    mix, miy, miz = 0, 0, 0
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

        arr = (rotations[i] @ np.array([x, y, z])).reshape(3, 100, 100)
        x, y, z = arr[0], arr[1], arr[2]
        x += cell[0]
        y += cell[1]
        z += cell[2]

        mxx = max(mxx, np.max(np.abs(x)))
        mxy = max(mxy, np.max(np.abs(y)))
        mxz = max(mxz, np.max(np.abs(z)))

        mix = np.min(x)
        miy = np.min(y)
        miz = np.min(z)
        # Plot:
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=colors[gens[i] % len(colors)])
        # ax.quiver(budsite[0], budsite[1], budsite[2], cell[0], cell[1], cell[2], color=colors[gens[i] % len(colors)])

    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(mxx, mxy, mxz)
    print(max_radius)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    print(f'Number of cells {len(cells)}')
