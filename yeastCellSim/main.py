from simulation import SimEnvironment
import util.visual
from util.util import ConfigLoader
from matplotlib import pyplot as plt
import numpy as np


config = ConfigLoader.load_config()
simulation = SimEnvironment(config)
simulation.run_until_size(75)
print(f'Number of cells: {len(simulation.network.centers)}')
print(f'Number of generations: {simulation.generation}')

fig = plt.figure(figsize=(10, 10))  # Square figure
ax = fig.add_subplot(111, projection='3d')
util.visual.draw_cell(fig, ax, simulation.network.centers, simulation.network.rotations,
                      simulation.network.bud_scars, simulation.network.generations,
                      config)
plt.show()
