from simulation import SimEnvironment
import util.visual
from util.util import ConfigLoader
from matplotlib import pyplot as plt

config = ConfigLoader.load_config()
simulation = SimEnvironment(config)
simulation.run_generations()

fig = plt.figure(figsize=(10, 10))  # Square figure
ax = fig.add_subplot(111, projection='3d')
util.visual.draw_cell(fig, ax, simulation.network.cell_centers, simulation.network.cell_rotations, config)
plt.show()
