import unittest

from simulation import SimEnvironment
from util.util import ConfigLoader
import util.visual
from matplotlib import pyplot as plt
import numpy as np


class SimulationTest(unittest.TestCase):
    def test_run_configurations_until_size(self):
        size = 50
        config = ConfigLoader.load_config()

        bound_volumes = np.arange(0, 1, 0.05)
        config.simulation.prune_collisions = True
        config.simulation.bud_angle_pattern = 'None'
        config.simulation.remove_children = True
        config.verbose = False

        for i in range(len(bound_volumes)):
            config.simulation.bound_volume = bound_volumes[i]
            simulation = SimEnvironment(config)
            simulation.run_until_size(size)
            print(
                f'generating {len(simulation.network)} cells took {simulation.generation} generations. '
                f'configuration: {config.simulation}')

    def test_run_until_size(self):
        config = ConfigLoader.load_config()
        simulation = SimEnvironment(config)
        simulation.run_until_size(500)
        print(f'Number of cells: {len(simulation.network.centers)}')
        print(f'Number of generations: {simulation.generation}')

        fig = plt.figure(figsize=(10, 10))  # Square figure
        ax = fig.add_subplot(111, projection='3d')
        util.visual.draw_cell(fig, ax, simulation.network.centers, simulation.network.rotations,
                              simulation.network.bud_scars, simulation.network.generations,
                              config)
        plt.show()

        print(simulation.network.network)

    def test_simulations(self):
        config = ConfigLoader.load_config()
        simulation = SimEnvironment(config)
        simulation.run_generations()

        print(f'Number of cells: {len(simulation.network.cell_centers)}')


if __name__ == '__main__':
    unittest.main()
