import unittest

from simulation import SimEnvironment
from util.util import ConfigLoader


class SimulationTest(unittest.TestCase):
    def test_run_until_size(self):
        config = ConfigLoader.load_config()
        simulation = SimEnvironment(config)
        simulation.run_until_size(5000)
        print(f'Number of cells: {len(simulation.network.cell_centers)}')
        print(f'Number of generations: {simulation.generation}')

    def test_simulations(self):
        config = ConfigLoader.load_config()
        simulation = SimEnvironment(config)
        simulation.run_generations()

        print(f'Number of cells: {len(simulation.network.cell_centers)}')


if __name__ == '__main__':
    unittest.main()
