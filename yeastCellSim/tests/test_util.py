import unittest

from simulation import SimEnvironment
from util.util import Stats
from util.util import ConfigLoader
import util.visual
from matplotlib import pyplot as plt


class UtilTest(unittest.TestCase):
    def test_distribution(self):
        n = 1000
        polar_angle_dist = Stats.generate_polar_angle(n)
        azimuthal_angle_dist = Stats.generate_azimuthal_angle(n)

        fig, axs = plt.subplots(1, 2, squeeze=True)
        axs[0].hist(polar_angle_dist)
        axs[1].hist(azimuthal_angle_dist)

        plt.show()


class ConfigTest(unittest.TestCase):
    def test_load_config(self):
        config = ConfigLoader.load_config()
        print(config.cell_properties.radius)


class VisualTest(unittest.TestCase):

    def test_visual(self):
        config = ConfigLoader.load_config()
        simulation = SimEnvironment(config)
        simulation.run_until_size(500)

        fig = plt.figure(figsize=(10, 10))  # Square figure
        ax = fig.add_subplot(111, projection='3d')
        util.visual.draw_cell(fig, ax, simulation.network.cell_centers, simulation.network.cell_rotations,
                              simulation.network.budding_vectors, simulation.network.generations,
                              config)
        plt.show()


if __name__ == '__main__':
    unittest.main()
