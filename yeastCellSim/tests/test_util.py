import unittest

from simulation import SimEnvironment
from util.util import Stats, LinAlg
from util.util import ConfigLoader
import util.visual
from matplotlib import pyplot as plt
import numpy as np


class SingleOverlapCollisionTest(unittest.TestCase):
    def test_simple_collision(self):
        for i in range(1000):
            ids = (np.array([0]), np.array([1]))
            centers = (np.array([0, 0, 0]), np.array(
                [np.random.uniform(low=-1.5, high=1.5), 0, 0]))
            radii = np.array([1.75, 1, 1])
            rot = np.random.uniform(low=-90, high=90, size=2)
            rot = LinAlg.rotation_matrix_from_spherical_degs(np.array([rot[0]]), np.array([rot[1]]))
            rotations = (np.array([np.eye(3)]), np.array(rot))
            print(f"Testing: {centers[1]}, Rotation: {rot}")
            removal = LinAlg.smart_collision(ids, centers, rotations, radii)
            self.assertGreaterEqual(len(removal), 1)

    def test_no_collisions(self):
        np.random.seed(2)
        for i in range(1000):
            ids = (np.array([0]), np.array([1]))
            centers = (np.array([100, -10, 2]), np.array([3.5, 0, 0]))
            radii = np.array([1.75, 1, 1])

            rot = np.random.uniform(low=-90, high=90, size=2)
            rot = LinAlg.rotation_matrix_from_spherical_degs(np.array([rot[0]]), np.array([rot[1]]))
            rotations = (np.array([np.eye(3)]), np.array(rot))
            print(f"Testing: {centers[1]}, Rotation: {rot}")
            removal = LinAlg.smart_collision(ids, centers, rotations, radii)
            self.assertEqual(len(removal), 0)

    def test_collision1(self):
        # no overlap, aligned along x-axis
        ids = (np.array([0, 1, 2]), np.array([3, 4, 5]))
        centers = (np.array([[0, 0, 0], [3.5, 0, 0]]), np.array([[-3.5, 0, 0], [7, 0, 0]]))
        rotations = (np.array([np.eye(3), np.eye(3)]), np.array([np.eye(3), np.eye(3)]))
        radii = np.array([1.75, 1, 1])
        print(f"Testing: {centers}")
        removal = LinAlg.smart_collision(ids, centers, rotations, radii)
        self.assertEqual(len(removal), 0)

    def test_collision_slight_overlap1(self):
        # slightest overlap, aligned along x-axis
        ids = (np.array([0, 1, 2]), np.array([3, 4, 5]))
        centers = (np.array([[0, 0, 0], [3.5, 0, 0]]), np.array([[-3.5, 0, 0], [6.99, 0, 0]]))
        rotations = (np.array([np.eye(3), np.eye(3)]), np.array([np.eye(3), np.eye(3)]))
        radii = np.array([1.75, 1, 1])
        print(f"Testing: {centers}")
        removal = LinAlg.smart_collision(ids, centers, rotations, radii)
        self.assertEqual(len(removal), 1)

    def test_collision2(self):
        # overlap, complete over id 0 and 4
        ids = (np.array([0, 1, 2]), np.array([3, 4, 5]))
        centers = (np.array([[0, 0, 0], [3.5, 0, 0]]), np.array([[-3.5, 0, 0], [3.5, 0, 0]]))
        rotations = (np.array([np.eye(3), np.eye(3)]), np.array([np.eye(3), np.eye(3)]))
        radii = np.array([1.75, 1, 1])
        print(f"Testing: {centers}")
        removal = LinAlg.smart_collision(ids, centers, rotations, radii)
        self.assertEqual(len(removal), 1)

    def test_collision3(self):
        # slight overlap test
        ids = (np.array([0, 1, 2]), np.array([3, 4, 5]))
        centers = (np.array([[0, 0, 0], [3.5, 0, 0]]), np.array([[-3.5, 0, 0], [4, 0, 0]]))
        rotations = (np.array([np.eye(3), np.eye(3)]), np.array([np.eye(3), np.eye(3)]))
        radii = np.array([1.75, 1, 1])
        print(f"Testing: {centers}")
        removal = LinAlg.smart_collision(ids, centers, rotations, radii)
        self.assertEqual(len(removal), 1)


class MultipleOverlapCollisionTest(unittest.TestCase):
    def test_multiple_collisions1(self):
        # slight overlap test with multiple collisions
        ids = (np.array([0, 1]), np.array([2, 3]))
        centers = (np.array([[0, 0, 0], [3.5, 0, 0]]), np.array([[-3.49, 0, 0], [6.99, 0, 0]]))
        rotations = (np.array([np.eye(3), np.eye(3)]), np.array([np.eye(3), np.eye(3)]))
        radii = np.array([1.75, 1, 1])
        print(f"Testing: {centers[0].tolist()} -> {centers[1].tolist()}")
        removal = LinAlg.smart_collision(ids, centers, rotations, radii)
        print(removal)
        self.assertEqual(len(removal), 2)


class RotationCollisionTest(unittest.TestCase):
    def test_rotation_collision1(self):
        # 1st attempt didn't work : fix floating point precision
        ids = (np.array([0]), np.array([1]))
        centers = (np.array([[0, 0, 0]]), np.array([[2.75, 0, 0]]))
        radii = np.array([1.75, 1, 1])

        rotations = (
            np.array([np.eye(3)]),
            np.array(LinAlg.rotation_matrix_from_spherical_degs(np.array([0.0]), np.array([0.0]))))
        print(f"Testing: {centers[0].tolist()} -> {centers[1].tolist()}")
        removal = LinAlg.smart_collision(ids, centers, rotations, radii)
        self.assertEqual(len(removal), 0)

    def test_rotation_collision2(self):
        ids = (np.array([0]), np.array([1]))
        centers = (np.array([[0, 0, 0]]), np.array([[2.75, 0, 0]]))
        radii = np.array([1.75, 1, 1])

        rotations = (
            np.array([np.eye(3)]),
            np.array(LinAlg.rotation_matrix_from_spherical_degs(np.array([45.0]), np.array([0.0]))))
        print(f"Testing: {centers[0].tolist()} -> {centers[1].tolist()}")
        removal = LinAlg.smart_collision(ids, centers, rotations, radii)
        self.assertEqual(len(removal), 1)

    def test_rotation_azimuthal_collision(self):
        ids = (np.array([0]), np.array([1]))
        centers = (np.array([[0, 0, 0]]), np.array([[2.75, 0, 0]]))
        radii = np.array([1.75, 1, 1])
        rotations = (
            np.array([np.eye(3)]),
            np.array(LinAlg.rotation_matrix_from_spherical_degs(np.array([0.0]), np.array([90.0]))))
        print(f"Testing: {centers[0].tolist()} -> {centers[1].tolist()}")
        removal = LinAlg.smart_collision(ids, centers, rotations, radii)
        self.assertEqual(len(removal), 0)

    def test_brute_force_bounding_box(self):
        radii = np.array([1.75, 1, 1])
        br = LinAlg.brute_force_tight_bounding_box(np.array([np.eye(3)]), radii)
        self.assertTrue(np.allclose(br, radii, atol=0.1))

        rot = LinAlg.rotation_matrix_from_spherical_degs(np.array([0.0]), np.array([0.0]))
        br = LinAlg.brute_force_tight_bounding_box(rot, radii)
        self.assertTrue(np.allclose(br, np.array([1, 1, 1.75]), atol=0.1))

        rot = LinAlg.rotation_matrix_from_spherical_degs(np.array([90.0]), np.array([90.0]))
        br = LinAlg.brute_force_tight_bounding_box(rot, radii)
        self.assertTrue(np.allclose(br, np.array([1, 1.75, 1]), atol=0.1))

    def test_bounding_box(self):
        n = 1000
        thetas = np.random.uniform(low=0, high=90, size=n)
        phis = np.random.uniform(low=0, high=360, size=n)
        radii = np.array([1.75, 1, 1])

        rotations = LinAlg.rotation_matrix_from_spherical_degs(thetas, phis)
        alg_br = LinAlg.tight_bounding_box(rotations, radii)
        for i, rot in enumerate(rotations):
            true_br = LinAlg.brute_force_tight_bounding_box(rot, radii)
            self.assertTrue(np.allclose(alg_br[i], true_br, atol=0.1))
            print(f'Passed {i + 1} / {len(rotations)}')


class UtilTest(unittest.TestCase):

    def test_distribution(self):
        n = 1000
        polar_angle_dist = Stats.generate_polar_angle(n)
        azimuthal_angle_dist = Stats.generate_azimuthal_angle(n)

        fig, axs = plt.subplots(1, 2, squeeze=True)
        axs[0].hist(polar_angle_dist * 180 / 3.14)
        axs[1].hist(azimuthal_angle_dist * 180 / 3.14)

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
