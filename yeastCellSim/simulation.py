from dataclasses import dataclass
import numpy as np
from models.config import Configuration
from models.cell import Network, Cell
from util.util import Stats, LinAlg


@dataclass
class SimEnvironment:
    """
    In order to vectorize inputs to speed up for numpy,
     we will refrain to use object-oriented design patterns.
     This removes any python wrappings around objects, and
     should speed up simulation times.
    """

    def __init__(self, config: Configuration):
        self.config = config
        self.generation = 0
        self.network = Network(root=Cell.root())

    def run_until_size(self, size: int):
        while len(self.network.cell_centers) < size:
            self.step()
            self.generation += 1

    def run_generations(self):
        for generation in range(self.config.generations):
            self.step()
            self.generation += 1

    def step(self):
        old_centers = self.network.cell_centers
        old_rotations = self.network.cell_rotations
        old_bud_vecs = self.network.budding_vectors

        polar_angles = Stats.generate_polar_angle(len(old_centers)) * np.pi / 180
        azimuthal_angles = Stats.generate_azimuthal_angle(len(old_centers)) * np.pi / 180

        rx, ry, rz = self.config.cell_properties.radius
        x = rx * np.sin(polar_angles) * np.cos(azimuthal_angles)
        y = ry * np.sin(polar_angles) * np.sin(azimuthal_angles)
        z = rz * np.cos(polar_angles)

        surface_points = np.array([x, y, z]).transpose()
        gradient = 2 * surface_points
        # normalize gradient
        gradient = gradient / np.linalg.norm(gradient, axis=1).reshape(-1, 1)
        gradient = (old_rotations @ gradient.reshape(*gradient.shape, 1)).reshape(gradient.shape[0],
                                                                                  gradient.shape[1])

        D = np.linalg.norm(surface_points, axis=1)
        new_centers = old_centers + (D + rx).reshape(-1, 1) * gradient

        # calculate rotation matrices
        v1 = np.tile(np.array([rx, 0, 0]), (len(new_centers), 1))
        v2 = gradient
        rotations = LinAlg.rotation_matrix_3d_vecs(v1, v2)

        idxs_to_remove = LinAlg.check_overlaps(np.vstack([old_centers, new_centers]),
                                               np.vstack([old_rotations, rotations]),
                                               self.config.cell_properties.radius, len(old_centers))
        keep_slice = np.delete(np.arange(len(new_centers)), idxs_to_remove)
        new_centers = new_centers[keep_slice]
        surface_points = surface_points[keep_slice]
        rotations = rotations[keep_slice]
        old_centers = old_centers[keep_slice]
        old_bud_vecs = old_bud_vecs[keep_slice]
        old_rotations = old_rotations[keep_slice]

        # add new centers to graph
        for i in range(len(old_centers)):
            mother = Cell(old_centers[i], old_bud_vecs[i], old_rotations[i], self.generation)
            daughter = Cell(new_centers[i], surface_points[i].transpose(), rotations[i], self.generation + 1)
            self.network.add_cell(mother, daughter)
