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
        self.network = Network(root=Cell.root(), config=self.config)

    def reset(self):
        self.generation = 0
        self.network = Network(root=Cell.root(), config=self.config)

    def run_until_size(self, size: int):
        while len(self.network.centers) < size:
            self.step()
            self.generation += 1

    def run_generations(self):
        for generation in range(self.config.generations):
            self.step()
            self.generation += 1

    def step(self):
        """
        steps through the next generation
        :return:
        """
        N = len(self.network)
        old_centers = self.network.centers
        old_rotations = self.network.rotations
        old_bud_scars = self.network.pattern_bud_scar_angles

        # generate polar and azimuthal angles
        delta_angle = np.array([Stats.generate_polar_angle(N), Stats.generate_azimuthal_angle(N)]).transpose()

        # generate new bud angles: point in opposite direction of bud scars and generate a new bud there
        new_bud_angles = -old_bud_scars + delta_angle
        rx, ry, rz = self.config.cell_properties.radius
        x = rx * np.sin(new_bud_angles[:, 0]) * np.cos(new_bud_angles[:, 1])
        y = ry * np.sin(new_bud_angles[:, 0]) * np.sin(new_bud_angles[:, 1])
        z = rz * np.cos(new_bud_angles[:, 0])

        surface_points = np.array([x, y, z]).transpose()
        gradient = 2 * surface_points
        # normalize gradient
        gradient = gradient / np.linalg.norm(gradient, axis=1).reshape(-1, 1)
        gradient = (old_rotations @ np.expand_dims(gradient, axis=2)).reshape(gradient.shape)

        D = np.linalg.norm(surface_points, axis=1)
        new_centers = old_centers + (D + rx).reshape(-1, 1) * gradient

        # calculate rotation matrices
        v1 = np.tile(np.array([rx, 0, 0]), (N, 1))
        v2 = gradient
        new_rotations = LinAlg.rotation_matrix_3d_vecs(v1, v2)

        max_id = max(self.network.ids)
        child_ids = np.arange(max_id + 1, max_id + N + 1)

        # calculate collisions
        ids = (self.network.ids, child_ids)
        centers = (old_centers, new_centers)
        rotations = (old_rotations, new_rotations)

        remove_idxs = LinAlg.smart_collision(ids, centers, rotations, np.array([rx, ry, rz]),
                                             bound_volume=self.config.simulation.bound_volume,
                                             prune=self.config.simulation.prune_collisions,
                                             verbose=self.config.verbose)

        # remove collision events
        keep_slice = np.delete(np.arange(N), np.searchsorted(child_ids, np.array(list(remove_idxs))))

        new_centers = new_centers[keep_slice]
        new_rotations = new_rotations[keep_slice]
        new_bud_angles = new_bud_angles[keep_slice]
        mother_ids = self.network.ids[keep_slice]

        self.network.add_cells(len(keep_slice), new_centers, new_bud_angles, new_rotations, mother_ids,
                               self.generation + 1)
        if self.config.verbose:
            print(f'generation {self.generation}. # of nodes: {len(self.network)}')
