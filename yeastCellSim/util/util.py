import yaml
from dacite import from_dict
import conf
from models.cell import Network
from models.config import Configuration, CellProperties
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

np.random.seed(45998)


class ConfigLoader:

    @staticmethod
    def load_config(path_to_config='./config.yaml'):
        """
        load configuration file from yaml into access types
        :param path_to_config:
        :return:
        """
        with open(path_to_config) as stream:
            data = yaml.safe_load(stream)
            return from_dict(data_class=Configuration, data=data)


class Stats:

    @staticmethod
    def _sample_uniform_dist(a: float, b: float, skew: float, n: int):
        return np.power(np.random.uniform(0, 1, size=n), skew) * (b - a) + a

    @staticmethod
    def generate_polar_angle(n: int):
        return Stats._sample_uniform_dist(0, 120, 3, n) * np.pi / 180.0

    @staticmethod
    def generate_azimuthal_angle(n: int):
        return Stats._sample_uniform_dist(0, 300, 1, n) * np.pi / 180.0


class LinAlg:

    @staticmethod
    def rotation_matrix_from_degrees(x, y, z):
        return R.from_euler('xyz', [x, y, z], degrees=True).as_matrix()

    @staticmethod
    def spherical_to_cartesian(theta, azimuthal, radii, rotation):
        return rotation @ (radii * np.array(
            [np.sin(theta) * np.cos(azimuthal), np.sin(theta) * np.sin(azimuthal), np.cos(theta)])).reshape(-1, 1)

    @staticmethod
    def rotation_matrix_from_spherical(theta, azimuthal):
        start = np.tile(np.array([1, 0, 0]), (theta.shape[0], 1))
        end = np.array(
            [np.sin(theta) * np.cos(azimuthal), np.sin(theta) * np.sin(azimuthal), np.cos(theta)]).transpose()
        return LinAlg.rotation_matrix_3d_vecs(start, end)

    @staticmethod
    def rotation_matrix_from_spherical_degs(theta, azimuthal):
        theta *= np.pi / 180.0
        azimuthal *= np.pi / 180.0
        return LinAlg.rotation_matrix_from_spherical(theta, azimuthal)

    @staticmethod
    def rotation_matrix_3d_vecs(v1: np.ndarray, v2: np.ndarray):
        assert v1.shape == v2.shape
        assert v1.shape[1] == 3
        a, b = (v1 / np.linalg.norm(v1, axis=1).reshape(-1, 1)), (v2 / np.linalg.norm(v2, axis=1).reshape(-1, 1))
        v = np.cross(a, b)
        c = np.sum(a * b, axis=1)
        s = np.linalg.norm(v, axis=1)
        kmat = np.apply_along_axis(lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]]), 1, v)
        rotation_matrix = np.eye(3) + kmat + (kmat @ kmat) * np.reshape((1 - c) / (0.00000001 + s ** 2),
                                                                        (len(v1), 1, 1))
        return rotation_matrix

    @staticmethod
    def brute_force_tight_bounding_box(rotation: np.ndarray, radii: np.ndarray, steps=100):
        theta = np.linspace(0, np.pi, num=100)
        phi = np.linspace(0, 2 * np.pi, num=100)
        mins = np.inf * np.ones(3)

        for t in theta:
            for p in phi:
                coords = LinAlg.spherical_to_cartesian(t, p, radii, rotation).flatten()
                mins[coords < mins] = coords[coords < mins]

        return np.abs(mins)

    @staticmethod
    def tight_bounding_box(rotations: np.ndarray, radii: np.ndarray):
        """
            make tight bounding box using homogenous coordinate transformation -- Notion reference if needed
            E (ellipsoid) = [1/rx^2 0 0 0; 0 1/ry^2 0 0; 0 0 1/rz^2 0; 0 0 0 1]
            M (affine transformation) = [Rx ... Cx ; Ry ... Cy ; Rz ... Cz ; 0 0 0 1]

            any point p on the surface of the conic described by E satisfies:
            p^T * E * p = 0
            (M^-1 * p)^T * E * (M^-1 * p) = 0
            p^T*((M^-1)^T * E  * M^-1) * p = 0
            Q (transformed ellipsoid) = (M^-1)^T * E  * M^-1

            the tangent plane to a point p on Q can be described as:
            u = p^T * Q

            p^T * Q * p = 0
            p^T * Q * Q^-1 * Q * p = 0
            u^T * Q^-1 * u = 0

        """
        E = np.array(np.diag([*(1 / (radii ** 2)), -1]))
        M = np.pad(rotations, ((0, 0), (0, 1), (0, 1)))
        M[:, 3, 3] = 1
        MI = np.linalg.inv(M)
        Q = np.transpose(MI, axes=(0, 2, 1)) @ E @ MI
        R = np.linalg.inv(Q)
        R_44 = R[:, 3, 3].reshape(-1, 1)

        bounding_radius = np.abs((R[:, 0:3, 3] + np.sqrt(
            R[:, 0:3, 3] ** 2 - np.diagonal(R, axis1=1, axis2=2)[:, 0:3] * R_44)) / R_44)

        return bounding_radius

    @staticmethod
    def smart_collision(ids: tuple, centers: tuple, rotations: tuple, radii: np.ndarray, bound_threshold=0.3):
        parent_ids, child_ids = ids
        parent_centers, child_centers = centers
        parent_rots, child_rots = rotations

        # currently assumes that each parent has a child to check
        assert len(parent_ids) == len(child_ids), "parent and child ID lengths were not the same"
        assert len(parent_centers) == len(child_centers), "parent and child Center lengths were not the same"
        assert len(parent_rots) == len(child_rots), "parent and child Rotation lengths were not the same"

        ids = np.vstack([parent_ids, child_ids]).reshape(-1)
        centers = np.vstack([parent_centers, child_centers])
        rotations = np.vstack([parent_rots, child_rots])

        space_dims = np.array([5, 5, 5])
        bounding_radius = LinAlg.tight_bounding_box(rotations, radii)
        spatial_graph = dict()

        def hash_point(pt: np.ndarray, id):
            key = tuple((pt // space_dims).tolist())
            if key in spatial_graph:
                spatial_graph[key].add(id)
            else:
                spatial_graph[key] = set([id])

        for i in range(8):
            corners = centers + bounding_radius * np.array(
                [2 * (i % 2) - 1, 2 * ((i // 2) % 2) - 1, 2 * ((i // 4) % 2) - 1])
            for j in range(len(corners)):
                hash_point(corners[j], j)

        removed_ids = set()
        # check collisions
        for block, collision_ids in spatial_graph.items():
            collision_ids = list(set(collision_ids) - removed_ids)
            n = len(collision_ids)
            collisions = np.zeros(shape=(n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    # index of object from master list
                    ci, cj = collision_ids[i], collision_ids[j]
                    # check collision on each axis
                    c1, c2 = centers[ci], centers[cj]
                    br1, br2 = bounding_radius[ci], bounding_radius[cj]
                    s1, s2 = c1 - br1, c2 - br2
                    e1, e2 = c1 + br1, c2 + br2
                    starts, ends = np.vstack([s1, s2]), np.vstack([e1, e2])
                    if np.all(np.max(starts, axis=0) < np.min(ends, axis=0) - bound_threshold):
                        # collision detected
                        collisions[i][j] = 1
                        collisions[j][i] = 1
            while np.sum(collisions) > 0:
                remove = np.argmax(np.sum(collisions, axis=0))

                if ids[collision_ids[remove]] not in parent_ids:
                    collisions[remove, :] = np.zeros(n)
                    collisions[:, remove] = np.zeros(n)

                    removed_ids.add(ids[collision_ids[remove]])
                else:
                    child_removals = np.argwhere(collisions[remove] == 1)
                    for child in child_removals:
                        collisions[int(child), :] = np.zeros(n)
                        collisions[:, int(child)] = np.zeros(n)
                        # check if child of parent, ignore
                        removed_ids.add(ids[collision_ids[int(child)]])
        print(f'{len(removed_ids)} Collisions Found for {len(ids)} nodes.')
        return removed_ids

    @staticmethod
    def check_overlaps(centers: np.ndarray, rotations: np.ndarray, radii: np.ndarray, old_cells_size: int):
        radii = np.array(radii)
        start = np.tile(radii, (centers.shape[0], 1))
        start = -np.abs((rotations @ start.reshape(*start.shape, 1)).reshape(start.shape)) + centers
        start = start.T
        end = np.tile(radii, (centers.shape[0], 1))
        end = np.abs((rotations @ end.reshape(*end.shape, 1)).reshape(end.shape)) + centers
        end = end.T

        overlapping_points = np.arange(centers.shape[0])

        for i in range(3):
            axis = start[i]
            argsorted = np.argsort(axis)
            axis_end = end[i, argsorted]
            axis_start = np.sort(axis)
            diff = np.around(axis_start[1:] - axis_end[:-1], 3)
            overlapping_intervals = argsorted[np.union1d(np.argwhere(diff < 0), np.argwhere(diff < 0) + 1)]
            overlapping_points = np.intersect1d(overlapping_intervals, overlapping_points)

        overlapping_adj_matrix = np.zeros(shape=(len(overlapping_points), len(overlapping_points)))
        for idx in range(len(overlapping_points)):
            i = overlapping_points[idx]
            overlapping_args = np.arange(len(overlapping_points))
            for j in range(3):
                a = (np.intersect1d(np.argwhere(start[j, i] <= end[j, overlapping_points]).reshape(-1),
                                    np.argwhere(start[j, i] >= start[j, overlapping_points]).reshape(-1)))
                b = (np.intersect1d(np.argwhere(end[j, i] <= end[j, overlapping_points]).reshape(-1),
                                    np.argwhere(end[j, i] >= start[j, overlapping_points]).reshape(-1)))
                overlapping_args = np.intersect1d(overlapping_args, np.union1d(a, b))
            for j in overlapping_args:
                overlapping_adj_matrix[j][idx] = 1
                overlapping_adj_matrix[idx][j] = 1

        overlapping_adj_matrix -= np.eye(len(overlapping_adj_matrix))
        remove_args = []

        # remove anything gthat intersects with old cells
        for i in np.argwhere(overlapping_points < old_cells_size).reshape(-1):
            overlaps = np.argwhere(overlapping_adj_matrix[i] == 1).reshape(-1)
            for point in overlaps:
                remove_args.append(point)
                overlapping_adj_matrix[point] = np.zeros(len(overlapping_adj_matrix))
                overlapping_adj_matrix[:, point] = np.zeros(len(overlapping_adj_matrix))

        while np.sum(overlapping_adj_matrix) > 0:
            row = np.argmax(np.sum(overlapping_adj_matrix, axis=1))
            remove_args.append(row)
            overlapping_adj_matrix[row] = np.zeros(len(overlapping_adj_matrix))
            overlapping_adj_matrix[:, row] = np.zeros(len(overlapping_adj_matrix))

        print(f'{centers.shape[0]}. Overlapping #s: {len(remove_args)}')
        return overlapping_points[remove_args] - old_cells_size
