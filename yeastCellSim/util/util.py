import yaml
from dacite import from_dict
import conf
from models.config import Configuration, CellProperties
import numpy as np
import os


class ConfigLoader:

    @staticmethod
    def load_config(path_to_config='../config.yaml'):
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
        return Stats._sample_uniform_dist(0, 100, 3, n) * np.pi / 180.0

    @staticmethod
    def generate_azimuthal_angle(n: int):
        return Stats._sample_uniform_dist(0, 80, 1, n) * np.pi / 180.0


class LinAlg:

    @staticmethod
    def rotation_matrix_3d_vecs(v1: np.ndarray, v2: np.ndarray):
        a, b = (v1 / np.linalg.norm(v1, axis=1).reshape(-1, 1)), (v2 / np.linalg.norm(v2, axis=1).reshape(-1, 1))
        v = np.cross(a, b)
        c = np.sum(a * b, axis=1)
        s = np.linalg.norm(v, axis=1)
        kmat = np.apply_along_axis(lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]]), 1, v)
        rotation_matrix = np.eye(3) + kmat + (kmat @ kmat) * np.reshape((1 - c) / (0.00000001 + s ** 2),
                                                                        (len(v1), 1, 1))
        return rotation_matrix

    @staticmethod
    def smart_collision(ids: np.ndarray, centers: np.ndarray, radii: np.ndarray):
        # make big bounding box
        radius = max(radii)
        mins = np.min(centers, axis=0) - radius
        maxs = np.max(centers, axis=0) + radius
        space_dims = (maxs - mins) // 1000

        spatial_graph = dict()

        def hash_point(pt: np.ndarray, id):
            key = tuple(pt // space_dims)
            if key in spatial_graph:
                spatial_graph[key].add(id)
            else:
                spatial_graph[key] = set(id)

        for i in range(8):
            corners = centers + np.array([radius * (i % 2), radius * (i // 2) % 2, radius * (i // 4) % 2])
            for corner in corners:
                hash_point(corner)

        print(spatial_graph)

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
