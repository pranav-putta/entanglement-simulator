from dataclasses import dataclass
import numpy as np


@dataclass
class Cell:

    def __init__(self, center: np.ndarray, budding_vector: np.ndarray, rotation: np.ndarray, age: float):
        self.center = center
        self.rotation = rotation
        self.budding_vector = budding_vector
        self.age = age

    @staticmethod
    def root():
        """
        generates a root cell with center at the origin
        :return: Cell object
        """
        return Cell(center=np.zeros((3,)), rotation=np.identity(3), budding_vector=np.zeros((3,)), age=0)

    def to_vector(self):
        """
        vectorizes cell object
        :return: a numpy array containing
        """
        return np.array([self.center[0], self.center[1],
                         self.center[2], self.age])

    def __hash__(self):
        return hash(str(self.to_vector()))

    def __str__(self):
        return f'Cell: ({self.center[0]}, {self.center[1]}, {self.center[2]}), Gen: {self.age}'


@dataclass
class Network:
    cell_centers: np.ndarray
    cell_rotations: np.ndarray
    budding_vectors: np.ndarray
    generations: np.ndarray
    edges: dict

    def __init__(self, root: Cell):
        self.cell_centers = np.array([root.center])
        self.cell_rotations = np.array([root.rotation])
        self.budding_vectors = np.array([root.budding_vector])
        self.generations = np.array([root.age])
        self.edges = {tuple(root.center): []}

    def add_cell(self, mother: Cell, daughter: Cell):
        """
        add cell to the graph network by simulating budding
         from the mother cell
        :param mother: mother cell
        :param daughter: daughter cell
        :return:
        """
        self.cell_centers = np.vstack([self.cell_centers, daughter.center])
        self.cell_rotations = np.append(self.cell_rotations, [daughter.rotation], axis=0)
        self.budding_vectors = np.vstack([self.budding_vectors, daughter.budding_vector])
        self.generations = np.append(self.generations, daughter.age)

        self.edges[tuple(daughter.center)] = []
        self.edges[tuple(mother.center)].append(daughter.center)
