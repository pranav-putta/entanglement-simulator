from dataclasses import dataclass, fields, astuple
import numpy as np
import pandas as pd


@dataclass
class Cell:
    id: int
    center: np.ndarray
    rotation: np.ndarray
    parent_bud_scar_angle: np.ndarray
    generation: int
    mother_id: int

    @staticmethod
    def root():
        """
        generates a root cell with center at the origin
        :return: Cell object
        """
        return Cell(id=0, center=np.zeros((3,)), rotation=np.identity(3), parent_bud_scar_angle=np.array([1, 0, 0]),
                    generation=0,
                    mother_id=-1)

    def __str__(self):
        return f'Cell: ({self.center[0]}, {self.center[1]}, {self.center[2]}), Gen: {self.generation}'


@dataclass
class Network:
    network: pd.DataFrame

    def __init__(self, root: Cell):
        self.network = pd.DataFrame(columns=[field.name for field in fields(Cell)])
        self.network.loc[0] = list(astuple(root))

    @property
    def ids(self):
        return self['id']

    @property
    def centers(self):
        return self['center']

    @property
    def rotations(self):
        return self['rotation']

    @property
    def bud_scars(self):
        return self['parent_bud_scar_angle']

    @property
    def generations(self):
        return self['generation']

    @property
    def mothers(self):
        return self['mother_id']

    @property
    def mean_bud_scar_angle(self):
        mean_bud_angles = np.zeros(shape=(len(self.network), 2))
        for i in range(len(self.network)):
            # find the bud scars from all the children of this cell
            child_bud_scars = np.array(
                self.network[self.mothers == self.ids[i]]['parent_bud_scar_angle'].values.tolist())
            # mother bud angle is at (-rx, 0, 0) on ellipsoid
            parent_bud_scar = np.array([-np.pi / 2, 0])
            if len(child_bud_scars) != 0:
                # find the average bud scar location on the cell
                child_bud_scars = np.sum(child_bud_scars, axis=0)
                # parent_bud_scar = (child_bud_scars + parent_bud_scar) / (len(child_bud_scars) + 1)
            mean_bud_angles[i] = parent_bud_scar

        return mean_bud_angles

    def __getitem__(self, item):
        return np.array(self.network[item].values.tolist())

    def __iter__(self):
        return self.network.__iter__()

    def __len__(self):
        return len(self.network)

    def add_cells(self, N: int, centers: np.ndarray, bud_angles: np.ndarray, rotations: np.ndarray,
                  mothers: np.ndarray, generation: int):
        """
        add cell to the graph network by simulating budding
         from the mother cell
        :param mother: mother cell
        :param daughter: daughter cell
        :return:
        """
        if N == 0:
            return
        df = pd.DataFrame(
            {'id': np.arange(self.ids.max() + 1, self.ids.max() + N + 1), 'center': list(centers),
             'rotation': list(rotations),
             'parent_bud_scar_angle': list(bud_angles), 'generation': np.repeat(generation, N),
             'mother_id': list(mothers)})
        self.network = self.network.append(df)
