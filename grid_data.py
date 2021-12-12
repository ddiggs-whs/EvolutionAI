import numpy as np
import params


class ObjectGrid:
    def __init__(self):
        self.data = np.zeros(params.GRID_SIZE, dtype=np.int)


class PheromoneGrid:
    def __init__(self):
        self.data = np.zeros(params.GRID_SIZE, dtype=float)


class CellTable:
    def __init__(self):
        self.data = [[0 for _ in range(32 + params.INNER_NEURON_COUNT)] for _ in range(params.POPULATION)]
        for i, row in enumerate(self.data):
            row[0] = i + 1
