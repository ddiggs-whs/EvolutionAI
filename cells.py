import math
from math import sin
from random import random
import params
from difflib import SequenceMatcher
from neural_networks import Brain


class SmartCell:
    def __init__(self, DNA, id_num, pheromone_grid, object_grid):
        self.location = [0, 0]  # integer location showing what grid position it occupies
        self.genome = DNA  # string defining the genome of the cell
        self.error_rate = 1 / 1000  # chance of a random mutation occurring when passing genome to offspring

        self.id = id_num
        self.age = 0
        self.last_move_direction = [None, None]
        self.movement = [None, None]  # movement in X, movement in Y
        self.oscillator_period = 30
        self.pheromone_grid = pheromone_grid
        self.object_grid = object_grid

        self.long_range_distance = 8
        self.short_range_distance = 1
        self.brain = Brain(self.genome, self)

    @property
    def Slr(self):
        """pheromone gradient left-right"""
        if self.last_move_direction == [None, None]:
            return 0
        else:
            try:
                left = self.pheromone_grid[self.left_square()[0]][self.left_square()[1]]
            except IndexError:
                left = 0
            try:
                right = self.pheromone_grid[self.right_square()[0]][self.right_square()[1]]
            except IndexError:
                right = 0
            return (left - right) / 2 + 0.5

    @property
    def Sfd(self):
        """pheromone gradient forward"""
        if self.last_move_direction == [None, None]:
            return 0
        else:
            try:
                forward_pheromone = self.pheromone_grid[self.location[0] + self.last_move_direction[0]][
                    self.pheromone_grid[1] + self.last_move_direction[1]]
            except IndexError:
                forward_pheromone = 0
            try:
                backward_pheromone = self.pheromone_grid[self.location[0] - self.last_move_direction[0]][
                    self.pheromone_grid[1] - self.last_move_direction[1]]
            except IndexError:
                backward_pheromone = 0
            return (forward_pheromone - backward_pheromone) / 2 + 0.5

    @property
    def Sg(self):
        """pheromone density"""
        total = 0
        count = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                try:
                    total += self.pheromone_grid[self.location[0] + i][self.location[1] + j]
                    count += 1
                except IndexError:
                    pass

        return total / count

    @property
    def Age(self):
        """age"""
        return self.age / params.STEPS_PER_GENERATION

    @property
    def Rnd(self):
        """random input"""
        return random()

    @property
    def Blr(self):
        """blockage left-right"""
        if self.last_move_direction == [None, None]:
            return 0
        else:
            try:
                left_space = self.object_grid[self.left_square()[0]][self.left_square()[1]]
            except IndexError:
                left_space = -1
            try:
                right_space = self.object_grid[self.right_square()[0]][self.right_square()[1]]
            except IndexError:
                right_space = -1
            if left_space != 0:
                left = 1
            else:
                left = 0
            if right_space != 0:
                right = 1
            else:
                right = 0
            return (left - right) / 2 + 0.5

    @property
    def Osc(self):
        """oscillator"""
        return sin(self.Age * 2 * math.pi / self.oscillator_period)

    @property
    def Bfd(self):
        """blockage forward"""
        if self.last_move_direction == [None, None]:
            return 0
        else:
            try:
                neighbor = self.object_grid[self.location[0] + self.last_move_direction[0]][
                    self.location[1] + self.last_move_direction[1]]
            except IndexError:
                neighbor = -1
            if neighbor != 0:
                return 1
            else:
                return 0

    @property
    def Plr(self):
        """population gradient left-right"""
        if self.last_move_direction == [None, None]:
            return 0
        else:
            try:
                left_space = self.object_grid[self.left_square()[0]][self.left_square()[1]]
            except IndexError:
                left_space = -1
            try:
                right_space = self.object_grid[self.right_square()[0]][self.right_square()[1]]
            except IndexError:
                right_space = -1
            if left_space not in [-1, 0]:
                left = 1
            else:
                left = 0
            if right_space not in [-1, 0]:
                right = 1
            else:
                right = 0
            return (left - right) / 2 + 0.5

    @property
    def Pop(self):
        """population density"""
        total = 0
        count = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                try:
                    if self.object_grid[self.location[0] + i][self.location[1] + j] not in [0, -1]:
                        total += 1
                    count += 1
                except IndexError:
                    pass
        return total / count

    @property
    def Pfd(self):
        """population gradient forward"""
        if self.last_move_direction == [None, None]:
            return 0
        else:
            try:
                forward_space = self.object_grid[self.location[0] + self.last_move_direction[0]][
                    self.object_grid[1] + self.last_move_direction[1]]
                backward_space = self.object_grid[self.location[0] - self.last_move_direction[0]][
                    self.object_grid[1] - self.last_move_direction[1]]
                if forward_space not in [0, -1]:
                    forward = 1
                else:
                    forward = 0
                if backward_space not in [0, -1]:
                    backward = 1
                else:
                    backward = 0
                return (forward - backward) / 2 + 0.5
            except IndexError:
                return 0

    @property
    def LPf(self):
        """population long-range forward"""
        if self.last_move_direction == [None, None]:
            return 0
        else:
            total = 0
            count = 0
            for i in range(1, self.long_range_distance + 1):
                try:
                    neighbor = self.object_grid[self.location[0] + self.last_move_direction[0] * i][
                        self.location[1] + self.last_move_direction[1] * i]
                    if neighbor not in [0, -1]:
                        total += 1
                    count += 1
                except IndexError:
                    pass
            return total / count

    @property
    def LMy(self):
        """last movement Y"""
        if self.last_move_direction == [None, None]:
            return 0
        else:
            return self.last_move_direction[1] + 1

    @property
    def LBf(self):
        """blockage long-range forward"""
        if self.last_move_direction == [None, None]:
            return 0
        else:
            total = 0
            count = 0
            for i in range(1, self.long_range_distance + 1):
                try:
                    neighbor = self.object_grid[self.location[0] + self.last_move_direction[0] * i][
                        self.location[1] + self.last_move_direction[1] * i]
                    count += 1
                except IndexError:
                    continue
                if neighbor != 0:
                    total += 1
            return total / count

    @property
    def LMx(self):
        """last movement X"""
        if self.last_move_direction == [None, None]:
            return 0
        else:
            return self.last_move_direction[0] + 1

    @property
    def BDy(self):
        """north/south border distance"""
        return min([self.location[1], params.GRID_SIZE[1] - self.location[1]])

    @property
    def Gen(self):
        """genetic similarity of fwd neighbor"""
        if self.last_move_direction == [None, None]:
            return 0
        else:
            neighbor = self.object_grid[self.location[0] + self.last_move_direction[0]][
                self.location[1] + self.last_move_direction[1]]
            if neighbor not in [-1, 0]:
                return SequenceMatcher(self.genome, neighbor.genome)
            else:
                return 0

    @property
    def BDx(self):
        """east/west border distance"""
        return min([self.location[0], params.GRID_SIZE[0] - self.location[0]])

    @property
    def Lx(self):
        """east/west world location"""
        return self.location[0] / params.GRID_SIZE[0]

    @property
    def BD(self):
        """nearest border distance"""
        return min([self.location[0], params.GRID_SIZE[0] - self.location[0], self.location[1],
                    params.GRID_SIZE[1] - self.location[1]])

    @property
    def Ly(self):
        """north/south world location"""
        return self.location[1] / params.GRID_SIZE[1]

    def left_square(self):
        if self.last_move_direction == [None, None]:
            return None
        else:
            if self.last_move_direction == [1, -1]:
                return [self.location[0] - 1, self.location[1] - 1]
            elif self.last_move_direction == [1, -0]:
                return [self.location[0], self.location[1] - 1]
            elif self.last_move_direction == [1, 1]:
                return [self.location[0] + 1, self.location[1] - 1]
            elif self.last_move_direction == [-1, 0]:
                return [self.location[0], self.location[1] + 1]
            elif self.last_move_direction == [-1, 1]:
                return [self.location[0] + 1, self.location[1] + 1]
            elif self.last_move_direction == [-1, -1]:
                return [self.location[0] - 1, self.location[1] + 1]
            elif self.last_move_direction == [0, 1]:
                return [self.location[0] + 1, self.location[1]]
            elif self.last_move_direction == [0, -1]:
                return [self.location[0] - 1, self.location[1]]

    def right_square(self):
        if self.last_move_direction == [None, None]:
            return None
        else:
            if self.last_move_direction == [1, -1]:
                return [self.location[0] + 1, self.location[1] + 1]
            elif self.last_move_direction == [1, -0]:
                return [self.location[0], self.location[1] + 1]
            elif self.last_move_direction == [1, 1]:
                return [self.location[0] - 1, self.location[1] + 1]
            elif self.last_move_direction == [-1, 0]:
                return [self.location[0], self.location[1] - 1]
            elif self.last_move_direction == [-1, 1]:
                return [self.location[0] - 1, self.location[1] - 1]
            elif self.last_move_direction == [-1, -1]:
                return [self.location[0] + 1, self.location[1] - 1]
            elif self.last_move_direction == [0, 1]:
                return [self.location[0] - 1, self.location[1]]
            elif self.last_move_direction == [0, -1]:
                return [self.location[0] + 1, self.location[1]]
