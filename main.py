import random

import params
import cells
import grid_data
import neural_networks
import useful_functions


def main():
    global object_grid, pheromone_grid, cell_table, all_cells
    object_grid = grid_data.ObjectGrid()
    pheromone_grid = grid_data.PheromoneGrid()
    cell_table = grid_data.CellTable()
    all_cells = []
    for row in cell_table.data:
        row[1] = ''.join([random.choice('0123456789abcdef') for n in range(8)])

    for i in range(params.GENERATION_COUNT):
        print(i)
        for row in cell_table.data:
            new_cell = cells.SmartCell(row[1], row[0], pheromone_grid.data, object_grid.data)
            all_cells.append(new_cell)
            valid_location = False
            while not valid_location:
                new_location = [random.randint(0, params.GRID_SIZE[0] - 1), random.randint(0, params.GRID_SIZE[1] - 1)]
                if object_grid.data[new_location[0]][new_location[1]] == 0:
                    object_grid.data[new_location[0]][new_location[1]] = row[0]
                    valid_location = True

# assign each genotype a color

        for STEP in range(params.STEPS_PER_GENERATION):
            print('STEP', STEP)
            for cell in all_cells:
                cell.brain.recalculate_values()

# recalculate values for all cells

# check their action potentials to see what they do THIS WILL BE THE HARD PART

# go through their options and attempt their actions

# UPDATE IMAGE

# go to INNER LOOP

# kill off the losers according to some criteria

# reproduce using the ones that are alive

# assign each genotype a color

# go to NEW GENERATION

if __name__ == '__main__':
    main()
