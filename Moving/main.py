import numpy as np
import random
import matplotlib.pyplot as plt
from evolutionary_search import EvolSearch


class Optimizer:
    def __init__(self, classes):
        self.Classes = classes
        self.bounds = []
        self.probability_mask = []
        self.bounds_constructor()
        self.probability_mask_constructor()
        gen = self.gen_constructor()
        grid_1 = gen[0]
        p_x, p_y = grid_1[0], grid_1[1]
        offset_x, offset_y = grid_1[2], grid_1[3]
        n_x, n_y = grid_1[4], grid_1[5]
        off2_x, off2_y = grid_1[6], grid_1[7]
        grid_angle = grid_1[8]
        objects_angles = grid_1[10:]
        amount = len(objects_angles)
        previous_grid = self.get_cells_grids([30, 40], [offset_x, offset_y], 50, int(n_x), p_x, off2_x, int(n_y), p_x,
                                             off2_y,
                                             grid_angle)

        grid = self.cut_by_rectagular(previous_grid, [20, 60], [70, 20])
        centre_points = self.sorting_drops_bydistant(grid, [30, 40])[:amount]
        rectangulars = self.get_objects_angle(amount, p_x, p_y, objects_angles)
        print(self.locate_objects(rectangulars, centre_points))

        plt.plot(grid[:, 0], grid[:, 1], 'o')
        plt.show()
        # self.bounds = np.array(self.bounds)
        # self.probability_mask = np.array(self.probability_mask)
        # print(self.probability_mask)
        # gen = self.gen_constructor()
        # koef = self.calibration_function(gen)
        # new_gen = self.gen_constructor()
        # result = self.fitness_function(new_gen, koef)

        # evol_params = {
        #     'num_processes': 4,  # (optional) number of proccesses for multiprocessing.Pool
        #     'pop_size': 200,  # population size
        #     'fitness_function': self.fitness_function,  # custom function defined to evaluate fitness of a solution
        #     'calibration_function': self.calibration_function,
        #     'elitist_fraction': 2,  # fraction of population retained as is between generations
        #     'bounds': self.bounds,
        #     'probability_mask': self.probability_mask,
        #     'num_branches': 3
        # }
        #
        # es = EvolSearch(evol_params)
        #
        # '''OPTION 1
        # # execute the search for 100 generations
        # num_gens = 100
        # es.execute_search(num_gens)
        # '''
        #
        # '''OPTION 2'''
        # # keep searching till a stopping condition is reached
        # num_gen = 0
        # max_num_gens = 102
        # desired_fitness = 0.05
        # es.step_generation()
        # print(es.get_best_individual_fitness())
        # print(es.get_best_individual_fitness() > desired_fitness) # and num_gen > max_num_gens)
        # while es.get_best_individual_fitness() > desired_fitness and num_gen < max_num_gens:
        #     print('Gen #' + str(num_gen) + ' Best Fitness = ' + str(es.get_best_individual_fitness()))
        #     es.step_generation()
        #     num_gen += 1
        #
        # # print results
        # print('Max fitness of population = ', es.get_best_individual_fitness())
        # print('Best individual in population = ', es.get_best_individual())

    def bounds_constructor(self):
        for kind in self.Classes:
            layer_parameters = [[self.Classes[kind]['Environment_x'], self.Classes[kind]['Environment_x'] * 2],
                                [self.Classes[kind]['Environment_y'], self.Classes[kind]['Environment_y'] * 2],
                                [0, self.Classes[kind]['Environment_x']], [0, self.Classes[kind]['Environment_y']],
                                [1, 10], [1, 10],
                                [self.Classes[kind]['Environment_x'] * 2, self.Classes[kind]['Environment_x'] * 10],
                                [self.Classes[kind]['Environment_y'] * 2, self.Classes[kind]['Environment_y'] * 10],
                                [0, 45, 90], [0, 1]]
            for drop in range(0, self.Classes[kind]["Amount"]):
                layer_parameters.append([0, 45, 90])
            self.bounds.append(layer_parameters)

    def probability_mask_constructor(self):
        for kind in self.Classes:
            layer_parameters = [0, 0, 0, 0, 0.5, 0.5,
                                'Equally_distributed', 'Equally_distributed',
                                'Equally_distributed', 'Equally_distributed',
                                [0.4, 0.2, 0.4], 0]
            for drop in range(0, self.Classes[kind]["Amount"]):
                layer_parameters.append([0.4, 0.2, 0.4])
            self.probability_mask.append(layer_parameters)

    def gen_constructor(self):
        big_gen = []
        for net_mask, net_doubt in zip(self.probability_mask, self.bounds):
            net_gen = []
            for probability, value_slot in zip(net_mask, net_doubt):

                if type(probability) is int:
                    net_gen.append(random.triangular(value_slot[0], value_slot[1],
                                                     value_slot[0] + (value_slot[1] - value_slot[0]) * probability))
                elif type(probability) is float:
                    net_gen.append(random.triangular(value_slot[0], value_slot[1],
                                                     value_slot[0] + (value_slot[1] - value_slot[0]) * probability))
                elif type(probability) is str:
                    net_gen.append(random.uniform(value_slot[0], value_slot[1]))
                else:
                    # print(value_slot, probability)
                    net_gen.append(random.choices(value_slot, cum_weights=probability, k=1)[0])

                # elif type(probability) == 'str':
                # else:
            big_gen.append(net_gen)
        # print(big_gen)
        return big_gen

    def test_function(self, x_vector):
        A = np.sum(x_vector[:][0])
        B = np.sum(x_vector[:][1])
        C = np.sum(x_vector[:][2])
        D = np.sum(x_vector[:][3])
        # print([A, B, C, D])
        return np.array([A, B, C, D])

    def calibration_function(self, x_vector):
        x_vector = np.array(x_vector, dtype="object")
        # print(x_vector)
        return self.test_function(x_vector)

    def fitness_function(self, gen, koef):
        x_vector = np.array(gen, dtype="object")
        test_attention = np.array([0.3, 0.3, 0.2, 0.2])
        D = self.test_function(x_vector)
        result = np.sum(test_attention * D / koef)
        mask = []
        for grid in gen:
            mask.append(np.ones_like(np.array(grid)))
        # print(result)
        return [result, np.array(mask, dtype='object')]

    def get_cells_grids(self, main_point, offset_grid, max_rad, n_x, p_x, dist_x, n_y, p_y, dist_y, deg):
        xy = np.mgrid[0:max_rad:n_x * p_x + dist_x, 0:max_rad:n_y * p_y + dist_y].reshape(2, -1).T
        xy_copy = np.copy(xy)
        xy_copy[:, 1] *= -1
        xy = np.append(xy, xy_copy, axis=0)
        xy_copy = np.copy(xy)
        xy_copy[:, 0] *= -1
        xy = np.append(xy, xy_copy, axis=0)
        # print(xy)
        bases_grid = np.unique(xy, axis=0)
        bases_grid_copy = bases_grid.copy()
        for r_x in range(n_x):
            for r_y in range(n_y):
                xy_ = bases_grid_copy + [r_x * p_x, r_y * p_y]
                bases_grid = np.append(bases_grid, xy_, axis=0)
        # print(bases_grid.shape)
        bases_grid = self.ratation(bases_grid, deg)
        bases_grid = bases_grid + main_point
        bases_grid = bases_grid + offset_grid
        print(bases_grid.shape)
        plt.plot(bases_grid[:, 0], bases_grid[:, 1], 'o')
        plt.show()
        return bases_grid

    def cut_by_rectagular(self, grid, up_left_cords, down_right_cords):
        grid = grid[grid[:, 0] > up_left_cords[0]]
        grid = grid[grid[:, 0] < down_right_cords[0]]
        grid = grid[grid[:, 1] < up_left_cords[1]]
        grid = grid[grid[:, 1] > down_right_cords[1]]

        return grid

    def get_objects_angle(self, amount, p_x, p_y, rotation_list):
        rects = np.array([[[-p_x, p_y], [p_x, p_y], [p_x, -p_y], [-p_x, -p_y]]]) / 2
        rects = np.repeat(rects, amount, axis=0)
        rotated_rects = []
        for rect, angle in zip(rects, rotation_list):
            rotated_rects.append(self.ratation(rect, angle))

        return np.array(rotated_rects)

    def sorting_drops_bydistant(self, drops, main_point):
        distant = ((drops[:, 0] - main_point[0]) ** 2 + (drops[:, 1] - main_point[1]) ** 2) ** 0.5
        return drops[np.argsort(distant)]

    def locate_objects(self, rects, centre_points):
        centre_points = centre_points[:, np.newaxis, :]
        centre_points = np.repeat(centre_points, 4, axis=2)
        return rects + centre_points

    def ratation(self, drops, deg):
        theta = np.radians(deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])

        # Define resulting zero array B.
        B = np.zeros(drops.shape)

        # Loop over rows and determine rotated vectors.
        for ra, i in zip(drops, np.arange(drops.shape[0])):
            rb = np.dot(R, ra)
            B[i] = rb

        return B


if __name__ == "__main__":
    Classes = {
        'Workplace': {"Amount": 12, "rectangular_x": 2, "rectangular_y": 1, 'Environment_x': 4, "Environment_y": 3,
                      "Need_lighting": 9,
                      "Classes_for_short_path": ["Printers", "Cabinets"], "Classes_ignored_intersections": ["lamp"],
                      "Classes_for_distant": {"Machine_tool": 8}},
        'Printers': {"Amount": 3, "rectangular_x": 1, "rectangular_y": 1, 'Environment_x': 3, "Environment_y": 3,
                     "Need_lighting": 9,
                     "Classes_for_short_path": ["Workplace"], "Classes_ignored_intersections": ["lamp"],
                     "Classes_for_distant": {"Machine_tool": 9}},
        'Cabinets': {"Amount": 4, "rectangular_x": 0.5, "rectangular_y": 2, 'Environment_x': 1.5, "Environment_y": 1,
                     "Need_lighting": 6,
                     "Classes_for_short_path": ["Workplace"], "Classes_ignored_intersections": ["lamp"],
                     "Classes_for_distant": {"Machine_tool": 5}},
        'lamp': {"Amount": 60, "rectangular_x": 0.4, "rectangular_y": 0.4, 'Environment_x': 0.4, "Environment_y": 0.4,
                 "Need_lighting": "reverse",
                 "Classes_for_short_path": [None],
                 "Classes_ignored_intersections": ["Workplace", "Printers", "Cabinets", "Machine_tool"],
                 "Classes_for_distant": {None}},
        'Machine_tool': {"Amount": 4, "rectangular_x": 3, "rectangular_y": 4, 'Environment_x': 8, "Environment_y": 8,
                         "Need_lighting": 8,
                         "Classes_for_short_path": ["Printers", "Cabinets"], "Classes_ignored_intersections": ["lamp"],
                         "Classes_for_distant": {"Workplace": 8, "Printers": 9, "Cabinets": 5}}
    }

    Opt = Optimizer(Classes)
