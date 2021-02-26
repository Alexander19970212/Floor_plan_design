import numpy as np
import random
import matplotlib.pyplot as plt
from evolutionary_search import EvolSearch



#  debug distance between classes
#  check sum by axis
#  don't forget to get mask for mutation


class Optimizer:
    def __init__(self, classes):
        self.Classes = classes
        self.bounds = []
        self.probability_mask = []
        self.bounds_constructor()
        self.probability_mask_constructor()
        gen = self.gen_constructor()
        windows = np.array([[[5, 45], [65, 5]], [[5, 75], [65, 35]], [[70, 75], [95, 5]], [[100, 40], [165, 5]],
                            [[100, 75], [165, 45]]])
        main_points = np.array([[5, 5], [45, 55], [95, 35], [165, 40], [135, 45]])

        obj_classes = self.builder(gen, windows, main_points, 50)
        mat_dist = self.get_minimal_dist_mat()
        self.distant_between_classes(obj_classes, mat_dist)
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

    def get_minimal_dist_mat(self):
        minimal_distances = []
        for class_1 in self.Classes:
            min_dist_one_axies = []
            for class_2 in self.Classes:
                if class_1 != class_2:
                    try:
                        dist = self.Classes[class_1]["Classes_for_distant"][class_2]
                    except:
                        dist = 0
                else:
                    dist = 0
                min_dist_one_axies.append(dist)
            minimal_distances.append( min_dist_one_axies)

        return np.array(minimal_distances)

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
        return np.array(big_gen, dtype='object')

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
        if deg != 0:
            bases_grid = self.ratation(bases_grid, deg)
        bases_grid = bases_grid + main_point
        bases_grid = bases_grid + offset_grid
        print(bases_grid.shape)
        # plt.plot(bases_grid[:, 0], bases_grid[:, 1], 'o')
        # plt.show()
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
        drops = np.unique(drops, axis=0)
        distant = ((drops[:, 0] - main_point[0]) ** 2 + (drops[:, 1] - main_point[1]) ** 2) ** 0.5
        return drops[np.argsort(distant)]

    def get_centre_pints_opinion(self, drops, n, index):
        n_drops = drops.shape[0]
        offset = int(index * (n_drops - n))
        return drops[offset:offset + n, :]

    def locate_objects(self, rects, centre_points):
        centre_points = centre_points[:, np.newaxis, :]
        centre_points = np.repeat(centre_points, 4, axis=1)
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

    def builder(self, gen, windows, main_points, dioganal):
        colors = ['red', 'blue', 'yellow', 'black', 'green']
        colors = colors[:gen.shape[0]]

        object_class = []
        for chromosome, window, main_point, color in zip(gen, windows, main_points, colors):
            grid = chromosome[0]
            p_x, p_y = chromosome[0], chromosome[1]
            offset_x, offset_y = chromosome[2], chromosome[3]
            n_x, n_y = chromosome[4], chromosome[5]
            off2_x, off2_y = chromosome[6], chromosome[7]
            layout_ind = chromosome[9]
            grid_angle = chromosome[8]
            objects_angles = chromosome[10:]
            amount = len(objects_angles)
            previous_grid = self.get_cells_grids(main_point, [offset_x, offset_y], dioganal, int(n_x), p_x, off2_x,
                                                 int(n_y), p_x, off2_y, grid_angle)
            grid = self.cut_by_rectagular(previous_grid, window[0], window[1])
            centre_points = self.sorting_drops_bydistant(grid, main_point)
            centre_points = self.get_centre_pints_opinion(centre_points, amount, layout_ind)
            rectangulars = self.get_objects_angle(amount, p_x, p_y, objects_angles)
            rects = self.locate_objects(rectangulars, centre_points)

            plt.axis('equal')
            plt.plot(grid[:, 0], grid[:, 1], 'o')
            for rect in rects:
                x_list = np.append(rect[:, 0], rect[0, 0])
                y_list = np.append(rect[:, 1], rect[0, 1])
                plt.plot(x_list, y_list, color=color)

            object_class.append(rects)

        plt.show()
        return np.array(object_class, dtype='object')

    def distant_between_two_classes(self, rects_1, rects_2):
        amount_rects_1 = rects_1.shape[0]
        amount_rects_2 = rects_2.shape[0]
        rects_1_matrix = rects_1[np.newaxis, :, :, :]
        rects_2_matrix = rects_2[np.newaxis, :, :, :]

        rects_1_matrix = np.repeat(rects_1_matrix, amount_rects_2, axis=0)
        rects_2_matrix = np.repeat(rects_2_matrix, amount_rects_1, axis=0)

        x_distances = rects_1_matrix[:, :, :, 0] - rects_2_matrix[:, :, :, 0].transpose((1, 0, 2))
        y_distances = rects_1_matrix[:, :, :, 1] - rects_2_matrix[:, :, :, 1].transpose((1, 0, 2))

        distances = (x_distances**2 + y_distances**2)**0.5

        return distances

    def distant_between_classes(self, rects_classes, minimal_distances):
        print(minimal_distances)
        object_distant = 0
        number_classes = rects_classes.shape[0]
        for i in range(number_classes):
            for j in range(number_classes):
                if i != j:
                    distances = self.distant_between_two_classes(rects_classes[i], rects_classes[j])
                    distances_mistake = (distances < minimal_distances[i, j])*1
                    distances_mistake = np.sum(distances_mistake, axis=1)





if __name__ == "__main__":
    Classes = {
        'Workplace': {"Amount": 12, "rectangular_x": 2, "rectangular_y": 1, 'Environment_x': 4, "Environment_y": 3,
                      "Need_lighting": 9,
                      "Classes_for_short_path": ["Printers", "Cabinets"], "Classes_ignored_intersections": ["lamp"],
                      "Classes_for_distant": {"Machine_tool": 8}},
        # 'Printers': {"Amount": 3, "rectangular_x": 1, "rectangular_y": 1, 'Environment_x': 3, "Environment_y": 3,
        #              "Need_lighting": 9,
        #              "Classes_for_short_path": ["Workplace"], "Classes_ignored_intersections": ["lamp"],
        #              "Classes_for_distant": {"Machine_tool": 9}},
        # 'Cabinets': {"Amount": 4, "rectangular_x": 0.5, "rectangular_y": 2, 'Environment_x': 1.5, "Environment_y": 1,
        #              "Need_lighting": 6,
        #              "Classes_for_short_path": ["Workplace"], "Classes_ignored_intersections": ["lamp"],
        #              "Classes_for_distant": {"Machine_tool": 5}},
        # 'lamp': {"Amount": 60, "rectangular_x": 0.4, "rectangular_y": 0.4, 'Environment_x': 0.4, "Environment_y": 0.4,
        #          "Need_lighting": "reverse",
        #          "Classes_for_short_path": [None],
        #          "Classes_ignored_intersections": ["Workplace", "Printers", "Cabinets", "Machine_tool"],
        #          "Classes_for_distant": {None}},
        'Machine_tool': {"Amount": 4, "rectangular_x": 3, "rectangular_y": 4, 'Environment_x': 8, "Environment_y": 8,
                         "Need_lighting": 8,
                         "Classes_for_short_path": ["Printers", "Cabinets"], "Classes_ignored_intersections": ["lamp"],
                         "Classes_for_distant": {"Workplace": 8, "Printers": 9, "Cabinets": 5}}
    }

    Opt = Optimizer(Classes)
