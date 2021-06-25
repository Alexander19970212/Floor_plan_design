import numpy as np
import random
import matplotlib.pyplot as plt
from evolutionary_search import EvolSearch
from derected_search import DirSearch
from map_search import MapSearch
from PIL import Image
from shapely.ops import unary_union
from shapely.geometry import Polygon, mapping


#  debug distance between classes+
#  check sum by axis+
#  don't forget to get mask for mutation+

#  artist function
#  result drowing function
#  save minimal for each dynasty
#  right gen code for minimal fitness function


def closest(lst, k):
    """
    Function searches the nearest values in list.
    :param lst: list for searching
    :param k: k object for search
    :return: The nearest values from list to k.
    """
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - k))]


class Optimizer:
    def __init__(self, classes):
        self.Classes = classes
        self.colors = []
        for class_name in self.Classes:
            self.colors.append(self.Classes[class_name]["Color"])

        #  initialization parameters of genes: bounds (limits or list of variants, which variable can be),
        #                                      probability_mask (density of probability in bounds)
        self.bounds = []
        self.probability_mask = []
        self.windows_lines = np.array([])
        self.light_coefficients = self.get_light_coefficients()
        self.bounds_constructor()
        self.probability_mask_constructor()
        gen = self.gen_constructor()

        #  that parameters are used for definition case plan. Windows are ares for classes where it could be.
        #  Main_points are points for each class. That points point are used for sorting cells of grid by distant.
        #  In further these parameters should be obtain by other class.
        # self.windows = np.array(
        #     [[[5, 100], [100, 5]], [[5, 100], [100, 5]], [[5, 100], [100, 5]], [[100, 40], [165, 5]],
        #      [[100, 75], [165, 45]]])
        lines_index = [0, 3, 4, 5]
        self.windows = np.array([[(7, 15), (7, 22), (20, 22), (20, 15)], [(4, 8), (4, 18), (12, 18), (12, 8)],
                                 [(7, 4), (7, 11), (20, 11), (20, 4)]]) * 10

        self.main_points = np.array([[5, 5], [45, 55], [95, 35], [165, 40], [135, 45],
                                     [5, 5], [45, 55], [95, 35], [165, 40], [135, 45],
                                     [5, 5], [45, 55], [95, 35]])
        self.max_diagonal = 300
        self.windows_lines = self.get_win_lines(self.windows, lines_index)
        # self.windows_lines = np.array([[[150, 0], [0, 0]], [[0, 0], [0, 300]], [[0, 300], [150, 300]]])
        # self.windows_lines = np.array([[[150, 0], [0, 0]], [[0, 0], [0, 300]]])
        #  That part of code is used for testing object function without GA.

        gen = self.gen_constructor()
        obj_classes, obj_centres, obj_all_centress = self.builder(gen, self.windows, self.main_points,
                                                                  self.max_diagonal)
        mat_dist = self.get_minimal_dist_mat()
        object_distant_value, result_broken_gen, obj_dist, sep_val_dist = self.distant_between_classes(obj_classes,
                                                                                                       mat_dist)
        self.light_object_function(obj_centres, self.windows_lines, self.light_coefficients, gen)
        self.constructor_broken_gen(result_broken_gen, gen)

        self.master_slave_function(obj_centres)

        # gen = self.gen_constructor()
        # coefficients = self.calibration_function(gen)
        # new_gen = self.gen_constructor()
        # result = self.fitness_function(new_gen, coefficients)

        #  there is using that code row because classes don't have similar lengths.

        self.probability_mask = np.array(self.probability_mask, dtype="object")
        self.coefficients = np.array([])
        self.best_evol_individuals = np.array([])

        self.evol_optimization()
        # self.direct_optimization()
        self.map_optimization()
        self.artist("test_gens.txt", 'test_values.txt', gen)  # Plot rendering and saving as GIF

    def evol_optimization(self):

        evol_params = {
            'num_processes': 4,  # (optional) number of processes for multiprocessing.Pool
            'pop_size': 60,  # population size
            'fitness_function': self.fitness_function,  # custom function defined to evaluate fitness of a solution
            'calibration_function': self.calibration_function,
            'elitist_fraction': 2,  # fraction of population retained as is between generations
            'bounds': self.bounds,  # limits or list of variants, which variable can be
            'probability_mask': self.probability_mask,  # density of probability in bounds
            'num_branches': 6  # Amount of dynasties
        }

        es = EvolSearch(evol_params)  # Creating class for evolution search

        '''OPTION 1
        # execute the search for 100 generations
        num_gens = 100
        es.execute_search(num_gens)
        '''

        '''OPTION 2'''
        # keep searching till a stopping condition is reached
        num_gen = 0  # counter of pops
        max_num_gens = 2  # Maximal amount of pops
        desired_fitness = 0.05  # sufficient value of object function for finishing

        es.step_generation()  # Creating the first population

        #  Evolutionary search will be stopped if population counter is exceeded or satisfactory solution is found
        while es.get_best_individual_fitness() > desired_fitness and num_gen < max_num_gens:
            print('Gen #' + str(num_gen) + ' Best Fitness = ' + str(es.get_best_individual_fitness()))
            self.save_best_gen(es.get_best_individual()[0], "test_gens.txt")  # saving the best individual
            self.save_dynasties(es.get_dynasties_best_value(),
                                'test_values.txt')  # saving the best fitness values for each dynasties
            es.step_generation()  # Creating new population
            num_gen += 1

        # print results
        # print('Max fitness of population = ', es.get_best_individual_fitness())
        # print('Best individual in population = ', es.get_best_individual())
        self.coefficients = es.get_coefficients()  # Getting found coefficients for recount getting plot result
        self.best_evol_individuals = es.get_best_individual()[0]

    def direct_optimization(self):

        evol_params = {
            'num_processes': 4,  # (optional) number of processes for multiprocessing.Pool
            'fitness_function': self.function_for_sep,  # custom function defined to evaluate fitness of a solution
            'bounds': self.bounds,  # limits or list of variants, which variable can be
            'probability_mask': self.probability_mask,  # density of probability in bounds
            'first_pop': self.best_evol_individuals,
            'coefficients': self.coefficients
        }

        es = DirSearch(evol_params)  # Creating class for evolution search

        '''OPTION 1
        # execute the search for 100 generations
        num_gens = 100
        es.execute_search(num_gens)
        '''

        '''OPTION 2'''
        # keep searching till a stopping condition is reached
        num_gen = 0  # counter of pops
        max_num_gens = 30  # Maximal amount of pops
        desired_fitness = 0.05  # sufficient value of object function for finishing

        es.step_generation()  # Creating the first population

        #  Evolutionary search will be stopped if population counter is exceeded or satisfactory solution is found
        while es.get_best_individual_fitness() > desired_fitness and num_gen < max_num_gens:
            print('Gen #' + str(num_gen) + ' Best Fitness = ' + str(es.get_best_individual_fitness()))
            self.save_best_gen(es.get_best_individual(), "test_gens.txt")  # saving the best individual
            self.save_dynasties(es.get_dynasties_best_value(),
                                'test_values.txt')  # saving the best fitness values for each dynasties
            es.step_generation()  # Creating new population
            num_gen += 1

            # print results
        # print('Max fitness of population = ', es.get_best_individual_fitness())
        # print('Best individual in population = ', es.get_best_individual())
        self.best_evol_individuals = es.get_best_individual()

    def map_optimization(self):

        strategy_list = [0, 2, 4, 6, 11, 12, 8, 9, 10, 1, 3, 5, 7, 0, 2, 4, 6, 11, 12, 11, 12, 11, 12, 11, 12]
        function_starategy_list = np.array([[1, 0, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 0, 0, 0],
                                            [1, 0, 0, 0],
                                            [1, 0, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 0, 0, 1],
                                            [1, 0, 0, 1],
                                            [1, 0, 0, 1],
                                            [1, 0, 0, 1],
                                            [1, 0, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 0, 0, 0],
                                            [1, 0, 0, 0],
                                            [1, 0, 0, 0],
                                            [1, 0, 0, 0],
                                            [1, 0, 0, 0],
                                            [1, 0, 0, 0],
                                            [1, 0, 0, 0],
                                            [1, 0, 0, 0],
                                            ])

        evol_params = {
            'num_processes': 16,  # (optional) number of processes for multiprocessing.Pool
            'fitness_function': self.function_for_sep,  # custom function defined to evaluate fitness of a solution
            'first_gen': self.best_evol_individuals,
            'coefficients': self.coefficients,
            'function_get_oth_indexes': self.function_get_oth_indexes
        }

        es = MapSearch(evol_params)  # Creating class for evolution search

        '''OPTION 1
        # execute the search for 100 generations
        num_gens = 100
        es.execute_search(num_gens)
        '''

        '''OPTION 2'''
        # keep searching till a stopping condition is reached
        num_gen = 0  # counter of pops
        max_num_gens = 5  # Maximal amount of pops
        desired_fitness = 0.05  # sufficient value of object function for finishing

        es.step_generation()  # Creating the first population

        for cl_obj, cl_funtions_indexes in zip(strategy_list, function_starategy_list):
            es.set_class_for_opt(cl_obj)
            es.set_function_indexes(cl_funtions_indexes)
            num_gen = int((len(self.best_evol_individuals[cl_obj]) - 9) / 2)
            #  Evolutionary search will be stopped if population counter is exceeded or satisfactory solution is found
            for object_index in range(0, num_gen):
                # while es.get_best_individual_fitness() > desired_fitness and num_gen < max_num_gens:

                print('Class #' + str(cl_obj) + ' ' + str(object_index) + '/' + str(num_gen) + ' Best Fitness = ' + str(
                    es.get_best_individual_fitness()))
                es.set_current_index(object_index)
                self.save_best_gen(es.get_best_individual(), "test_gens.txt")  # saving the best individual
                self.save_dynasties(es.get_dynasties_best_value(),
                                    'test_values.txt')  # saving the best fitness values for each dynasties
                es.step_generation()  # Creating new population
            # num_gen += 1

        # print results
        # print('Max fitness of population = ', es.get_best_individual_fitness())
        # print('Best individual in population = ', es.get_best_individual())

    def get_light_coefficients(self):
        light_coeffs = []
        for obj_class in self.Classes:
            light_coeffs.append(self.Classes[obj_class]["Need_lighting"])

        return light_coeffs

    def get_win_lines(self, windows_rects, lines_index):
        polygons = []

        for rectangle in windows_rects:
            polygons.append(Polygon(rectangle))

        polygons = unary_union(polygons)

        points = np.array(mapping(polygons)['coordinates'][0])
        lines = []

        for i in range(points.shape[0] - 1):
            lines.append([points[i], points[i + 1]])

        lines = np.array(lines)

        return lines[lines_index]

    def get_minimal_dist_mat(self):
        """
        The function creates 2-d matrix with distant between classes. That is based on parameters
        in classes' description.
        :return: 2-d numpy array with distances.
        """
        minimal_distances = []  # 2d matrix
        for class_1 in self.Classes:
            min_dist_one_axes = []  # row of 2d matrix
            for class_2 in self.Classes:
                if class_1 != class_2:  # only for different classes
                    try:  # in the case if minimal distant is described
                        dist = self.Classes[class_1]["Classes_for_distant"][class_2]
                    except:
                        dist = 0  # 0 means distant between classes is not considered
                else:
                    dist = 0
                min_dist_one_axes.append(dist)
            minimal_distances.append(min_dist_one_axes)

        return np.array(minimal_distances)

    def bounds_constructor(self):
        """
        Function creates limits or list of variants which could be as variable in genes.
        Description:
        1: x grid step (1-2 environment rectangular x)
        2: y grid step (1-2 environment rectangular y)
        3-4: offset (x, y) grid form main point (0-1 environment rectangular)
        5-6: Small group shape (x, y) of grid
        7-8: x, y distances between small group (1.5 - 3 environment rectangular)
        9: Rotation of grid (list of degrees)
        11 - ... : normalized locations' indexes' list
        ... - last: rotation angle for each component
        :return: None
        """
        for kind in self.Classes:
            layer_parameters = [[self.Classes[kind]['Environment_x'], self.Classes[kind]['Environment_x'] * 2],
                                [self.Classes[kind]['Environment_y'], self.Classes[kind]['Environment_y'] * 2],
                                [0, self.Classes[kind]['Environment_x']], [0, self.Classes[kind]['Environment_y']],
                                [1, 10], [1, 10],
                                [self.Classes[kind]['Environment_x'] * 1.5, self.Classes[kind]['Environment_x'] * 3],
                                [self.Classes[kind]['Environment_y'] * 1.5, self.Classes[kind]['Environment_y'] * 3],
                                [0, 45, 90]]
            for drop in range(0, self.Classes[kind]["Amount"]):  # locations of each component on grid
                layer_parameters.append([0, 1])
            for drop in range(0, self.Classes[kind]["Amount"]):
                layer_parameters.append([0, 45, 90])  # same variants list for each component
            self.bounds.append(layer_parameters)

        #  there is using that code row because classes don't have similar lengths.
        self.bounds = np.array(self.bounds, dtype="object")

    def probability_mask_constructor(self):
        """
        The function describes density of probability in bounds.
        int or float - offset in triangular density of probability
        'Equally_distributed' - equally density in whole bounds
        list - probability for each element in bounds list (sum should be 1)
        :return:
        """
        for kind in self.Classes:
            layer_parameters = [0, 0,  # x, y grid steps
                                0, 0,  # Offset_x, Offset_y from main point
                                0.5, 0.5,  # Shape (x, y) of small group
                                'Equally_distributed', 'Equally_distributed',  # distances (x, y) between small groups
                                [0.4, 0.2, 0.4]]  # grid rotation angle
            for drop in range(0, self.Classes[kind]["Amount"]):
                layer_parameters.append('Equally_distributed')  # location for each component
            for drop in range(0, self.Classes[kind]["Amount"]):
                layer_parameters.append([0.4, 0.2, 0.4])  # rotation for each component
            self.probability_mask.append(layer_parameters)

        #  there is using that code row because classes don't have similar lengths.
        self.probability_mask = np.array(self.probability_mask, dtype="object")

    def gen_constructor(self):
        """
        The function creates gen based on list probability and bounds.
        :return: One gen as numpy array
        """
        big_gen = []
        for net_mask, net_doubt in zip(self.probability_mask, self.bounds):  # cycle for each grid
            net_gen = []
            for probability, value_slot in zip(net_mask, net_doubt):

                # for (x, y) grid steps, (x, y) offset from main point, shape of small group, index distribution
                if type(probability) is int:
                    net_gen.append(random.triangular(value_slot[0], value_slot[1],
                                                     value_slot[0] + (value_slot[1] - value_slot[0]) * probability))
                elif type(probability) is float:
                    net_gen.append(random.triangular(value_slot[0], value_slot[1],
                                                     value_slot[0] + (value_slot[1] - value_slot[0]) * probability))
                #  if probability distribution is equally
                elif type(probability) is str:
                    net_gen.append(random.uniform(value_slot[0], value_slot[1]))

                #  if probability is defended by list
                else:
                    net_gen.append(random.choices(value_slot, cum_weights=probability, k=1)[0])

            big_gen.append(net_gen)

        # returning gen as numpy array
        return np.array(big_gen, dtype='object')

    def test_function(self, gen):
        """
        The function gets gen, builds individual, analyzes it and creates mask list with broken parts.
        :param gen: Numpy array - list of variables.
        :return: list of penalty values and broken mask
        """
        # building floor plan which based on gen.
        obj_classes, obj_centres, obj_all_centress = self.builder(gen, self.windows, self.main_points,
                                                                  self.max_diagonal)  # list rectangles

        if obj_classes == False:
            return False, False

        mat_dist = self.get_minimal_dist_mat()  # getting matrix of minimal distant between classes

        # getting distant sum between classes' rectangles, mask gen where distant less then allowed,
        # sum amount if its
        object_distant_value, result_broken_gen, dist_value, sep_val_dist = self.distant_between_classes(obj_classes,
                                                                                                         mat_dist)
        light_distance_sum, broken_gen_light, sep_val_light = self.light_object_function(obj_centres,
                                                                                         self.windows_lines,
                                                                                         self.light_coefficients, gen)
        # getting full broken mask
        master_slave_sum, broken_gen_master_slave, sep_val_master_slave = self.master_slave_function(obj_centres)
        mask, amount_inters = self.constructor_broken_gen(result_broken_gen, gen)

        return np.array([object_distant_value, dist_value, light_distance_sum, master_slave_sum]), broken_gen_light

    def function_get_oth_indexes(self, gen, class_ind, obj_ind):
        x_vector = np.array(gen, dtype="object")
        try:
            object_class = []
            object_centres = []
            #  classes of objects are built separately

            chromosome = gen[class_ind]
            main_point = self.main_points[class_ind]
            # gen parsing
            p_x, p_y = chromosome[0], chromosome[1]
            offset_x, offset_y = chromosome[2], chromosome[3]
            n_x, n_y = chromosome[4], chromosome[5]
            off2_x, off2_y = chromosome[6], chromosome[7]
            grid_angle = chromosome[8]
            locations_angles_amount = len(chromosome[9:])
            locations_indexes = chromosome[9:int(9 + locations_angles_amount / 2)]
            objects_angles = chromosome[int(9 + locations_angles_amount / 2):]
            amount = len(objects_angles)

            # Get grid that is covering all space.
            previous_grid = self.get_cells_grids(main_point, [offset_x, offset_y], self.max_diagonal, int(n_x), p_x,
                                                 off2_x,
                                                 int(n_y), p_x, off2_y, grid_angle)

            # Remove points which are outside counter.
            grid = self.cut_by_some_rectangles(previous_grid, self.windows, p_x, p_y)

            centre_points = self.sorting_drops_bydistant(grid, main_point)  # Sorting by distant from main point

            #  get cells for location according locations' indexes
            centre_points, other_indexes, corrective_indexes = self.get_centre_points_option_4(centre_points,
                                                                                            locations_indexes)

            return other_indexes, corrective_indexes
        except:
            return [obj_ind], [obj_ind]

    def function_for_sep(self, gen, weights, functions_indexes):
        """
        The function gets gen, builds individual, analyzes it and creates mask list with broken parts.
        :param gen: Numpy array - list of variables.
        :return: list of penalty values and broken mask
        """

        x_vector = np.array(gen, dtype="object")
        test_attention = np.array([0.35, 0.35, 0.1, 0.2])  # influence penalty values to result

        try:
            # building floor plan which based on gen.
            obj_classes, obj_centres, obj_all_centress = self.builder(gen, self.windows, self.main_points,
                                                                      self.max_diagonal)  # list rectangles
            mat_dist = self.get_minimal_dist_mat()  # getting matrix of minimal distant between classes

            # getting distant sum between classes' rectangles, mask gen where distant less then allowed,
            # sum amount if its
            object_distant_value, result_broken_gen, dist_value, sep_val_dist = self.distant_between_classes(
                obj_classes,
                mat_dist)
            light_distance_sum, broken_gen_light, sep_val_light = self.light_object_function(obj_centres,
                                                                                             self.windows_lines,
                                                                                             self.light_coefficients,
                                                                                             gen)

            master_slave_sum, broken_ge_master_slave, sep_val_master_slave = self.master_slave_function(obj_centres)

            # print(master_slave_sum)

            vector_values = self.balancing_function([sep_val_dist, sep_val_light])

            result = np.sum(test_attention * functions_indexes * np.array(
                [object_distant_value, dist_value, light_distance_sum, master_slave_sum]) / weights)

            # print(result)

            return [result, np.array(sep_val_light, dtype="object")]

        except:
            return [1, False]

    def balancing_function(self, sep_values):

        bufer_for_analyz = []
        for val_funct in sep_values:
            for_combine = []
            for chr in val_funct:
                for_combine.extend(chr)
            bufer_for_analyz.append(for_combine)

        bufer_for_analyz = np.array(bufer_for_analyz)

        maxes = np.amax(bufer_for_analyz, axis=1)
        coeff = np.prod(maxes)

        coefficients = coeff / maxes

        new_values = []
        for val_funct, coeffinc in zip(sep_values, coefficients):
            new_gen = []
            for chr in val_funct:
                new_gen.append(chr * coeffinc)
            new_values.append(new_gen)

        # new_values = np.array(new_values, dtype="object")
        balancing_values = new_values[0]
        result_values = []

        for index_funct in range(1, len(new_values)):
            for chr, chr_bal in zip(new_values[index_funct], balancing_values):
                result_values.append(chr + chr_bal)
            balancing_values = result_values
            result_values = []

        return balancing_values

    def calibration_function(self, x_vector):
        """
        The function is used for automated scaling (selecting penalty weights). That returns only penalty values
        :param x_vector: numpy array - gen
        :return: numpy array - list of penalty values
        """
        x_vector = np.array(x_vector, dtype="object")
        penalties, mask = self.test_function(x_vector)
        if type(penalties) is bool:
            return np.array([0, 0, 0, 0])
        return penalties

    def fitness_function(self, gen, weights):
        """
        The function calculates fitness function as sum of penalty values multiplied by penalty weights.
        :param gen: numpy array - list variables
        :param weights: list of penalty weights, which was gotten in automated scaling.
        :return: float (0-1) - fitness values, numpy array - broken mask
        """
        x_vector = np.array(gen, dtype="object")
        test_attention = np.array([0.35, 0.35, 0.1, 0.2])  # influence penalty values to result

        try:
            penalties, mask = self.test_function(x_vector)  # getting list penalty values and broken mask
            if type(penalties) is bool:
                result = 1
                mask = []
                for grid in gen:
                    mask.append(np.ones_like(np.array(grid)))
            else:
                result = np.sum(test_attention * penalties / weights)
        except:  # if floor plan could be built by gen
            result = 1
            mask = []
            for grid in gen:
                mask.append(np.ones_like(np.array(grid)))

        # This cod rows turn off directed evolution. That creates broken mask with only ones.
        if result >= 0.6:
            mask = []
            for grid in gen:
                mask.append(np.ones_like(np.array(grid)))

        return [result, np.array(mask, dtype='object')]

    def get_cells_grids(self, main_point, offset_grid, max_rad, n_x, p_x, dist_x, n_y, p_y, dist_y, deg):
        """
        The function builds grid and returns cells which are separated to smalls groups.
        :param main_point: list x, y coords - point for base cell of grid
        :param offset_grid: list x, y distant - offset of base cell from main point
        :param max_rad: int (float) distant - cells exist only inside circle with that radius
        :param n_x: int - cells' amount in x row in small group
        :param p_x: float - x grid step in small groups
        :param dist_x: float - x distant between small group
        :param n_y: int - cells' amount in y row in small group
        :param p_y: float - y grid step in small group
        :param dist_y: y distant between small groups
        :param deg: rotation of whole grid (degree)
        :return: numpy array with coords of cells
        """
        # Get cells in the first quarter
        xy = np.mgrid[0:max_rad:n_x * p_x + dist_x, 0:max_rad:n_y * p_y + dist_y].reshape(2, -1).T

        #  Copy to the fourth quarter
        xy_copy = np.copy(xy)
        xy_copy[:, 1] *= -1
        xy = np.append(xy, xy_copy, axis=0)

        #  Copy to the second and third quarters
        xy_copy = np.copy(xy)
        xy_copy[:, 0] *= -1
        xy = np.append(xy, xy_copy, axis=0)

        # remove repetitive values
        bases_grid = np.unique(xy, axis=0)
        bases_grid_copy = bases_grid.copy()

        # copy as smalls group
        for r_x in range(n_x):
            for r_y in range(n_y):
                xy_ = bases_grid_copy + [r_x * p_x, r_y * p_y]
                bases_grid = np.append(bases_grid, xy_, axis=0)

        # rotation
        if deg != 0:
            bases_grid = self.rotation(bases_grid, deg)

        # move to main point and shift by offset
        bases_grid = bases_grid + main_point
        bases_grid = bases_grid + offset_grid

        # For debugging: show grid

        # print(bases_grid.shape)
        # plt.plot(bases_grid[:, 0], bases_grid[:, 1], 'o')
        # plt.show()

        return bases_grid

    def cut_by_rectagular(self, grid, up_left_cords, down_right_cords):
        """
        The function returns only points within window.
        :param grid: list - coords (x, y) of all grid cells
        :param up_left_cords: list (shape = (2, 1)) coords (x, y) of left up corner
        :param down_right_cords: list (shape = (2, 1)) coords (x, y) of raght down
        :return: list - coords (x, y) of cells inside window
        """
        grid = grid[grid[:, 0] > up_left_cords[0]]
        grid = grid[grid[:, 0] < down_right_cords[0]]
        grid = grid[grid[:, 1] < up_left_cords[1]]
        grid = grid[grid[:, 1] > down_right_cords[1]]

        return grid

    def cut_by_some_rectangles(self, grid, rectangles, p_x, p_y):
        """
        The function returns points, which only in rectangles.
        :param grid: numpy array - coords(x, y) of all points
        :param rectangles: list - coords of rectangles' corners.
        :return: numpy array - coords (x, y) of points in rectangles.
        """
        result = np.array([])

        # check each rectangle if points are in it.
        for rectangle in rectangles:

            # get corners' coords (for the case if corners were not rowed).
            up_coords = max(rectangle[:, 1]) - p_y / 2
            down_coords = min(rectangle[:, 1]) + p_y / 2
            left_coords = min(rectangle[:, 0]) + p_x / 2
            right_coords = max(rectangle[:, 0]) - p_x / 2

            #  for the first points or if result list is empty.
            if result.size == 0:
                result = np.array(self.cut_by_rectagular(grid, [left_coords, up_coords], [right_coords, down_coords]))

            #  detected points are added into result list.
            else:
                result = np.append(result,
                                   self.cut_by_rectagular(grid, [left_coords, up_coords], [right_coords, down_coords]),
                                   axis=0)

        result = np.array(result)

        return np.unique(result, axis=0)  # remove repetitive points

    def get_objects_angle(self, amount, p_x, p_y, rotation_list):
        """
        The function creates rectangles (list coords) and rotates it by angles list.
        :param amount: int - amount of rectangles
        :param p_x: float - x size of rectangle
        :param p_y: float - y size of rectangle
        :param rotation_list: list - angles of rotation of each rectangle
        :return: numpy array - list of rectangles coords
        """
        rects = np.array([[[-p_x, p_y], [p_x, p_y], [p_x, -p_y], [-p_x, -p_y]]]) / 2
        rects = np.repeat(rects, amount, axis=0)
        rotated_rects = []
        for rect, angle in zip(rects, rotation_list):
            if angle == 0:
                rotated_rects.append(self.rotation(rect, angle))
            else:
                rotated_rects.append(rect)

        return np.array(rotated_rects)

    def sorting_drops_bydistant(self, drops, main_point):
        """
        The function sorts cells of grid by distant from main point.
        :param drops: numpy array - coords list of cells
        :param main_point: list - (x, y) coords of point.
        :return: numpy array sorted by distant from main point
        """
        drops = np.unique(drops, axis=0)  # remove repetitive cells

        # get distant list
        distant = ((drops[:, 0] - main_point[0]) ** 2 + (drops[:, 1] - main_point[1]) ** 2) ** 0.5
        return drops[np.argsort(distant)]

    def get_centre_pints_opinion(self, drops, n, index):
        """
        The function return n points from list drops. This points are got by offset which are depended by index (0, 1).
        0 means zero offset from first index. 1 means offset from last index.
        :param drops: numpy array - coords' list of permitted cells
        :param n:  int - amount of cells which have to be returned
        :param index: float (0-1) - index of option
        :return: numpy array - coords list of n points
        """
        # scaling and getting offset
        n_drops = drops.shape[0]
        offset = int(index * (n_drops - n))

        return drops[offset:offset + n, :]

    def get_centre_points_option_2(self, drops, n, index):
        """
        The function searches all opinions of distribution, normalizes theirs' amount and returns one of them by index.
        The function isn't used due to so much amount of opinions.
        :param drops: numpy array - coords' list if permitted cells
        :param n: int - amount of cells which have to be returned
        :param index: float (0-1) - index of opinion
        :return: numpy array - coords list of n points
        """
        binary_length = '1' * n + '0' * (drops.shape[0] - n)  # get max limit opinion
        max_dec_number = int(binary_length, 2)  # transforming binary max limit to dec int

        # get all options like arrange to max index
        dec_numbers = np.arange(int('1' * n, 2), max_dec_number, int((max_dec_number - int('1' * n, 2)) / 1000))

        # transforming all options like binary code list
        binary_list = ((dec_numbers[:, None] & (1 << np.arange(drops.shape[0]))) > 0).astype(int)
        binary_list = binary_list[0, :]

        # get options where amount of ones equals n
        binary_sum = np.sum(binary_list, axis=0)
        all_options = binary_list[binary_sum == n]

        # normalizing index and getting the nearest option to it
        option_index = int(index * drops.shape[0])
        selected_option = all_options[option_index]
        return drops[selected_option == 1]

    def get_centre_points_option_3(self, drops, n, index):
        """
        The function search 1000 usable options of distribution and returns one of them by index.
        The function isn't used due to good options is missed.
        :param drops: numpy array - coords' list of permitted cells
        :param n: int - amount of cells which have to be returned
        :param index: float (0-1) - index of option
        :return: numpy array - coords list of n points
        """

        # In the cycle counter is transformed into binary. If sum of ones in it is n, that binary code will be saved as
        # option. Cycle continues while amount of options less then threshold.
        all_options = []
        count = 0
        i = 0
        while count <= 1000:
            if bin(i).count('1') == n:
                option = [int(x) for x in list(format(i, '0' + str(drops.shape[0]) + 'b'))]
                all_options.append(option)
                count += 1
            i += 1

        all_options = np.array(all_options)

        # normalizing index and getting the nearest option
        option_index = int(index * all_options.shape[0])
        selected_option = all_options[option_index]
        returned_drops = drops[selected_option == 1]
        return returned_drops

    def get_centre_points_option_4(self, drops, indexes):
        """
        The function transformers list of indexes (0-1) to list of points' coords (first cell - last cell)
        :param drops: numpy array - coords' list of permitted cells
        :param indexes: list of floats (0-1) - normalized indexes of locations
        :return: numpy array - coords list of n points
        """
        # Create range list indexes of each drops. Scale float indexes to int (0 - index of last cell) list.
        # For each index in cycle found the closest, remove it from previous so that couldn't be repeated.
        all_indexes = list(range(drops.shape[0]))
        selected_indexes = []
        for index in indexes:
            founded_index = closest(all_indexes, index * drops.shape[0])
            selected_indexes.append(founded_index)
            all_indexes.remove(founded_index)

        all_indexes = np.array(all_indexes) / drops.shape[0]
        corrective_indexes = np.array(selected_indexes) / drops.shape[0]

        return drops[selected_indexes], all_indexes, corrective_indexes

    def locate_objects(self, rects, centre_points):
        """
        The function moves rectangles to grid cells using centre of rectangles.
        :param rects: numpy array - coords rectangles' list all centres of which in (0, 0)
        :param centre_points: numpy array - grid coords' list.
        :return: numpy array - coords' list offset rectangles.
        """
        # centre points are transformed to size like rects by adding new axis and repeating for four corners.
        centre_points = centre_points[:, np.newaxis, :]
        centre_points = np.repeat(centre_points, 4, axis=1)
        return rects + centre_points

    def rotation(self, drops, deg):
        """
        The function rotates points in 2-d space relatively (0, 0).
        :param drops: numpy array - coords' list of points for rotation.
        :param deg: int or float - angle for rotation in degree.
        :return: numpy array - coords' list of rotated points.
        """
        # Get rotation matrix.
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
        """
        The function build floor plan according gen.
        :param gen: numpy array - x vector. It includes description for each class.
                    Description:
                    0: x grid step (1-2 environment rectangular x)
                    1: y grid step (1-2 environment rectangular y)
                    2-3: offset (x, y) grid form main point (0-1 environment rectangular)
                    4-5: Small group shape (x, y) of grid
                    6-7: x, y distances between small group (1.5 - 3 environment rectangular)
                    8: Rotation of grid (list of degrees)
                    9 - ... : normalized locations' indexes' list
                    ... - last: rotation angle for each component
        :param windows: numpy array - list of rectangles which constitute counter.
        :param main_points: numpy array - list points for sorting grids' points
        :param dioganal: int (float) - max dimension of counter
        :return: numpy array - coords of rectangles in floor plan
        """
        colors = ['red', 'blue', 'yellow', 'black', 'green']  # for showing result that step of necessity.
        colors = colors[:gen.shape[0]]

        object_class = []
        object_centres = []
        object_all_centres = []
        #  classes of objects are built separately
        for chromosome, main_point in zip(gen, main_points):
            # gen parsing
            p_x, p_y = chromosome[0], chromosome[1]
            offset_x, offset_y = chromosome[2], chromosome[3]
            n_x, n_y = chromosome[4], chromosome[5]
            off2_x, off2_y = chromosome[6], chromosome[7]
            grid_angle = chromosome[8]
            locations_angles_amount = len(chromosome[9:])
            locations_indexes = chromosome[9:int(9 + locations_angles_amount / 2)]
            objects_angles = chromosome[int(9 + locations_angles_amount / 2):]
            amount = len(objects_angles)

            # Get grid that is covering all space.
            previous_grid = self.get_cells_grids(main_point, [offset_x, offset_y], dioganal, int(n_x), p_x, off2_x,
                                                 int(n_y), p_x, off2_y, grid_angle)

            # Remove points which are outside counter.
            grid = self.cut_by_some_rectangles(previous_grid, windows, p_x, p_y)

            centre_points_pre = self.sorting_drops_bydistant(grid, main_point)  # Sorting by distant from main point

            if (centre_points_pre.shape[0] < amount):
                return False, False, False

            #  get cells for location according locations' indexes
            centre_points, other_indexes, corrective_indexes = self.get_centre_points_option_4(centre_points_pre, locations_indexes)

            # Get rotated rectangles according angles' list
            rectangles = self.get_objects_angle(amount, p_x, p_y, objects_angles)

            # Move rectangles to grids' points.
            rects = self.locate_objects(rectangles, centre_points)

            # plt.axis('equal')
            # plt.plot(grid[:, 0], grid[:, 1], 'o')
            # for rect in rects:
            #     x_list = np.append(rect[:, 0], rect[0, 0])
            #     y_list = np.append(rect[:, 1], rect[0, 1])
            #     plt.plot(x_list, y_list, color=color)

            object_class.append(rects)
            object_centres.append(centre_points)
            object_all_centres.append(centre_points_pre)
        #  plt.show()
        return np.array(object_class, dtype='object'), np.array(object_centres, dtype='object'), np.array(
            object_all_centres, dtype='object')

    def distant_between_two_classes(self, rects_1, rects_2):
        """
        The function calculates distances between rectangles' corners from different objects' classes.
        :param rects_1: numpy array - coords list of rectangles in class 1
        :param rects_2: numpy array - coords list of rectangles in class 2
        :return:  numpy array - distances' list
        """

        # Transform classes matrix to one size.
        # Each matrix expansions in axis 0 by repeating times like number of objects in other class.
        amount_rects_1 = rects_1.shape[0]
        amount_rects_2 = rects_2.shape[0]
        rects_1_matrix = rects_1[np.newaxis, :, :, :]
        rects_2_matrix = rects_2[np.newaxis, :, :, :]
        rects_1_matrix = np.repeat(rects_1_matrix, amount_rects_2, axis=0)
        rects_2_matrix = np.repeat(rects_2_matrix, amount_rects_1, axis=0)

        # Find distances like in Pythagoras theorem
        x_distances = rects_1_matrix[:, :, :, 0] - rects_2_matrix[:, :, :, 0].transpose((1, 0, 2))
        y_distances = rects_1_matrix[:, :, :, 1] - rects_2_matrix[:, :, :, 1].transpose((1, 0, 2))

        distances = (x_distances ** 2 + y_distances ** 2) ** 0.5

        return distances

    def distant_between_classes(self, rects_classes, minimal_distances, flag_debug=False):
        """
        The function finds distances between classes, detects disturbances,
        sums that and builds part of mask where disturbances were detected.
        :param rects_classes: numpy array - classes of objects in built floor plan.
        :param minimal_distances: numpy array - square matrix. Minimal distances between i and j classes
                                    is at the intersection i raw and j column.
        :param flag_debug: bool - for debugging. True - "prints" will be showed.
        :return: Number of disturbances, mask where disturbances are at, numpy array - matrix with disturbances.
        """
        object_distant = 0  # for disturbances sum
        number_classes = rects_classes.shape[0]
        broken_gens = []
        sep_val_gen = []

        if flag_debug:
            print(minimal_distances)

        # check distances inside all couple of classes
        for i in range(number_classes):
            broken_gen = []
            sep_val_chr = []
            for j in range(number_classes):
                if i != j:
                    # get matrix with distances
                    distances = self.distant_between_two_classes(rects_classes[i], rects_classes[j])

                    # find position where distances are exceeded
                    distances_mistake = (distances < minimal_distances[i, j]) * 1
                    mistake_value = (minimal_distances[i, j] - distances) * distances_mistake
                    mistake_value = np.sum(mistake_value, axis=2)
                    mistake_value = np.sum(mistake_value, axis=0)
                    sep_val_chr.append(mistake_value)

                    # Sum disturbances
                    distances_mistake = np.sum(distances_mistake, axis=2)
                    distances_mistake = np.sum(distances_mistake, axis=0)

                    # Save for couple
                    broken_gen.append(distances_mistake)
                    object_distant += np.sum(distances)

            sep_val_gen.append(np.sum(sep_val_chr, axis=0))
            broken_gens.append(broken_gen)

        if flag_debug:
            print('mistake_matrix', broken_gens)

        # Transform to one side broken gen mask
        if number_classes >= 3:
            result_broken_gen = np.sum(broken_gens, axis=1)
        else:
            result_broken_gen = broken_gens

        object_distant_value = 0  # additional sum of disturbances
        for chromosome_index in range(number_classes):
            object_distant_value += np.sum(result_broken_gen[chromosome_index])
            # transform broken mask as part of chromosome
            result_broken_gen[chromosome_index] = (result_broken_gen[chromosome_index] >= 1) * 1

        return object_distant_value, result_broken_gen, object_distant, sep_val_gen

    def light_object_function(self, classes_centre_points, win_lines, light_coefficients, gen_ex):
        """
        The function calculates distances to window and sums it.
        :param classes_centre_points: numpy array - list with rectangles centres' coords
        :param win_lines: list - list with edges' coords of window
        :param light_coefficients: numpy array - how is light needed for class (0-10)
        :param numpy array - gen for example for broken gen creating
        :return: list - sum of light distant for each class, numpy array - piece of broken gen
        """
        light_distance = []
        broken_gen = []
        sep_val_gen = []
        for class_recs in classes_centre_points:
            class_value, class_piece_broken_gen, sep_values_chr = self.light_function_angle_for_class(class_recs,
                                                                                                      win_lines)
            light_distance.append(class_value)
            broken_gen.append(class_piece_broken_gen)
            sep_val_gen.append(sep_values_chr)

        broken_gen = self.get_brokengen_light(broken_gen, gen_ex)
        light_distance_sum = np.sum(np.array(light_distance) * light_coefficients)

        return light_distance_sum, broken_gen, sep_val_gen

    def master_slave_for_class(self, slave, master):

        best_amount_daughter = int(master.shape[0] / slave.shape[0])
        distances = self.distant_between_two_classes(slave[:, np.newaxis, :], master[:, np.newaxis, :])
        distances = np.squeeze(distances, axis=2)
        min_distances = np.repeat([np.amin(distances, axis=1)], slave.shape[0], axis=0).T
        binar_min_dist = (distances == min_distances) * 1
        # anti_binar_min_dis = (binar_min_dist - 1) * (-1)
        distances_for_print_faml = distances * binar_min_dist
        sum_dist_for_printers = np.sum(distances_for_print_faml, axis=0)
        sum_for_printers = np.sum(binar_min_dist, axis=0)
        values_for_class = np.absolute(sum_for_printers - best_amount_daughter)

        result = sum_dist_for_printers * (1 + 0.25 * values_for_class)

        return result

    def master_slave_function(self, classes_centre_points):
        sep_val_gen = []
        broken_gen = []
        sum_penalty = 0
        keys = list(self.Classes.keys())

        for class_obj, class_rects in zip(self.Classes, classes_centre_points):
            if "Master" not in self.Classes[class_obj]:
                sep_val_gen.append(np.zeros(class_rects.shape[0]))
                broken_gen.append(np.zeros(class_rects.shape[0]))
            else:
                master = self.Classes[class_obj]["Master"][0]
                master_index = keys.index(master)
                values_for_class = self.master_slave_for_class(class_rects, classes_centre_points[master_index])
                sep_val_gen.append(values_for_class)

                max_val = np.amax(values_for_class)
                broken_gen.append((values_for_class > 0.75 * max_val) * 1)

                sum_penalty += np.sum(values_for_class)

        broken_gen = np.array(broken_gen, dtype='object')
        sep_val_gen = np.array(sep_val_gen, dtype='object')

        # print(sep_val_gen)

        return sum_penalty, broken_gen, sep_val_gen

    def get_brokengen_light(self, parts_gen, example_gen):
        """
        The function includes parts gen where broken pieces are ones into zero array with size like gen.
        :param parts_gen: list - broken pieces of gen
        :param example_gen: numpy array - gen for example
        :return: numpy array - gen where ones are broken parts
        """
        mask_gen = []
        for chromosome, broken_part in zip(example_gen, parts_gen):
            void_gen = np.zeros_like(chromosome)
            void_gen[9:void_gen.shape[0] - broken_part.shape[0]] = broken_part
            mask_gen.append(void_gen)

        return np.array(mask_gen, dtype='object')

    def light_function_for_class(self, rects_centres, windows):
        """
        THe function calculates distances' sum for each rectangles in class to all windows.
        :param rects_centres: numpy array - list with rectangles' centres' coords
        :param windows: list - list with edges' coords of windows
        :return: float - sum of distances (sum of minimal distances),
                    list - '1' where distances more then 10 cent of max.
        """
        rects_centres = rects_centres[np.newaxis, :, :]
        windows = windows[np.newaxis, :, :, :]
        rects_centres = np.repeat(rects_centres, windows.shape[1], axis=0)
        windows = np.transpose(np.repeat(windows, rects_centres.shape[1], axis=0),
                               [1, 0, 2, 3])  # DEB check tranpose for many axis

        d_1_sqr = (rects_centres[:, :, 0] - windows[:, :, 0, 0]) ** 2 + (
                rects_centres[:, :, 1] - windows[:, :, 0, 1]) ** 2
        d_2_sqr = (rects_centres[:, :, 0] - windows[:, :, 1, 0]) ** 2 + (
                rects_centres[:, :, 1] - windows[:, :, 1, 1]) ** 2

        d_sqr = (windows[:, :, 0, 0] - windows[:, :, 1, 0]) ** 2 + (windows[:, :, 0, 1] - windows[:, :, 1, 1]) ** 2

        d_1 = np.sqrt(d_1_sqr)
        d_2 = np.sqrt(d_2_sqr)
        d = np.sqrt(d_sqr)

        p = (d_1 + d_2 + d) / 2

        s = np.sqrt(p * (p - d_1) * (p - d_2) * (p - d))

        h = (s / d) * 2

        g_1_sqr = d_1_sqr - h ** 2
        g_2_sqr = d_2_sqr - h ** 2

        g_1 = np.sqrt(g_1_sqr)
        g_2 = np.sqrt(g_2_sqr)

        dist_to_point_1 = g_1 + h
        dist_to_point_2 = g_2 + h

        mask_1 = (d_2_sqr < d_sqr + d_1_sqr) * 1
        mask_2 = (d_1_sqr < d_sqr + d_2_sqr) * 1
        mask = ((mask_1 + mask_2) > 1) * 1  # mask  where have to be h
        anti_mask = (mask - 1) * -1

        d_corner_mask = (d_1 < d_2) * 1
        d_corner_anti_mask = (d_corner_mask - 1) * -1
        d_corner = dist_to_point_1 * d_corner_mask + dist_to_point_2 * d_corner_anti_mask

        # d_corner = np.minimum(d_1, d_2)
        min_dist = mask * h + anti_mask * d_corner

        # min_dist = np.sum(min_dist, axis=2)  # FOR 1 OPTION
        min_dist = np.amin(min_dist, axis=0)  # DEBBB

        max_dist = np.amax(min_dist)

        broken_gen = (min_dist > 0.4 * max_dist)
        value = np.sum(min_dist)

        return value, broken_gen, min_dist

    def light_function_angle_for_class(self, rects_centres, windows):
        """
        THe function calculates distances' sum for each rectangles in class to all windows.
        :param rects_centres: numpy array - list with rectangles' centres' coords
        :param windows: list - list with edges' coords of windows
        :return: float - sum of distances (sum of minimal distances),
                    list - '1' where distances more then 10 cent of max.
        """
        rects_centres = rects_centres[np.newaxis, :, :]
        windows = windows[np.newaxis, :, :, :]
        rects_centres = np.repeat(rects_centres, windows.shape[1], axis=0)
        windows = np.transpose(np.repeat(windows, rects_centres.shape[1], axis=0),
                               [1, 0, 2, 3])  # DEB check tranpose for many axis

        d_1_sqr = (rects_centres[:, :, 0] - windows[:, :, 0, 0]) ** 2 + (
                rects_centres[:, :, 1] - windows[:, :, 0, 1]) ** 2
        d_2_sqr = (rects_centres[:, :, 0] - windows[:, :, 1, 0]) ** 2 + (
                rects_centres[:, :, 1] - windows[:, :, 1, 1]) ** 2

        d_sqr = (windows[:, :, 0, 0] - windows[:, :, 1, 0]) ** 2 + (windows[:, :, 0, 1] - windows[:, :, 1, 1]) ** 2

        d_1 = np.sqrt(d_1_sqr)
        d_2 = np.sqrt(d_2_sqr)
        d = np.sqrt(d_sqr)

        cos_a = -(d_sqr - d_1_sqr - d_2_sqr) / (2 * d_1 * d_2)

        # obtus_mask = (cos_a > 0) * 1
        angles = np.arccos(cos_a)

        weight_angle = 3.142 - angles

        # min_dist = np.sum(min_dist, axis=2)  # FOR 1 OPTION
        min_dist = np.amin(weight_angle, axis=0)  # DEBBB

        max_dist = np.amax(min_dist)

        broken_gen = (min_dist > 0.8 * max_dist)
        value = np.sum(min_dist)

        return value, broken_gen, min_dist

    def constructor_broken_gen(self, parts_gen, example_gen):
        #  temprorary
        amount_intersections = 0
        mask_gen = []
        for chromosome, broken_part in zip(example_gen, parts_gen):
            void_gen = np.zeros_like(chromosome)
            void_gen[9:void_gen.shape[0] - broken_part.shape[0]] = broken_part
            void_gen[4] = 1
            void_gen[5] = 1
            amount_intersections += np.sum(broken_part)
            mask_gen.append(void_gen)

        return mask_gen, amount_intersections

    def artist(self, filename_gens, filename_values, gen_example, gif_time=100, type_draw="all"):
        from shapely.ops import unary_union
        from shapely.geometry import Polygon, mapping

        polygons = []

        for rectangle in self.windows:
            polygons.append(Polygon(rectangle))

        polygons = unary_union(polygons)

        countur_coords = np.array(mapping(polygons)['coordinates'][0])

        dynasties_values = []
        with open(filename_values, "r") as file:
            for line in file:
                dynasties_values.append(line.split())

        dynasties_values = dynasties_values[1:]

        dynasties_values = np.float_(dynasties_values)

        gens = []
        with open(filename_gens, "r") as file:
            for line in file:
                gens.append(line.split())

        gens = gens[1:]
        gens = np.float_(gens)
        gens = np.array(gens)

        for gen, values, i in zip(gens, dynasties_values, range(gens.shape[0])):

            print("Plot_number_", i)
            fig, axs = plt.subplots(2)
            axs[0].axis('equal')
            axs[1].set_xlim(0, gens.shape[0])
            axs[1].set_ylim(0.1, 0.91)

            x_list = np.append(countur_coords[:, 0], countur_coords[0, 0])
            y_list = np.append(countur_coords[:, 1], countur_coords[0, 1])
            axs[0].plot(x_list, y_list, color="black")

            for line in self.windows_lines:
                axs[0].plot([line[0, 0], line[1, 0]], [line[0, 1], line[1, 1]], color="blue")

            transformed_gen = []
            first_index = 0
            gen_example = np.array(gen_example, dtype='object')
            for obj_class in gen_example:
                length = len(obj_class)

                transformed_gen.append(gen[first_index:first_index + length])
                first_index += length

            transformed_gen = np.array(transformed_gen, dtype="object")
            obj_classes, obj_centres, obj_all_centress = self.builder(transformed_gen, self.windows, self.main_points,
                                                                      self.max_diagonal)

            # delete later
            # mat_dist = self.get_minimal_dist_mat()
            # object_distant_value, result_broken_gen, dist_value, sep_val_dist = self.distant_between_classes(
            #    obj_classes, mat_dist)
            # print('Distance', object_distant_value)
            # mask, amount_inters = self.constructor_broken_gen(result_broken_gen, gen)
            # D = np.array([object_distant_value, dist_value])
            # test_attention = np.array([0.4, 0.1, 0.5])
            # result = np.sum(test_attention * D / self.coefficients)
            # print('Comparition', result, values)

            # colors = ['red', 'blue', 'yellow', 'black', 'green']
            # colors = colors[:transformed_gen.shape[0]]

            for rect_class, centres_class, color in zip(obj_classes, obj_all_centress, self.colors):
                # axs[0].scatter(centres_class[:, 0], centres_class[:, 1], s=0.2, color=color)
                for rect in rect_class:
                    x_list = np.append(rect[:, 0], rect[0, 0])
                    y_list = np.append(rect[:, 1], rect[0, 1])
                    axs[0].plot(x_list, y_list, color=color)

            for dynasty in range(dynasties_values.shape[1]):
                axs[1].plot(range(i), dynasties_values[:i, dynasty].flatten())

            fig.savefig(f"Scrins/band{i}.jpg", dpi=150, bbox_inches='tight', pad_inches=0)

            plt.close(fig)

        if type_draw == 'all':
            print('Udate Images', gens.shape[0])
            names = [f"Scrins/band{band}.jpg" for band in range(0, gens.shape[0], 2)]
            images = [Image.open(f) for f in names]
            images = [image.convert("P", palette=Image.ADAPTIVE) for image in images]
            fp_out = "image.gif"
            print('Creating GIF')

            img = images[0]
            img.save(fp=fp_out, format="GIF", append_images=images[1:], save_all=True,
                     duration=100,  # int(gif_time / gens.shape[0]),
                     loop=0)

    def save_dynasties(self, values, filename="Dynasties_values"):
        try:
            with open(filename, "a") as file_values:
                file_values.write('\n')
                for item in values:
                    file_values.write("%s\t" % item)

        except:
            with open(filename, "w") as file_values:
                for item in values:
                    file_values.write("%s\t" % item)

    def save_best_gen(self, gen, filename="Gens"):
        try:
            with open(filename, "a") as file_values:
                file_values.write('\n')
                for obj_class in gen:  # [0]:
                    for item in obj_class:
                        file_values.write("%s\t" % item)

        except:
            with open(filename, "w") as file_values:
                for obj_class in gen:
                    for item in obj_class:
                        file_values.write("%s\t" % item)


if __name__ == "__main__":
    Classes = {
        'Students': {"Amount": 80, "rectangular_x": 2, "rectangular_y": 1, 'Environment_x': 4, "Environment_y": 3,
                     "Need_lighting": 9,
                     "Classes_for_short_path": ["Printers", "Cabinets"], "Classes_ignored_intersections": ["lamp"],
                     "Classes_for_distant":
                         {"Printers_st": 6, "Aspirants": 25, "Printers_as": 25,
                          "Engineers": 25, "Printers_eng": 25, "Administration": 25, "Printers_adm": 25,
                          "Professor_2": 30, "Professor_1": 30, "Professor_3": 30,
                          "Machine_tool_1": 40, "Machine_tool_2": 40}, "Color": "goldenrod"},
        'Printers_st': {"Amount": 8, "rectangular_x": 1, "rectangular_y": 1, 'Environment_x': 3, "Environment_y": 3,
                        "Need_lighting": 7,
                        "Master": ["Students"], "Classes_ignored_intersections": ["lamp"],
                        "Classes_for_distant": {
                            "Students": 6, "Aspirants": 25, "Printers_as": 25,
                            "Engineers": 25, "Printers_eng": 25, "Administration": 25, "Printers_adm": 25,
                            "Professor_2": 30, "Professor_1": 30, "Professor_3": 30,
                            "Machine_tool_1": 40, "Machine_tool_2": 40}, "Color": "crimson"},
        'Aspirants': {"Amount": 16, "rectangular_x": 2, "rectangular_y": 1, 'Environment_x': 4, "Environment_y": 3,
                      "Need_lighting": 9,
                      "Classes_for_short_path": ["Printers", "Cabinets"], "Classes_ignored_intersections": ["lamp"],
                      "Classes_for_distant": {
                          "Students": 25, "Printers_st": 25, "Printers_as": 6,
                          "Engineers": 25, "Printers_eng": 25, "Administration": 25, "Printers_adm": 25,
                          "Professor_2": 20, "Professor_1": 20, "Professor_3": 20,
                          "Machine_tool_1": 30, "Machine_tool_2": 30}, "Color": "brown"},
        'Printers_as': {"Amount": 2, "rectangular_x": 1, "rectangular_y": 1, 'Environment_x': 3, "Environment_y": 3,
                        "Need_lighting": 6,
                        "Master": ["Aspirants"], "Classes_ignored_intersections": ["lamp"],
                        "Classes_for_distant": {
                            "Students": 25, "Printers_st": 25, "Aspirants": 6,
                            "Engineers": 25, "Printers_eng": 25, "Administration": 25, "Printers_adm": 25,
                            "Professor_2": 20, "Professor_1": 20, "Professor_3": 20,
                            "Machine_tool_1": 30, "Machine_tool_2": 30}, "Color": "yellow"},
        'Engineers': {"Amount": 8, "rectangular_x": 2, "rectangular_y": 1, 'Environment_x': 4, "Environment_y": 3,
                      "Need_lighting": 9,
                      "Classes_for_short_path": ["Printers", "Cabinets"], "Classes_ignored_intersections": ["lamp"],
                      "Classes_for_distant": {
                          "Students": 25, "Printers_st": 25, "Aspirants": 25, "Printers_as": 25,
                          "Printers_eng": 6, "Administration": 25, "Printers_adm": 25,
                          "Professor_2": 40, "Professor_1": 40, "Professor_3": 40,
                          "Machine_tool_1": 15, "Machine_tool_2": 15}, "Color": "darkviolet"},
        'Printers_eng': {"Amount": 1, "rectangular_x": 1, "rectangular_y": 1, 'Environment_x': 3, "Environment_y": 3,
                         "Need_lighting": 6,
                         "Master": ["Engineers"], "Classes_ignored_intersections": ["lamp"],
                         "Classes_for_distant": {
                             "Students": 25, "Printers_st": 25, "Aspirants": 25, "Printers_as": 25,
                             "Engineers": 6, "Administration": 25, "Printers_adm": 25,
                             "Professor_2": 40, "Professor_1": 40, "Professor_3": 40,
                             "Machine_tool_1": 15, "Machine_tool_2": 15}, "Color": "tan"},
        'Administration': {"Amount": 8, "rectangular_x": 2, "rectangular_y": 1, 'Environment_x': 4, "Environment_y": 3,
                           "Need_lighting": 9,
                           "Classes_for_short_path": ["Printers", "Cabinets"],
                           "Classes_ignored_intersections": ["lamp"],
                           "Classes_for_distant": {
                               "Students": 25, "Printers_st": 25, "Aspirants": 25, "Printers_as": 25,
                               "Engineers": 25, "Printers_eng": 25, "Printers_adm": 6,
                               "Professor_2": 25, "Professor_1": 25, "Professor_3": 25,
                               "Machine_tool_1": 40, "Machine_tool_2": 40}, "Color": "red"},
        'Printers_adm': {"Amount": 1, "rectangular_x": 1, "rectangular_y": 1, 'Environment_x': 3, "Environment_y": 3,
                         "Need_lighting": 6,
                         "Master": ["Aspirants"], "Classes_ignored_intersections": ["lamp"],
                         "Classes_for_distant": {
                             "Students": 25, "Printers_st": 25, "Aspirants": 25, "Printers_as": 25,
                             "Engineers": 25, "Printers_eng": 25, "Administration": 6,
                             "Professor_2": 25, "Professor_1": 25, "Professor_3": 25,
                             "Machine_tool_1": 40, "Machine_tool_2": 40}, "Color": "orange"},
        'Professor_1': {"Amount": 1, "rectangular_x": 2, "rectangular_y": 1, 'Environment_x': 4, "Environment_y": 3,
                        "Need_lighting": 9,
                        "Classes_for_short_path": ["Printers", "Cabinets"],
                        "Classes_ignored_intersections": ["lamp"],
                        "Classes_for_distant": {
                            "Students": 30, "Printers_st": 30, "Aspirants": 20, "Printers_as": 20,
                            "Engineers": 40, "Printers_eng": 40, "Administration": 25, "Printers_adm": 25,
                            "Professor_2": 10, "Professor_3": 10,
                            "Machine_tool_1": 40, "Machine_tool_2": 40}, "Color": "lime"},
        'Professor_2': {"Amount": 1, "rectangular_x": 2, "rectangular_y": 1, 'Environment_x': 4, "Environment_y": 3,
                        "Need_lighting": 9,
                        "Classes_for_short_path": ["Printers", "Cabinets"],
                        "Classes_ignored_intersections": ["lamp"],
                        "Classes_for_distant": {
                            "Students": 30, "Printers_st": 30, "Aspirants": 20, "Printers_as": 20,
                            "Engineers": 40, "Printers_eng": 40, "Administration": 25, "Printers_adm": 25,
                            "Professor_1": 10, "Professor_3": 10,
                            "Machine_tool_1": 40, "Machine_tool_2": 40}, "Color": "seagreen"},
        'Professor_3': {"Amount": 1, "rectangular_x": 2, "rectangular_y": 1, 'Environment_x': 4, "Environment_y": 3,
                        "Need_lighting": 9,
                        "Classes_for_short_path": ["Printers", "Cabinets"],
                        "Classes_ignored_intersections": ["lamp"],
                        "Classes_for_distant": {
                            "Students": 30, "Printers_st": 30, "Aspirants": 20, "Printers_as": 20,
                            "Engineers": 40, "Printers_eng": 40, "Administration": 25, "Printers_adm": 25,
                            "Professor_2": 10, "Professor_1": 10,
                            "Machine_tool_1": 40, "Machine_tool_2": 40}, "Color": "green"},

        # 'Cabinets': {"Amount": 4, "rectangular_x": 0.5, "rectangular_y": 2, 'Environment_x': 1.5, "Environment_y": 1,
        #              "Need_lighting": 6,
        #              "Classes_for_short_path": ["Workplace"], "Classes_ignored_intersections": ["lamp"],
        #              "Classes_for_distant": {"Machine_tool": 5}},
        # 'lamp': {"Amount": 60, "rectangular_x": 0.4, "rectangular_y": 0.4, 'Environment_x': 0.4, "Environment_y": 0.4,
        #          "Need_lighting": "reverse",
        #          "Classes_for_short_path": [None],
        #          "Classes_ignored_intersections": ["Workplace", "Printers", "Cabinets", "Machine_tool"],
        #          "Classes_for_distant": {None}},
        'Machine_tool_1': {"Amount": 30, "rectangular_x": 3, "rectangular_y": 4, 'Environment_x': 8, "Environment_y": 8,
                           "Need_lighting": 1,
                           "Classes_for_short_path": ["Printers", "Cabinets"],
                           "Classes_ignored_intersections": ["lamp"],
                           "Classes_for_distant": {
                               "Students": 40, "Printers_st": 40, "Aspirants": 30, "Printers_as": 30,
                               "Engineers": 15, "Printers_eng": 15, "Administration": 40, "Printers_adm": 40,
                               "Professor_2": 40, "Professor_1": 40, "Professor_3": 40,
                               "Machine_tool_2": 30}, "Color": "cornflowerblue"},
        'Machine_tool_2': {"Amount": 30, "rectangular_x": 3, "rectangular_y": 4, 'Environment_x': 8, "Environment_y": 8,
                           "Need_lighting": 1,
                           "Classes_for_short_path": ["Printers", "Cabinets"],
                           "Classes_ignored_intersections": ["lamp"],
                           "Classes_for_distant": {
                               "Students": 40, "Printers_st": 40, "Aspirants": 30, "Printers_as": 30,
                               "Engineers": 15, "Printers_eng": 15, "Administration": 40, "Printers_adm": 40,
                               "Professor_2": 40, "Professor_1": 40, "Professor_3": 40,
                               "Machine_tool_1": 30}, "Color": "mediumblue"}
    }

    Opt = Optimizer(Classes)

# {"Machine_tool_1", "Machine_tool_2", "Professor_3",
# "Professor_2", "Professor_1", "Printers_adm",
# "Administration", "Printers_eng", "Engineers",
# "Printers_as", "Aspirants", "Printers_st", "Students"}

# "Students", "Printers_st", "Aspirants", "Printers_as",
# "Engineers", "Printers_eng", "Administration", "Printers_adm",
# "Professor_2", "Professor_1", "Professor_3",
# "Machine_tool_1", "Machine_tool_2"
