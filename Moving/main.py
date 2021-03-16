import numpy as np
import random
import matplotlib.pyplot as plt
from evolutionary_search import EvolSearch
from PIL import Image


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

        #  initialization parameters of genes: bounds (limits or list of variants, which variable can be),
        #                                      probability_mask (density of probability in bounds)
        self.bounds = []
        self.probability_mask = []
        self.bounds_constructor()
        self.probability_mask_constructor()

        #  that parameters are used for definition case plan. Windows are ares for classes where it could be.
        #  Main_points are points for each class. That points point are used for sorting cells of grid by distant.
        #  In further these parameters should be obtain by other class.
        self.windows = np.array(
            [[[5, 100], [100, 5]], [[5, 100], [100, 5]], [[5, 100], [100, 5]], [[100, 40], [165, 5]],
             [[100, 75], [165, 45]]])
        self.main_points = np.array([[5, 5], [45, 55], [95, 35], [165, 40], [135, 45]])

        #  That part of code is used for testing object function without GA.

        # gen = self.gen_constructor()
        # obj_classes = self.builder(gen, self.windows, self.main_points, 150)
        # mat_dist = self.get_minimal_dist_mat()
        # object_distant_value, result_broken_gen = self.distant_between_classes(obj_classes, mat_dist)
        # self.constructor_broken_gen(result_broken_gen, gen)

        # gen = self.gen_constructor()
        # coefficients = self.calibration_function(gen)
        # new_gen = self.gen_constructor()
        # result = self.fitness_function(new_gen, coefficients)

        #  there is using that code row because classes don't have similar lengths.
        self.bounds = np.array(self.bounds, dtype="object")
        self.probability_mask = np.array(self.probability_mask, dtype="object")

        evol_params = {
            'num_processes': 4,  # (optional) number of processes for multiprocessing.Pool
            'pop_size': 200,  # population size
            'fitness_function': self.fitness_function,  # custom function defined to evaluate fitness of a solution
            'calibration_function': self.calibration_function,
            'elitist_fraction': 2,  # fraction of population retained as is between generations
            'bounds': self.bounds,  # limits or list of variants, which variable can be
            'probability_mask': self.probability_mask,  # density of probability in bounds
            'num_branches': 3  # Amount of dynasties
        }

        es = EvolSearch(evol_params)  # Creating class for evolution search

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
        print('Max fitness of population = ', es.get_best_individual_fitness())
        print('Best individual in population = ', es.get_best_individual())
        self.coefficients = es.get_coefficients()  # Getting found coefficients for recount getting plot result

        self.artist("test_gens.txt", 'test_values.txt', es.get_best_individual())  # Plot rendering and saving as GIF

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
            minimal_distances.append(min_dist_one_axies)

        return np.array(minimal_distances)

    def bounds_constructor(self):
        for kind in self.Classes:
            layer_parameters = [[self.Classes[kind]['Environment_x'], self.Classes[kind]['Environment_x'] * 2],
                                [self.Classes[kind]['Environment_y'], self.Classes[kind]['Environment_y'] * 2],
                                [0, self.Classes[kind]['Environment_x']], [0, self.Classes[kind]['Environment_y']],
                                [1, 10], [1, 10],
                                [self.Classes[kind]['Environment_x'] * 1.5, self.Classes[kind]['Environment_x'] * 3],
                                [self.Classes[kind]['Environment_y'] * 1.5, self.Classes[kind]['Environment_y'] * 3],
                                [0, 45, 90], [0, 1]]
            for drop in range(0, self.Classes[kind]["Amount"]):
                layer_parameters.append([0, 45, 90])
            self.bounds.append(layer_parameters)

    def probability_mask_constructor(self):
        for kind in self.Classes:
            layer_parameters = [0, 0,  # P_x, P_y
                                0, 0,  # Offset_x, Offset_y from main point
                                0.5, 0.5,  # N_x, N_y
                                'Equally_distributed', 'Equally_distributed',  # offsets from points group
                                # 'Equally_distributed', 'Equally_distributed',
                                [0.4, 0.2, 0.4],  # grid angle
                                0]  #
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

    def test_function(self, gen):
        obj_classes = self.builder(gen, self.windows, self.main_points, 150)
        mat_dist = self.get_minimal_dist_mat()
        object_distant_value, result_broken_gen, dist_value = self.distant_between_classes(obj_classes, mat_dist)
        # print('Distance', object_distant_value)
        mask, amount_inters = self.constructor_broken_gen(result_broken_gen, gen)

        return np.array([object_distant_value, dist_value]), mask

    def calibration_function(self, x_vector):
        x_vector = np.array(x_vector, dtype="object")
        # print(x_vector)
        penalties, mask = self.test_function(x_vector)
        return penalties

    def fitness_function(self, gen, koef):
        x_vector = np.array(gen, dtype="object")
        test_attention = np.array([0.9, 0.1])
        D, mask = self.test_function(x_vector)
        # print(D)
        mask = []
        for grid in gen:
            mask.append(np.ones_like(np.array(grid)))

        result = np.sum(test_attention * D / koef)
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
            # print("Deg", deg)
            bases_grid = self.ratation(bases_grid, deg)
        bases_grid = bases_grid + main_point
        bases_grid = bases_grid + offset_grid
        # print(bases_grid.shape)
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
            if angle == 0:
                rotated_rects.append(self.ratation(rect, angle))
            else:
                rotated_rects.append(rect)

        return np.array(rotated_rects)

    def sorting_drops_bydistant(self, drops, main_point):
        drops = np.unique(drops, axis=0)
        distant = ((drops[:, 0] - main_point[0]) ** 2 + (drops[:, 1] - main_point[1]) ** 2) ** 0.5
        return drops[np.argsort(distant)]

    def get_centre_pints_opinion(self, drops, n, index):
        n_drops = drops.shape[0]
        offset = int(index * (n_drops - n))
        return drops[offset:offset + n, :]

    def get_centre_points_option_2(self, drops, n, index):
        binary_legth = '1' * n + '0' * (drops.shape[0] - n)
        max_dec_number = int(binary_legth, 2)
        dec_numbers = np.arange(int('1' * n, 2), max_dec_number, int((max_dec_number - int('1' * n, 2)) / 1000))
        binary_list = ((dec_numbers[:, None] & (1 << np.arange(drops.shape[0]))) > 0).astype(int)
        binary_list = binary_list[0, :]
        binary_sum = np.sum(binary_list, axis=0)
        all_options = binary_list[binary_sum == n]
        option_index = int(index * drops.shape[0])
        selected_option = all_options[option_index]
        return drops[selected_option == 1]

    def get_centre_points_option_3(self, drops, n, index):
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
        option_index = int(index * all_options.shape[0])
        selected_option = all_options[option_index]
        returned_drops = drops[selected_option == 1]
        return returned_drops

    def get_centre_points_option_4(self, drops, indexes):
        indexes = indexes * drops.shape[0]
        all_indexes = range(drops.shape[0])
        selected_indexes = []
        for index in indexes:
            founded_index = closest(all_indexes, index)
            selected_indexes.append(founded_index)
            all_indexes.remove(founded_index)

        return drops[selected_indexes]

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
            # print("Grid_angle", grid_angle)
            objects_angles = chromosome[10:]
            amount = len(objects_angles)
            previous_grid = self.get_cells_grids(main_point, [offset_x, offset_y], dioganal, int(n_x), p_x, off2_x,
                                                 int(n_y), p_x, off2_y, grid_angle)
            grid = self.cut_by_rectagular(previous_grid, window[0], window[1])
            centre_points = self.sorting_drops_bydistant(grid, main_point)
            #  centre_points = self.get_centre_pints_opinion(centre_points, amount, layout_ind)
            centre_points = self.get_centre_points_option_3(centre_points, amount, layout_ind)
            rectangulars = self.get_objects_angle(amount, p_x, p_y, objects_angles)
            rects = self.locate_objects(rectangulars, centre_points)

            # plt.axis('equal')
            # plt.plot(grid[:, 0], grid[:, 1], 'o')
            # for rect in rects:
            #     x_list = np.append(rect[:, 0], rect[0, 0])
            #     y_list = np.append(rect[:, 1], rect[0, 1])
            #     plt.plot(x_list, y_list, color=color)

            object_class.append(rects)

        #  plt.show()
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

        distances = (x_distances ** 2 + y_distances ** 2) ** 0.5

        return distances

    def distant_between_classes(self, rects_classes, minimal_distances, flag_debug=False):
        # print(minimal_distances)
        object_distant = 0
        number_classes = rects_classes.shape[0]
        broken_gens = []
        if flag_debug:
            print(minimal_distances)
        for i in range(number_classes):
            broken_gen = []
            for j in range(number_classes):
                if i != j:
                    distances = self.distant_between_two_classes(rects_classes[i], rects_classes[j])
                    distances_mistake = (distances < minimal_distances[i, j]) * 1
                    distances_mistake = np.sum(distances_mistake, axis=2)
                    distances_mistake = np.sum(distances_mistake, axis=0)
                    broken_gen.append(distances_mistake)
                    object_distant += np.sum(distances)
            broken_gens.append(broken_gen)

        if flag_debug:
            print('mistake_matrix', broken_gens)

        if number_classes >= 3:
            result_broken_gen = np.sum(broken_gens, axis=1)
        else:
            result_broken_gen = broken_gens

        object_distant_value = 0
        for chromasome_index in range(number_classes):
            object_distant_value += np.sum(result_broken_gen[chromasome_index])
            result_broken_gen[chromasome_index] = (result_broken_gen[chromasome_index] >= 1) * 1

        return object_distant_value, result_broken_gen, object_distant

    def constructor_broken_gen(self, parts_gen, example_gen):
        #  temprorary
        amount_intersections = 0
        mask_gen = []
        for chromosome, broken_part in zip(example_gen, parts_gen):
            void_gen = np.zeros_like(chromosome)
            void_gen[void_gen.shape[0] - broken_part.shape[0]:] = broken_part
            amount_intersections += np.sum(broken_part)
            if np.sum(broken_part) >= 0.5 * broken_part.shape[0]:
                void_gen[9] = 1
            mask_gen.append(void_gen)

        return mask_gen, amount_intersections

    def artist(self, filename_gens, filename_values, gen_example, gif_time=100, type_draw="all"):
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

            fig, axs = plt.subplots(2)
            axs[0].axis('equal')
            axs[1].set_xlim(0, gens.shape[0])
            axs[1].set_ylim(0, 0.2)

            transformed_gen = []
            first_index = 0
            for obj_class in gen_example[0]:
                length = obj_class.shape[0]
                transformed_gen.append(gen[first_index:first_index + length])
                first_index += length

            transformed_gen = np.array(transformed_gen, dtype="object")
            obj_classes = self.builder(transformed_gen, self.windows, self.main_points, 150)

            # delete later
            mat_dist = self.get_minimal_dist_mat()
            object_distant_value, result_broken_gen, dist_value = self.distant_between_classes(obj_classes, mat_dist,
                                                                                               True)
            # print('Distance', object_distant_value)
            # mask, amount_inters = self.constructor_broken_gen(result_broken_gen, gen)
            D = np.array([object_distant_value, dist_value])
            test_attention = np.array([0.9, 0.1])
            result = np.sum(test_attention * D / self.koef)
            # print('Comparition', result, values)

            colors = ['red', 'blue', 'yellow', 'black', 'green']
            colors = colors[:transformed_gen.shape[0]]
            for rect_class, color in zip(obj_classes, colors):
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
            names = [f"Scrins/band{band}.jpg" for band in range(0, gens.shape[0])]
            images = [Image.open(f) for f in names]
            images = [image.convert("P", palette=Image.ADAPTIVE) for image in images]
            fp_out = "image.gif"
            print('Creating GIF')

            img = images[0]
            img.save(fp=fp_out, format="GIF", append_images=images[1:], save_all=True,
                     duration=int(gif_time / gens.shape[0]),
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
                for obj_class in gen[0]:
                    for item in obj_class:
                        file_values.write("%s\t" % item)

        except:
            with open(filename, "w") as file_values:
                for obj_class in gen:
                    for item in obj_class:
                        file_values.write("%s\t" % item)


if __name__ == "__main__":
    Classes = {
        'Workplace': {"Amount": 12, "rectangular_x": 2, "rectangular_y": 1, 'Environment_x': 4, "Environment_y": 3,
                      "Need_lighting": 9,
                      "Classes_for_short_path": ["Printers", "Cabinets"], "Classes_ignored_intersections": ["lamp"],
                      "Classes_for_distant": {"Machine_tool": 40, "Printers": 20}},
        'Printers': {"Amount": 3, "rectangular_x": 1, "rectangular_y": 1, 'Environment_x': 3, "Environment_y": 3,
                     "Need_lighting": 9,
                     "Classes_for_short_path": ["Workplace"], "Classes_ignored_intersections": ["lamp"],
                     "Classes_for_distant": {"Machine_tool": 50, "Workplace": 20}},
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
                         "Classes_for_distant": {"Workplace": 40, "Printers": 50, "Cabinets": 5}}
    }

    Opt = Optimizer(Classes)
