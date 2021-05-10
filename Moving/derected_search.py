import numpy as np
import random
import time

from pathos.multiprocessing import ProcessPool

__evolsearch_process_pool = None


class DirSearch:
    def __init__(self, evol_params):
        """
        Initialize evolutionary search
        ARGS:
        evol_params: dict
            required keys -
                pop_size: int - population size,
                calibration_function: function - a user-defined function that takes a genotype as args and returns
                                        a list of max values of penalty function,
                fitness_function: function - a user-defined function that takes a genotype
                                    as arg and returns a float fitness value
                elitist_fraction: int - parents number for every dynasty
                bounds: 2d-array - list that contains slots for each variables
                probability_mask: array - probability distribution within bounds
                num_branches: int - dynasty number
                num_processes: int -  pool size for multiprocessing.pool.Pool - defaults to os.cpu_count()
        """
        # check for required keys
        required_keys = ['calibration_function', 'fitness_function', "bounds",
                         "probability_mask"]
        for key in required_keys:
            if key not in evol_params.keys():
                raise Exception('Argument evol_params does not contain the following required key: ' + key)

        # checked for all required keys
        self.fitness_function = evol_params['fitness_function']
        self.calibration_function = evol_params['calibration_function']
        self.probability_mask = evol_params['probability_mask']
        self.bounds = evol_params['bounds']
        self.num_ind = evol_params['number_individuals']
        self.first_pop = evol_params['first_pop']
        self.coefficients = evol_params['coefficients']
        self.pop = self.first_pop.copy()
        self.pop_old = self.first_pop.copy()

        # create other required data
        self.num_processes = evol_params.get('num_processes', None)
        self.dynasties_best_values = []
        self.best_gen = None

        # creating the global process pool to be used across all generations
        global __evolsearch_process_pool
        __evolsearch_process_pool = ProcessPool(self.num_processes)
        time.sleep(0.5)

        global __evolsearch_process_pool

        # estimate fitness using multiprocessing pool
        if __evolsearch_process_pool:
            # pool exists
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(self.evaluate_fitness, np.arange(self.num_ind)))
        else:
            # re-create pool
            __evolsearch_process_pool = Pool(self.num_processes)
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(self.evaluate_fitness, np.arange(self.num_ind)))

        # returned list contains mask of broken genes and fitness value, so that is separated
        self.fitness = np.array(self.fitness, dtype="object")  # transforming to numpy array

        self.dynasties_best_values = self.fitness[:, 0]  # extraction of mask for mutation
        self.fitness_old = self.fitness[:, 1]  # remaining values are fitness values list


        # self.dynasties_best_values = np.sum(self.fitness_old, axis=1)
        # save best gen
        best_index = np.argsort(self.dynasties_best_values * (-1))[-1:]
        # self.dynasties_best_values.append(self.pop[best_index])
        self.best_gen = self.pop_old[best_index]

    def gen_genartion(self):
        """
        The function creates gen according probability distribution
        :return: list - generated gen that contains chromosomes
        """
        big_gen = []  # void list for appending chromosomes
        for net_mask, net_doubt in zip(self.probability_mask, self.bounds):  # cycle over chromosomes
            net_gen = []  # chromosome - void list for appending variables
            for probability, value_slot in zip(net_mask, net_doubt):  # cycle over variables

                #  appending variable from slot with offset distribution
                if type(probability) is float or type(probability) is int:
                    net_gen.append(random.triangular(value_slot[0], value_slot[1],
                                                     value_slot[0] + (value_slot[1] - value_slot[0]) * probability))

                #  appending variable from slot with equally distribution
                elif type(probability) is str:
                    net_gen.append(random.uniform(value_slot[0], value_slot[1]))

                #  appending variable from list for choice
                else:
                    net_gen.append(random.choices(value_slot, cum_weights=probability, k=1)[0])

            big_gen.append(net_gen)  # appending created chromosome

        return big_gen

    def evaluate_fitness(self, individual_index):
        """
        Function returns value of object function
        :param individual_index: index of population that is calculated
        :return: value of fitness function
        """
        if self.optional_args:  # for the case with kwargs
            if len(self.optional_args) == 1:
                return self.fitness_function(self.pop[individual_index, :], self.optional_args[0])
            else:
                return self.fitness_function(self.pop[individual_index, :], self.optional_args[individual_index])
        else:
            return self.fitness_function(self.pop[individual_index, :], self.coefficients)

    def step_generation(self):
        """
        evaluate fitness of pop, and create new pop after crossing and mutation
        """
        # create initial data of evolutionary search
        self.pop = np.array([self.gen_genartion() for i in range(self.first_pop.shape[0])], dtype='object')

        global __evolsearch_process_pool

        # estimate fitness using multiprocessing pool
        if __evolsearch_process_pool:
            # pool exists
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(self.evaluate_fitness, np.arange(self.num_ind)))
        else:
            # re-create pool
            __evolsearch_process_pool = Pool(self.num_processes)
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(self.evaluate_fitness, np.arange(self.num_ind)))

        # returned list contains mask of broken genes and fitness value, so that is separated
        self.fitness = np.array(self.fitness, dtype="object")  # transforming to numpy array

        self.fitness_values = self.fitness[:, 0]  # extraction of mask for mutation
        self.fitness = self.fitness[:, 1]  # remaining values are fitness values list

        # balancing !!!!!!!!!!!!!!!!!!!!!!!!!
        best_pop = []

        for gen, gen_old, gen_fit, gen_fit_old in zip(self.pop, self.pop_old, self.fitness, self.fitness_old):
            best_gen = []
            for chr, chr_old, chr_fit, chr_fit_old in zip(gen, gen_old, gen_fit, gen_fit_old):
                locations_angles_amount = len(chr[9:])

                chr_ind = chr[9:int(9 + locations_angles_amount / 2)]
                chr_angl = chr[int(9 + locations_angles_amount / 2):]

                chr_old_ind = chr_old[9:int(9 + locations_angles_amount / 2)]
                chr_old_angl = chr_old[int(9 + locations_angles_amount / 2):]

                mask_superiority = (chr_fit < chr_fit_old) * 1
                anti_mask_superiority = (mask_superiority - 1) * (-1)

                best_chr_ind = chr_ind * mask_superiority + chr_old_ind * anti_mask_superiority
                best_chr_angl = chr_angl * mask_superiority + chr_old_angl * anti_mask_superiority

                best_chr = chr.copy()
                best_chr[9:int(9 + locations_angles_amount / 2)] = best_chr_ind
                best_chr[int(9 + locations_angles_amount / 2):] = best_chr_angl

                best_gen.append(best_chr)

            best_pop.append(best_gen)

        self.pop = best_pop

        global __evolsearch_process_pool

        # estimate fitness using multiprocessing pool
        if __evolsearch_process_pool:
            # pool exists
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(self.evaluate_fitness, np.arange(self.num_ind)))
        else:
            # re-create pool
            __evolsearch_process_pool = Pool(self.num_processes)
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(self.evaluate_fitness, np.arange(self.num_ind)))

        # returned list contains mask of broken genes and fitness value, so that is separated
        self.fitness = np.array(self.fitness, dtype="object")  # transforming to numpy array

        self.dynasties_best_values = self.fitness[:, 0]  # extraction of mask for mutation
        self.fitness_old = self.fitness[:, 1]  # remaining values are fitness values list

        self.pop_old = self.pop

        # self.dynasties_best_values = np.sum(self.fitness_old, axis=1)
        # save best gen
        best_index = np.argsort(self.dynasties_best_values * (-1))[-1:]
        # self.dynasties_best_values.append(self.pop[best_index])
        self.best_gen = self.pop_old[best_index]

    def get_dynasties_best_value(self):
        return self.dynasties_best_values

    def get_best_individual(self):
        '''
        returns 1D array of the genotype that has max fitness
        '''
        return self.best_gen

    def get_best_individual_fitness(self):
        '''
        return the fitness value of the best individual
        '''
        return np.min(self.dynasties_best_values)

    def get_mean_fitness(self):
        '''
        returns the mean fitness of the population
        '''
        return np.mean(self.dynasties_best_values)


if __name__ == "__main__":
    def fitness_function(individual):
        '''
        sample fitness function
        '''
        return np.mean(individual)


    # defining the parameters for the evolutionary search
    evol_params = {
        'num_processes': 4,  # (optional) number of proccesses for multiprocessing.Pool
        'pop_size': 100,  # population size
        'genotype_size': 10,  # dimensionality of solution
        'fitness_function': fitness_function,  # custom function defined to evaluate fitness of a solution
        'elitist_fraction': 0.04,  # fraction of population retained as is between generations
        'mutation_variance': 0.05  # mutation noise added to offspring.
    }

    # create evolutionary search object
    es = EvolSearch(evol_params)

    '''OPTION 1
    # execute the search for 100 generations
    num_gens = 100
    es.execute_search(num_gens)
    '''

    '''OPTION 2'''
    # keep searching till a stopping condition is reached
    num_gen = 0
    max_num_gens = 100
    desired_fitness = 0.75
    while es.get_best_individual_fitness() < desired_fitness and num_gen < max_num_gens:
        print('Gen #' + str(num_gen) + ' Best Fitness = ' + str(es.get_best_individual_fitness()))
        es.step_generation()
        num_gen += 1

    # print results
    print('Max fitness of population = ', es.get_best_individual_fitness())
    print('Best individual in population = ', es.get_best_individual())
