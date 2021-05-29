import numpy as np
import random
import copy
import time

from pathos.multiprocessing import ProcessPool

__evolsearch_process_pool = None


class MapSearch:
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
        required_keys = ['fitness_function', 'function_get_oth_indexes', 'first_gen', 'coefficients']
        for key in required_keys:
            if key not in evol_params.keys():
                raise Exception('Argument evol_params does not contain the following required key: ' + key)

        # checked for all required keys
        self.fitness_function = evol_params['fitness_function']
        self.function_get_other = evol_params['function_get_oth_indexes']
        self.best_gen = evol_params['first_gen']
        self.coefficients = evol_params['coefficients']

        #self.pop = np.squeeze(self.first_pop.copy())
        #self.pop_old = np.squeeze(self.first_pop.copy())

        # create other required data
        self.num_processes = evol_params.get('num_processes', None)
        self.dynasties_best_values = []
        #self.best_gen = None
        self.class_index = 0
        self.optional_args = None

        # creating the global process pool to be used across all generations
        global __evolsearch_process_pool
        __evolsearch_process_pool = ProcessPool(self.num_processes)
        time.sleep(0.5)

    def set_class_for_opt(self, class_index):
        self.class_index = class_index

    def set_gen_for_opt(self, gen):
        self.gen = gen

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
        values = self.fitness_function(self.best_gen, self.coefficients)
        res = values[0]
        res_sep = values[1]

        worse_ind = np.argsort(res_sep[self.class_index] )[-1:] #!!!!!!!!!!!!!!!!!!!
        other_indexes = self.function_get_other(self.best_gen, self.class_index, worse_ind)

        amount_other_indexes = other_indexes.shape[0]

        best_gen = self.best_gen[np.newaxis, :]
        pop_bufer = np.repeat(best_gen, amount_other_indexes, axis=0)

        new_pop = []
        for gen, rep, i in zip(pop_bufer, other_indexes, range(amount_other_indexes)):
            #chr_chg = gen[self.class_index]
            #chr_chg[9 + int(worse_ind)] = rep
            #pop_bufer[i, self.class_index]=chr_chg
            gen_chg = copy.deepcopy(self.best_gen)
            #gen_chg = gen_chg[0]
            gen_chg[self.class_index][9+int(worse_ind)] = rep
            new_pop.append(gen_chg)

        new_pop.append(self.best_gen)

        self.pop = np.array(new_pop, dtype="object")

        global __evolsearch_process_pool

        # estimate fitness using multiprocessing pool
        if __evolsearch_process_pool:
            # pool exists
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(self.evaluate_fitness, np.arange(amount_other_indexes+1)))
        else:
            # re-create pool
            __evolsearch_process_pool = Pool(self.num_processes)
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(self.evaluate_fitness, np.arange(amount_other_indexes+1)))

        # returned list contains mask of broken genes and fitness value, so that is separated
        self.fitness = np.array(self.fitness, dtype="object")  # transforming to numpy array

        self.fitness_values = self.fitness[:, 0]  # extraction of mask for mutation
        self.fitness = self.fitness[:, 1]  # remaining values are fitness values list

        list_values_by_place = []

        # for gen in self.fitness:
            # list_values_by_place.append(gen[self.class_index][worse_ind])

        best_index = np.argsort(self.fitness_values * (-1))[-1:]
        self.best_gen = self.pop[best_index[0]]

        self.dynasties_best_values = [1, 1, 1, 1, 1, 1]*self.fitness_values[best_index]  # extraction of mask for mutation


        # self.dynasties_best_values = np.sum(self.fitness_old, axis=1)
        # save best gen
        # self.dynasties_best_values.append(self.pop[best_index])

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
