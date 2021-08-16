import numpy as np
import random
import time

from pathos.multiprocessing import ProcessPool

__evolsearch_process_pool = None


class EvolSearch:
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
        required_keys = ['pop_size', 'calibration_function', 'fitness_function', 'elitist_fraction',
                         "bounds", "probability_mask", "num_branches"]
        for key in required_keys:
            if key not in evol_params.keys():
                raise Exception('Argument evol_params does not contain the following required key: ' + key)

        # checked for all required keys
        self.pop_size = evol_params['pop_size']
        self.fitness_function = evol_params['fitness_function']
        self.calibration_function = evol_params['calibration_function']
        self.elitist_fraction = evol_params['elitist_fraction']
        self.probability_mask = evol_params['probability_mask']
        self.bounds = evol_params['bounds']
        self.num_branches = evol_params['num_branches']

        # create other required data
        self.coefficients = None
        self.num_processes = evol_params.get('num_processes', None)
        self.dynasties_best_values = []
        self.best_in_dynasties = []
        self.sep_fitness = np.array([])
        self.best_sep_fitness = None
        self.best_gen = None
        self.direct_evolution = False
        self.mutation_coefficient = 0.1

        # check for fitness function kwargs
        if 'fitness_args' in evol_params.keys():
            optional_args = evol_params['fitness_args']
            assert len(optional_args) == 1 or len(optional_args) == self.pop_size, \
                "fitness args should be length 1 or pop_size."
            self.optional_args = optional_args
        else:
            self.optional_args = None

        # creating the global process pool to be used across all generations
        global __evolsearch_process_pool
        __evolsearch_process_pool = ProcessPool(self.num_processes)
        time.sleep(0.5)

        # create initial data of evolutionary search
        self.pop = np.array([self.gen_genartion() for i in range(self.pop_size)], dtype='object')
        self.fitness = np.zeros(self.pop_size)

        # auto scaling
        self.colibration()

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

    def colibarate_fitnes(self, individual_index):
        """
        Function returns list of penalty functions values for further auto scaling
        :param individual_index: index of population that is calculated
        :return: list of penalty functions values
        """
        if self.optional_args:  # for the case with kwargs
            if len(self.optional_args) == 1:
                return self.fitness_function(self.pop[individual_index, :], self.optional_args[0],
                                             self.direct_evolution)
            else:
                return self.fitness_function(self.pop[individual_index, :], self.optional_args[individual_index],
                                             self.direct_evolution)
        else:
            return self.calibration_function(self.pop[individual_index, :])

    def evaluate_fitness(self, individual_index):
        """
        Function returns value of object function
        :param individual_index: index of population that is calculated
        :return: value of fitness function
        """
        if self.optional_args:  # for the case with kwargs
            if len(self.optional_args) == 1:
                return self.fitness_function(self.pop[individual_index, :], self.optional_args[0],
                                             self.direct_evolution)
            else:
                return self.fitness_function(self.pop[individual_index, :], self.optional_args[individual_index],
                                             self.direct_evolution)
        else:
            return self.fitness_function(self.pop[individual_index, :], self.coefficients, self.direct_evolution)

    def mutation(self, parents, n_copies, mask, cross):
        """
        Function expansions list of parents and changes some variables
        :param parents: list of parents for copying
        :param n_copies: size of result list (number of copies)
        :param mask: mask for mutation - if mask is 1, there will mutation probability
        :param cross: 1 for mutation after crossing (as a multi-operators),
        :return: mutated copies of parents that have size n_copies
        """

        # representation list to numpy array
        mask = np.array(mask, dtype='object')
        parents = np.array(parents, dtype='object')

        if not cross:  # in case if mutation is independent operators
            parents_copies = np.tile(parents, [n_copies, 1])  # expansion area by copying parents
            parents = parents_copies[:n_copies, :]  # cutting excess copies
            mask_copies = np.tile(mask, n_copies)  # expansion probability mask by copying mask parents
            mask = mask_copies[:n_copies]  # cutting excess copies

        # create random matrix for carryover to mutation places
        random_mat = np.array([self.gen_genartion() for i in range(parents.shape[0])], dtype='object')

        # void list for mutated copies
        result_pop = []

        for gen_copy, random_mat_gen, mask_gen in zip(parents, random_mat, mask):  # cycle by copies list
            gen = []  # for new (mutated) gen
            for hrom_gen, hrom_random, hrom_mask in zip(gen_copy, random_mat_gen, mask_gen):  # cycle be chromosomes
                #  making 1-d array, 1 - there is mutation in variable, 0 - there isn't mutation
                mut_mask = np.random.choice(2, len(hrom_gen), p=[1 - self.mutation_coefficient,
                                                                 self.mutation_coefficient]) * hrom_mask  # making mutation more rare
                negative_mut_mask = (mut_mask - 1) * (-1)  # opposite matrix
                # from initial chromosome variable is replaced by variable from random chromosome
                hrom_gen = hrom_gen * negative_mut_mask + hrom_random * mut_mask
                gen.append(hrom_gen)  # fulling gen by chromosomes
            result_pop.append(gen)  # saving new gen

        # transformation new pop list as numpy array
        result_pop = np.array(result_pop, dtype='object')
        return result_pop

    def crosover(self, parents, req_pop):
        """
        Function creates new genes from parents by mixing their chromosomes.
        :param parents: initial genes
        :param req_pop: required quantity of children
        :return: numpy array 1-d, where children genes are objects
        """
        np.random.shuffle(parents)  # mixing of parents to get random children
        children = []  # void list for fulling by children
        while len(children) < req_pop:  # cycle while required quantity of children isn't reached
            i = 0  # create counter not to have same parents
            for parent_1 in parents:  # choosing the first parent
                j = 0  # create the second counter not to have same parents
                for parent_2 in parents:  # choosing the second parent
                    if random.randint(0, 1) and i != j:  # cross will occur randomly and if parents are not same
                        # chromosomes which will be crossed are selected according random list
                        mask = np.random.choice(2, parent_1.shape[0],
                                                p=[1 - self.mutation_coefficient, self.mutation_coefficient])
                        negativ_mask = (mask - 1) * (-1)  # opposite matrix
                        try:  # for some reason, some genes don't cross
                            # from the first parent some chromosomes are removed and chromosomes from the second parents
                            # are added to that places
                            child = parent_1 * negativ_mask + parent_2 * mask
                            children.append(child)  # new child are added to list of children
                        except:
                            pass
                    j += 1
                i += 1

        children = np.array(children, dtype='object')[:req_pop]  # cutting excess genes
        return children

    def step_generation(self):
        """
        evaluate fitness of pop, and create new pop after crossing and mutation
        """
        global __evolsearch_process_pool

        # estimate fitness using multiprocessing pool
        if __evolsearch_process_pool:
            # pool exists
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(self.evaluate_fitness, np.arange(self.pop_size)))
        else:
            # re-create pool
            __evolsearch_process_pool = Pool(self.num_processes)
            self.fitness = np.asarray(
                __evolsearch_process_pool.map(self.evaluate_fitness, np.arange(self.pop_size)))

        # returned list contains mask of broken genes and fitness value, so that is separated
        self.fitness = np.array(self.fitness, dtype="object")  # transforming to numpy array
        self.mask_broken = self.fitness[:, 1]  # extraction of mask for mutation
        self.sep_fitness = self.fitness[:, 2]
        self.fitness = self.fitness[:, 0]  # remaining values are fitness values list

        # separate to dynasties
        self.dynasties_pop = np.array_split(self.pop, self.num_branches)
        self.dynasties_fitness = np.array_split(self.fitness, self.num_branches)
        self.dynasty_mask_broken = np.array_split(self.mask_broken, self.num_branches)

        # create bufer list for crossing between different dynasties
        self.fund_best_parents = []

        new_pop = []  # void list for new population
        number_dynasty = 0  # counter of dynasties
        self.dynasties_best_values = []
        self.best_in_dynasties = []

        # save best gen
        best_index = np.argsort(self.fitness * (-1))[-1:]
        # self.dynasties_best_values.append(self.pop[best_index])
        self.best_gen = self.pop[best_index]
        self.best_sep_fitness = self.sep_fitness[best_index]

        # work over each dynasty separately
        for pop, fitness, mask_broken in zip(self.dynasties_pop, self.dynasties_fitness, self.dynasty_mask_broken):
            print('Dynasty: ', number_dynasty, np.min(fitness))
            number_dynasty += 1  # in that case counter needs only for printing so is grossing here

            # initial of parents
            parents_indexes = np.argsort(fitness * (-1))[-self.elitist_fraction:]  # select parents
            bufer_pop = pop[parents_indexes, :]
            self.best_in_dynasties.append(pop[np.argsort(fitness * (-1))[-1:]])

            #  saving best value
            self.dynasties_best_values.append(np.min(fitness))

            _mask = mask_broken[parents_indexes]  # define mask for mutation for parents

            # save parents for crossing with other dynasty
            if len(self.fund_best_parents) == 0:
                self.fund_best_parents = bufer_pop
            else:
                fund_best_parents = np.append(self.fund_best_parents, bufer_pop, axis=0)

            # calculate quantity of mutation genes, crossed genes, and combination returned genes
            dynasty_size = pop.shape[0]
            n_parents = bufer_pop.shape[0]  # number of parents
            n_mutation = int((dynasty_size - n_parents) / 3)  # number copies for mutation
            n_cross = int((dynasty_size - n_parents) / 3)  # number copies for crossover
            n_mutation_cross = dynasty_size - n_parents - n_mutation - n_cross  # number for cross and mutation

            # from time to time add gene from bufer to get crossing with previous dynasty
            if random.randint(0, 1) and self.fund_best_parents.shape[0] > 0:
                index_foundling = np.random.choice(np.arange(self.fund_best_parents.shape[0]))
                parents = np.vstack((bufer_pop, [self.fund_best_parents[index_foundling, :]]))
            else:
                parents = bufer_pop

            new_pop_dynasty = bufer_pop  # new dynasty starts from parents
            #  add genes that is gotten by mutation
            new_pop_dynasty = np.vstack((new_pop_dynasty, self.mutation(parents, n_mutation, _mask, 0)))
            # add genes that is gotten by crossing
            new_pop_dynasty = np.vstack((new_pop_dynasty, self.crosover(bufer_pop, n_cross)))

            # combined operators: crossing + mutation
            mut_for_cross = self.crosover(bufer_pop, n_mutation_cross)
            _mask = np.ones_like(mut_for_cross)
            new_pop_dynasty = np.vstack((new_pop_dynasty, self.mutation(mut_for_cross, 1, _mask, 1)))

            # add dynasty to population
            new_pop.extend(new_pop_dynasty)

        # from time to time clear bufer with genes for crossing between dynasties
        if random.randint(0, 1):
            self.fund_best_parents = []

        self.pop = np.array(new_pop, dtype='object')  # transforming pop as numpy array

    def colibration(self):
        global __evolsearch_process_pool

        coefficient = None
        # estimate fitness using multiprocessing pool
        if __evolsearch_process_pool:
            # pool exists
            coefficient = np.asarray(__evolsearch_process_pool.map(self.colibarate_fitnes, np.arange(self.pop_size)))
        else:
            # re-create pool
            __evolsearch_process_pool = Pool(self.num_processes)
            coefficient = np.asarray(__evolsearch_process_pool.map(self.colibarate_fitnes, np.arange(self.pop_size)))

        self.coefficients = np.amax(coefficient, axis=0)
        # print('SCH ', self.coefficients)

    def get_fitnesses(self):
        '''
        simply return all fitness values of current population
        '''
        return self.fitness

    def get_dynasties_best_value(self):
        return self.dynasties_best_values

    def get_best_individual(self):
        '''
        returns 1D array of the genotype that has max fitness
        '''
        return self.best_gen

    def get_best_in_dynasties(self):

        return np.array(self.best_in_dynasties, dtype='object')

    def get_coefficients(self):
        return self.coefficients

    def get_best_individual_fitness(self):
        '''
        return the fitness value of the best individual
        '''
        return np.min(self.fitness)

    def get_best_sep_values(self):
        return self.best_sep_fitness

    def get_mean_fitness(self):
        '''
        returns the mean fitness of the population
        '''
        return np.mean(self.fitness)

    def get_fitness_variance(self):
        '''
        returns variance of the population's fitness
        '''
        return np.std(self.fitness) ** 2

    def set_direct_evolution(self, arg):

        self.direct_evolution = arg

    def set_mutation_coefficient(self, coeff):
        self.mutation_coefficient = coeff


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
