import numpy as np
import random
import time

from pathos.multiprocessing import ProcessPool

__evolsearch_process_pool = None


class EvolSearch:
    def __init__(self, evol_params):
        '''
        Initialize evolutionary search
        ARGS:
        evol_params: dict
            required keys -
                pop_size: int - population size,
                genotype_size: int - genotype_size,
                fitness_function: function - a user-defined function that takes a genotype as arg and returns a float fitness value
                elitist_fraction: float - fraction of top performing individuals to retain for next generation
                mutation_variance: float - variance of the gaussian distribution used for mutation noise
            optional keys -
                fitness_args: list-like - optional additional arguments to pass while calling fitness function
                                           list such that len(list) == 1 or len(list) == pop_size
                num_processes: int -  pool size for multiprocessing.pool.Pool - defaults to os.cpu_count()
        '''
        # check for required keys
        required_keys = ['pop_size', 'calibration_function', 'fitness_function', 'elitist_fraction',
                         "bounds", "probability_mask", "num_branches"]
        for key in required_keys:
            if key not in evol_params.keys():
                raise Exception('Argument evol_params does not contain the following required key: ' + key)

        # checked for all required keys
        self.pop_size = evol_params['pop_size']
        self.optional_args = False

        self.fitness_function = evol_params['fitness_function']
        self.calibration_function = evol_params['calibration_function']
        self.elitist_fraction = evol_params['elitist_fraction']
        self.probability_mask = evol_params['probability_mask']
        self.bounds = evol_params['bounds']
        self.num_branches = evol_params['num_branches']

        self.coefficients = None

        # create other required data
        self.num_processes = evol_params.get('num_processes', None)
        # self.pop = np.random.rand(self.pop_size, self.genotype_size)
        self.fitness = np.zeros(self.pop_size)
        self.num_batches = int(self.pop_size / self.num_processes)
        self.num_remainder = int(self.pop_size % self.num_processes)

        # creating the global process pool to be used across all generations
        global __evolsearch_process_pool
        __evolsearch_process_pool = ProcessPool(self.num_processes)
        time.sleep(0.5)

        self.pop = np.array([self.gen_genartion() for i in range(self.pop_size)], dtype='object')
        self.colibration()

    def gen_genartion(self):
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

        return big_gen

    def colibarate_fitnes(self, individual_index):
        if self.optional_args:
            if len(self.optional_args) == 1:
                return self.fitness_function(self.pop[individual_index, :], self.optional_args[0])
            else:
                return self.fitness_function(self.pop[individual_index, :], self.optional_args[individual_index])
        else:
            return self.calibration_function(self.pop[individual_index, :])

    def evaluate_fitness(self, individual_index):
        '''
        Call user defined fitness function and pass genotype
        '''
        if self.optional_args:
            if len(self.optional_args) == 1:
                return self.fitness_function(self.pop[individual_index, :], self.optional_args[0])
            else:
                return self.fitness_function(self.pop[individual_index, :], self.optional_args[individual_index])
        else:
            return self.fitness_function(self.pop[individual_index, :], self.coefficients)

    def elitist_selection(self):
        '''
        from fitness select top performing individuals based on elitist_fraction
        '''
        self.pop = self.pop[np.argsort(self.fitness)[-self.elitist_fraction:], :]

    def mutation_eld(self):
        '''
        create new pop by repeating mutated copies of elitist individuals
        '''
        # number of copies of elitists required
        num_reps = int((self.pop_size - self.elitist_fraction) / self.elitist_fraction) + 1

        # creating copies and adding noise
        mutated_elites = np.tile(self.pop, [num_reps, 1])
        mutated_elites += np.random.normal(loc=0., scale=self.mutation_variance,
                                           size=[num_reps * self.elitist_fraction, self.genotype_size])

        # concatenating elites with their mutated versions
        self.pop = np.vstack((self.pop, mutated_elites))

        # clipping to pop_size
        self.pop = self.pop[:self.pop_size, :]

        # clipping to genotype range
        self.pop = np.clip(self.pop, 0, 1)

    def mutation(self, parents, n_copies, mask, cross):

        mask = np.array(mask, dtype='object')
        parents = np.array(parents, dtype='object')
        if not cross:
            parents_copies = np.tile(parents, [n_copies, 1])
            parents = parents_copies[:n_copies, :]
            mask_copies = np.tile(mask, n_copies)
            mask = mask_copies[:n_copies]
        random_mat = np.array([self.gen_genartion() for i in range(parents.shape[0])], dtype='object')
        result_pop = []

        for gen_copy, random_mat_gen, mask_gen in zip(parents, random_mat, mask):
            # max_length = 0
            # gen = []
            # rand_gen = []
            # for hromasoma in gen_copy:
            #     if len(hromasoma) > max_length:
            #         max_length = len(hromasoma)
            # for hromasoma, mask_hrom in zip(gen_copy, random_mat_gen):
            #     while len(hromasoma) < max_length:
            #         hromasoma.append(None)
            #         mask_hrom.append(None)
            #     gen.append(hromasoma)
            #     rand_gen.append(mask_hrom)
            #  = np.reshape(gen_copy, (gen_copy.shape[0], max_length))
            # gen_copy = np.array(gen)
            # random_mat_gen = np.array(rand_gen)
            gen = []
            for hrom_gen, hrom_random, hrom_mask in zip(gen_copy, random_mat_gen, mask_gen):
                mut_mask = np.random.choice(2, len(hrom_gen), p=[0.7, 0.3])
                negative_mut_mask = (mut_mask - 1) * (-1)
                hrom_gen = hrom_gen * negative_mut_mask + hrom_random * mut_mask
                gen.append(hrom_gen)
            # mut_mask = np.random.randint(2, size=gen_copy.shape) #* mask_gen
            # negative_mut_mask = (mut_mask - 1) * (-1)
            result_pop.append(gen)
        result_pop = np.array(result_pop, dtype='object')
        return result_pop

    def crosover(self, parents, req_pop):
        np.random.shuffle(parents)
        children = []
        while len(children) < req_pop:
            i = 0
            for parent_1 in parents:
                j = 0
                for parent_2 in parents:
                    if random.randint(0, 1) and i != j:
                        mask = np.random.choice(2, parent_1.shape[0], p=[0.8, 0.2])
                        # mask_gen = np.random.randint(2, size=(parent_1.shape[0], parent_1.shape[1]))
                        # mask_gen = np.tile(mask_gen, [1, 1, parent_1.shape[2]])
                        negativ_mask = (mask - 1) * (-1)
                        try:
                            child = parent_1 * negativ_mask + parent_2 * mask
                            children.append(child)
                        except:
                            # print('HZ')
                            pass
                    j += 1
                i += 1

        children = np.array(children, dtype='object')[:req_pop]
        return children

    def step_generation(self):
        '''
        evaluate fitness of pop, and create new pop after elitist_selection and mutation
        '''
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

        self.fitness = np.array(self.fitness, dtype="object")
        self.mask_broken = self.fitness[:, 1]
        self.fitness = self.fitness[:, 0]
        # print(self.mask_broken)

        self.dynasties_pop = np.array_split(self.pop, self.num_branches)
        self.dynasties_fitness = np.array_split(self.fitness, self.num_branches)
        self.dynasty_mask_broken = np.array_split(self.mask_broken, self.num_branches)

        self.fund_best_parents = []
        new_pop = []
        number_dynasty = 0

        for pop, fitness, mask_broken in zip(self.dynasties_pop, self.dynasties_fitness, self.dynasty_mask_broken):
            print('Dynasty: ', number_dynasty, np.min(fitness))
            number_dynasty += 1
            dynasty_size = pop.shape[0]
            parents_indexes = np.argsort(fitness * (-1))[-self.elitist_fraction:]
            # bufer_pop = pop[np.argsort(fitness)[:self.elitist_fraction]]  # parents
            # _mask = mask_broken[np.argsort(fitness)[:self.elitist_fraction]]
            bufer_pop = pop[parents_indexes, :]
            _mask = mask_broken[parents_indexes]
            # print(bufer_pop)
            if len(self.fund_best_parents) == 0:
                self.fund_best_parents = bufer_pop
            else:
                fund_best_parents = np.append(self.fund_best_parents, bufer_pop, axis=0)

            n_parents = bufer_pop.shape[0]  # number of parents
            n_mutation = int((dynasty_size - n_parents) / 3)  # number copies for mutation
            n_cross = int((dynasty_size - n_parents) / 3)  # number copies for crosingover
            n_mutation_cross = dynasty_size - n_parents - n_mutation - n_cross  # number for cross and mutation

            if random.randint(0, 1) and self.fund_best_parents.shape[0] > 0:
                index_foundling = np.random.choice(np.arange(self.fund_best_parents.shape[0]))
                parents = np.vstack((bufer_pop, [self.fund_best_parents[index_foundling, :]]))
            else:
                parents = bufer_pop

            new_pop_dynasty = bufer_pop
            new_pop_dynasty = np.vstack((new_pop_dynasty, self.mutation(parents, n_mutation, _mask, 0)))

            new_pop_dynasty = np.vstack((new_pop_dynasty, self.crosover(bufer_pop, n_cross)))
            mut_for_cross = self.crosover(bufer_pop, n_mutation_cross)
            _mask = np.ones_like(mut_for_cross)
            new_pop_dynasty = np.vstack((new_pop_dynasty, self.mutation(mut_for_cross, 1, _mask, 1)))
            new_pop.extend(new_pop_dynasty)

        if random.randint(0, 1):
            self.fund_best_parents = []

        self.pop = np.array(new_pop, dtype='object')
        print('Mass_after', self.pop.shape[0])

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

    def execute_search(self, num_gens):
        '''
        runs the evolutionary algorithm for given number of generations, num_gens
        '''
        # step generation num_gens times
        for gen in np.arange(num_gens):
            self.step_generation()

    def get_fitnesses(self):
        '''
        simply return all fitness values of current population
        '''
        return self.fitness

    def get_best_individual(self):
        '''
        returns 1D array of the genotype that has max fitness
        '''
        return self.pop[np.argmin(self.fitness), :]

    def get_best_individual_fitness(self):
        '''
        return the fitness value of the best individual
        '''
        return np.min(self.fitness)

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
