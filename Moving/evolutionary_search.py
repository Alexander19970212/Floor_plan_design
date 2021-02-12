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

        self.pop = np.array([self.gen_genartion() for i in range(self.pop_size)])
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

    def mutation(self, parents, n_copies, mask):
        mask = np.array(mask, dtype='object')
        parents = np.array(parents, dtype='object')
        parents_copies = np.tile(parents, [n_copies, 1])
        parents_copies = parents_copies[:n_copies, :]
        mask_copies = np.tile(mask, n_copies)
        mask_copies = mask_copies[:n_copies]
        random_mat = np.array([self.gen_genartion() for i in range(n_copies)])
        result_pop = []
        for gen_copy, random_mat_gen, mask_gen in zip(parents_copies, random_mat, mask_copies):
            mut_mask = np.random.randint(2, size=np.array(gen_copy).shape) #* mask_gen
            negative_mut_mask = (mut_mask - 1) * (-1)
            result_pop.append(gen_copy * negative_mut_mask + random_mat_gen * mut_mask)
        return result_pop

    def crosover(self, parents, req_pop):
        np.random.shuffle(parents)
        children = np.array([])
        while children.shape[0] < req_pop:
            for parent_1 in parents:
                for parent_2 in parents:
                    if random.randint(0, 1):
                        mask_gen = np.random.randint(2, size=(parent_1.shape[0], parent_1.shape[1]))
                        mask_gen = np.tile(mask_gen, [1, 1, parent_1.shape[2]])
                        negativ_mask_gen = (mask_gen - 1) * (-1)
                        child = parent_1 * negativ_mask_gen + parent_2 * mask_gen
                        children = np.append(children, [child], axis=0)

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
        #print(self.mask_broken)

        self.dynasties_pop = np.array_split(self.pop, self.num_branches)
        self.dynasties_fitness = np.array_split(self.fitness, self.num_branches)
        self.dynasty_mask_broken = np.array_split(self.mask_broken, self.num_branches)

        self.fund_best_parents = None
        new_pop = np.array([])

        for pop, fitness, mask_broken in zip(self.dynasties_pop, self.dynasties_fitness, self.dynasty_mask_broken):
            dynasty_size = pop.shape[0]
            bufer_pop = pop[np.argsort(fitness)[-self.elitist_fraction:], :]  # parents
            print(bufer_pop)
            if self.fund_best_parents == None:
                self.fund_best_parents = bufer_pop
            else:
                fund_best_parents = np.append(self.fund_best_parents, bufer_pop, axis=0)

            _mask = mask_broken[np.argsort(fitness)[-self.elitist_fraction:]]
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
            new_pop_dynasty = np.vstack((new_pop_dynasty, self.mutation(parents, n_mutation, _mask)))

            new_pop_dynasty = np.vstack((new_pop_dynasty, self.crosover(parents, n_cross)))
            mut_for_cross = self.crosover(bufer_pop, n_mutation_cross)
            _mask = np.ones_like(mut_for_cross)
            new_pop_dynasty = np.vstack((new_pop_dynasty, self.mutation(mut_for_cross, 1, _mask)))
            new_pop = np.append(new_pop, new_pop_dynasty, axis=0)

        if random.randint(0, 1):
            self.fund_best_parents = None

        self.pop = new_pop

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
        print('SCH ', self.coefficients)


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
        return self.pop[np.argmax(self.fitness), :]

    def get_best_individual_fitness(self):
        '''
        return the fitness value of the best individual
        '''
        return np.max(self.fitness)

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
