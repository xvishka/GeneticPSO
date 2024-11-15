# Algo devised by: https://github.com/datmemerboi/Test-Case-Optimization/blob/master/algo/GA.py
# Modified to fit this project

import os
import numpy as np
import matplotlib.pyplot as plt
import time

class GeneticAlgorithm:
    def __init__(self, fitness_function, pop_size=10, genome_length=20, lb=None, ub=None, timeplotbool=True):
        self.population = None
        self.fitness_function = fitness_function
        self.number_of_pairs = None
        self.mutation_rate = 0.005
        self.selective_pressure = 1.5
        self.allow_random_parent = True
        self.single_point_cross_over = False
        self.timeplotbool = timeplotbool

        self.pop_size = pop_size
        self.genome_length = genome_length

        self.lb = -np.ones(self.genome_length) if lb is None else np.array(lb)
        self.ub = np.ones(self.genome_length) if ub is None else np.array(ub)

        self.xaxis = []
        self.yaxis = []

    def generate_binary_population(self):
        self.population = np.random.randint(self.lb, self.ub + 1, size=(self.pop_size, self.genome_length))
        self._update_fitness_vector()
        return self.population

    def _generate_individual(self, genome_length):
        return np.random.randint(self.lb, self.ub, (genome_length), dtype=int)

    def get_fitness_vector(self):
        return self.fitness_vector

    def _update_fitness_vector(self):
        self.fitness_vector = [self.get_fitness(genome) for genome in self.population]

    def get_fitness(self, genome):
        return self.fitness_function(genome)

    def get_best_genome(self):
        self.best_genome = np.argmax(self.fitness_vector)
        return self.population[self.best_genome], self.fitness_vector[self.best_genome]

    def timeplot(self):
        if self.timeplotbool:
            if not os.path.exists('plot'):
                os.makedirs('plot')
            plt.figure(figsize=(25, 9))
            plt.xlabel("Iteration Number")
            plt.ylabel("Seconds taken")
            plt.plot(self.xaxis, self.yaxis, linewidth=4.0, color="#335241", marker="o", mfc="#29AB65")
            plt.savefig('plot/GA Iter time.png')
            plt.show()

    def run(self, max_iter):
        for i in range(max_iter):
            before = time.time()
            self._generate_offspring()  # Generate offspring from selected parents
            self._update_fitness_vector()  # Update fitness after generating offspring
            self._apply_mutation()  # Apply mutation to the new population
            self._select_new_population()  # Select the next generation based on fitness
            print(f"Population size after selection: {self.population.shape}")
            self.xaxis.append(i)
            self.yaxis.append(time.time() - before)
        self.timeplot()  # Plot only if enabled
        return self

    def _generate_offspring(self):
        parent_probabilities = np.random.rand(self.pop_size)
        parent_pairs = self._select_parents(self.number_of_pairs, parent_probabilities)
        offspring = []
        while len(offspring) < self.pop_size:
            for parent1, parent2 in parent_pairs:
                offspring.extend(self._crossover(parent1, parent2))
                if len(offspring) >= self.pop_size:
                    break
        self.population = np.array(offspring[:self.pop_size])
        print(f"Population size after offspring generation: {self.population.shape}")

    def _apply_mutation(self):
        for genome in self.population:
            self._mutate(genome, self.mutation_rate)

    def _select_new_population(self):
        fitness_scores = np.array(self.get_fitness_vector())
        print("Fitness Scores:", fitness_scores)
        probabilities = np.exp(self.selective_pressure * (fitness_scores - np.max(fitness_scores)))
        probabilities /= np.sum(probabilities)
        print("Probabilities:", probabilities)
        selected_indices = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=probabilities)
        self.population = self.population[selected_indices]

    def _crossover(self, parent1, parent2):
        if self.single_point_cross_over:
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        else:
            prob = np.random.rand(len(parent1))
            threshold = np.random.rand(len(parent1))
            children = np.empty((2, len(parent1)), dtype=bool)
            mask1 = prob < threshold
            mask2 = np.invert(mask1)
            children[0, mask1] = parent1[mask1]
            children[0, mask2] = parent2[mask2]
            children[1, mask1] = parent2[mask1]
            children[1, mask2] = parent1[mask2]
        return tuple(children)

    def _select_parents(self, number_of_pairs, parent_probabilities):
        parent_pairs = []
        for pair in range(number_of_pairs):
            parents_idx = []
            parents = []
            while len(parents_idx) != 2:
                rnd = np.random.rand()
                for i in range(len(parent_probabilities)):
                    p = parent_probabilities[i]
                    if rnd < p:
                        parents_idx.append(i)
                        parents.append(self.population[i].copy())
                        if len(parents_idx) == 2:
                            break
                        parent_probabilities += p / (len(parent_probabilities) - 1)
                        parent_probabilities[i] = 0
                        firstParentProbability = p
                        break
                    else:
                        rnd -= p
            parent_probabilities -= firstParentProbability / (len(parent_probabilities) - 1)
            parent_probabilities[parents_idx[0]] = firstParentProbability
            if self.allow_random_parent and np.all(parents[0] == parents[1]):
                parents[0] = self._generate_individual(len(parents[0]))
                parents[1] = self._generate_individual(len(parents[1]))
            parent_pairs.append(parents)
        return parent_pairs

    def _mutate(self, genome, mutation_rate):
        rnd = np.random.rand(len(genome))
        mutate_at = rnd < mutation_rate
        genome[mutate_at] = np.invert(genome[mutate_at])
        return genome

# Sample fitness function for testing
def fitness_fn(arguments):
    x1, x2, x3, x4, x5 = arguments

    if x2 == x3:
        x3 = 0
    if x1 == x2 or x1 == x3:
        x2 = x3 = 0
    if x3 > x1 and x3 > x2:
        x1 = x2 = 0
    if x2 > x1 and x2 > x3:
        x1 = x3 = 0
    if x1 > x2 and x1 > x3:
        x2 = x3 = 0

    return (x1 * 0.9 + x2 * 0.5 + x3 * 0.1) * x4 * x5

# Initialize and run the Genetic Algorithm
population_size = 100
genome_length = 5
ga = GeneticAlgorithm(
    fitness_fn,
    pop_size=population_size,
    genome_length=genome_length,
    lb=[0, 0, 0, 1, 1],
    ub=[2, 2, 2, 20, 20],
    timeplotbool=False  # Disable plotting
)
ga.generate_binary_population()
ga.number_of_pairs = 4
ga.selective_pressure = 1.4
ga.mutation_rate = 0.1
ga.run(100)

best_genome, best_fitness = ga.get_best_genome()
best_genome, best_fitness
