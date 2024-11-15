import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import time
from GA import GeneticAlgorithm
from PSO import PSO

class HybridGAPSO:
    def __init__(self, fitness_fn, pop_size, genome_length, lb, ub, ga_params, pso_params):
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.genome_length = genome_length
        self.lb = np.array(lb)
        self.ub = np.array(ub)

        # Initialize GA and PSO with correct parameters, disable GA plotting
        self.ga = GeneticAlgorithm(fitness_fn, pop_size, genome_length, lb, ub, timeplotbool=False)
        self.ga.mutation_rate = ga_params['mutation_rate']  # Set mutation rate separately
        self.pso = PSO(func=fitness_fn, dim=genome_length, lb=self.lb, ub=self.ub, pop=pop_size,
                       w=pso_params['w'], c1=pso_params['c1'], c2=pso_params['c2'])

        # Initialize PSO parameters
        self.pso_velocity = np.zeros((pop_size, genome_length))
        self.pso.pbest = np.copy(self.pso.X)
        self.pso.gbest_x = np.copy(self.pso.X[0])
        self.pso.gbest_y = float('inf')
        self.w = pso_params['w']
        self.c1 = pso_params['c1']
        self.c2 = pso_params['c2']

        self.xaxis = []
        self.yaxis = []

    def initialize_population(self):
        self.ga_population = self.ga.generate_binary_population()
        self.pso_population = self.pso.X.copy()

    def evaluate_fitness(self):
        self.ga_fitness = np.array(self.ga.get_fitness_vector())
        self.pso_fitness = np.array(self.pso.cal_y()).reshape(-1, 1)

    def evaluate_population(self, population):
        return np.array([self.fitness_fn(ind) for ind in population]).reshape(-1, 1)

    def selection(self, fitness):
        probabilities = fitness / fitness.sum()
        selected_indices = np.random.choice(np.arange(self.pop_size), size=self.pop_size, p=probabilities)
        return selected_indices

    def crossover(self, parents):
        offspring = np.empty(parents.shape)
        crossover_point = np.uint8(parents.shape[1] / 2)

        for k in range(parents.shape[0]):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

        return offspring

    def mutation(self, offspring):
        for idx in range(offspring.shape[0]):
            for _ in range(self.genome_length):
                gene_idx = np.random.randint(0, self.genome_length)
                offspring[idx, gene_idx] = np.random.uniform(self.lb[gene_idx], self.ub[gene_idx])
        return offspring

    def update_pso(self):
        r1, r2 = np.random.rand(), np.random.rand()
        self.pso_velocity = (self.w * self.pso_velocity +
                             self.c1 * r1 * (self.pso.pbest - self.pso.X) +
                             self.c2 * r2 * (self.pso.gbest_x - self.pso.X))
        self.pso.X = self.pso.X + self.pso_velocity
        self.pso.X = np.clip(self.pso.X, self.lb, self.ub)

    def hybrid_step(self):
        # GA selection, crossover, mutation
        selected_indices = self.selection(self.ga_fitness)
        offspring = self.crossover(self.ga_population[selected_indices])
        offspring = self.mutation(offspring)

        # PSO update
        self.update_pso()

        # Combine GA offspring and PSO particles
        combined_population = np.vstack((offspring, self.pso.X))
        combined_fitness = np.hstack((self.evaluate_population(offspring), self.pso.cal_y().reshape(-1, 1))).flatten()

        # Select the best individuals for the next generation
        best_indices = np.argsort(combined_fitness)[:self.pop_size]
        self.ga_population = combined_population[best_indices]
        self.pso.X = combined_population[best_indices]

    def timeplot(self):
        logging.info("Saving plot...")
        if not os.path.exists('plot'):
            logging.info("'plot' directory does not exist. Creating now.")
            os.makedirs('plot')
        plt.figure(figsize=(25, 9))
        plt.xlabel("Iteration Number")
        plt.ylabel("Seconds taken")
        plt.plot(self.xaxis, self.yaxis, linewidth=4.0, color="#570E1A", marker="o", mfc="#A6162E")
        plt.savefig('plot/HybridGA_PSO Iter time.png')
        plt.close()  # Use plt.close() instead of plt.show() to avoid unnecessary warnings

    def run(self, iterations):
        self.initialize_population()
        for i in range(iterations):
            before = time.time()
            self.evaluate_fitness()
            self.hybrid_step()
            best_fitness = min(self.ga_fitness.min(), self.pso.gbest_y)
            logging.info(f"Iteration {i + 1}/{iterations}, Best Fitness: {best_fitness}")
            self.xaxis.append(i)
            self.yaxis.append(time.time() - before)
        self.timeplot()
        return self.pso.gbest_x, self.pso.gbest_y

# Parameters for GA and PSO
ga_params = {
    'mutation_rate': 0.1
}

pso_params = {
    'w': 0.5,
    'c1': 0.8,
    'c2': 0.9
}

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def fitness_fn(arguments):
        customer_priority, pre_conditions, precedence, complexity = arguments
        precedence_mapping = {'H': 3, 'M': 2, 'L': 1}
        precedence_value = precedence_mapping.get(precedence, 1)
        w_cp, w_prec, w_comp = 0.4, 0.3, 0.3
        fitness = (w_cp * customer_priority + w_prec * precedence_value + w_comp * complexity)
        return fitness

    pop_size = 100
    genome_length = 4
    lb = [1, 0, 1, 1]
    ub = [5, 1, 3, 10]

    hybrid_algo = HybridGAPSO(fitness_fn, pop_size, genome_length, lb, ub, ga_params, pso_params)
    best_genome, best_fitness = hybrid_algo.run(iterations=100)
    print("Best values for x: ", best_genome)
    print("Best value for f(x):", best_fitness)
