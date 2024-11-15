import os
import json
import pandas as pd
import time
from HybridGAPSO import HybridGAPSO

def fitness_fn(arguments):
    customer_priority, pre_conditions, precedence, complexity = arguments
    precedence_mapping = {'H': 3, 'M': 2, 'L': 1}
    precedence_value = precedence_mapping.get(precedence, 1)
    w_cp, w_prec, w_comp = 0.4, 0.3, 0.3
    fitness = (w_cp * customer_priority + w_prec * precedence_value + w_comp * complexity)
    return fitness

test_case_dataset_path = 'Test Case Dataset.csv'
test_customer_ranking_path = 'Test Customer Ranking.csv'

if not os.path.exists(test_case_dataset_path):
    test_case_dataset_path = '/Users/kavindu/Desktop/Test-Case-Optimization-master/dataset/Test Case Dataset.csv'

if not os.path.exists(test_customer_ranking_path):
    test_customer_ranking_path = '/Users/kavindu/Desktop/Test-Case-Optimization-master/dataset/Test Customer Ranking.csv'

test_case_dataset = pd.read_csv(test_case_dataset_path)
test_customer_ranking = pd.read_csv(test_customer_ranking_path)

print("Test Case Dataset Columns:", test_case_dataset.columns)
print("Test Customer Ranking Columns:", test_customer_ranking.columns)

merged_dataset = pd.merge(test_case_dataset, test_customer_ranking, on='TEST ID', how='inner')

merged_dataset['COMPLEXITY'] = merged_dataset['COMPLEXITY'].astype(int)
merged_dataset['PRECEDENCE'] = merged_dataset['PRECEDENCE'].map({'H': 3, 'M': 2, 'L': 1})

fitness_data = merged_dataset[['RANKING', 'PRE-CONDITIONS', 'PRECEDENCE', 'COMPLEXITY']].values

population_size = len(merged_dataset)
genome_length = 4
ga_params = {
    'mutation_rate': 0.1
}
pso_params = {
    'w': 0.5,
    'c1': 0.8,
    'c2': 0.9
}

hybrid_algo = HybridGAPSO(fitness_fn, population_size, genome_length, [1, 0, 1, 1], [5, 1, 3, 10], ga_params, pso_params)

start_time = time.time()
best_genome, best_fitness = hybrid_algo.run(iterations=100)
end_time = time.time()
execution_time = end_time - start_time

print("Best values for x: ", best_genome)
print("Best value for f(x):", best_fitness)
print("Execution Time: {:.2f} seconds".format(execution_time))

optimized_test_cases = merged_dataset.copy()
optimized_test_cases['Optimized'] = [
    '✔' if best_genome[i % genome_length] else '✘' for i in range(len(optimized_test_cases))
]

results_dir = os.path.join(os.path.dirname(__file__), 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

results = {
    'HybridGA-PSO': {
        'optimized_state': {
            'test_cases': optimized_test_cases.to_dict(orient='records'),
            'best_genome': best_genome.tolist(),
            'best_fitness': best_fitness,
            'execution_time': execution_time
        }
    }
}

json_path = os.path.join(results_dir, 'HybridGA_PSO_result.json')
with open(json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

csv_path = os.path.join(results_dir, 'HybridGA_PSO_result.csv')
optimized_test_cases.to_csv(csv_path, index=False)

tsv_path = os.path.join(results_dir, 'HybridGA_PSO_result.tsv')
optimized_test_cases.to_csv(tsv_path, sep='\t', index=False)