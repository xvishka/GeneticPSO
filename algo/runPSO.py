import os
import json
import pandas as pd
import numpy as np
import time
from PSO import PSO

def fitness_fn(arguments):
    customer_priority, pre_conditions, precedence, complexity = arguments

    # Convert precedence to numerical values
    precedence_mapping = {'H': 3, 'M': 2, 'L': 1}
    precedence_value = precedence_mapping.get(precedence, 1)

    # Weights
    w_cp, w_prec, w_comp = 0.4, 0.3, 0.3

    # Calculate fitness
    fitness = (w_cp * customer_priority + w_prec * precedence_value + w_comp * complexity)
    return fitness

# Define paths to the test data files
test_case_dataset_path = 'Test Case Dataset.csv'
test_customer_ranking_path = 'Test Customer Ranking.csv'

# Ensure paths are valid
if not os.path.exists(test_case_dataset_path):
    test_case_dataset_path = '/Users/kavindu/Desktop/Test-Case-Optimization-master/dataset/Test Case Dataset.csv'  # Update this to the absolute path

if not os.path.exists(test_customer_ranking_path):
    test_customer_ranking_path = '/Users/kavindu/Desktop/Test-Case-Optimization-master/dataset/Test Customer Ranking.csv'  # Update this to the absolute path

# Read the test data files
test_case_dataset = pd.read_csv(test_case_dataset_path)
test_customer_ranking = pd.read_csv(test_customer_ranking_path)

# Inspect columns to ensure correct merge keys
print("Test Case Dataset Columns:", test_case_dataset.columns)
print("Test Customer Ranking Columns:", test_customer_ranking.columns)

# Merge the datasets on 'TEST ID'
merged_dataset = pd.merge(test_case_dataset, test_customer_ranking, on='TEST ID', how='inner')  # Use the correct column name

# Ensure 'PRECEDENCE' and 'COMPLEXITY' are in the expected format
merged_dataset['COMPLEXITY'] = merged_dataset['COMPLEXITY'].astype(int)
merged_dataset['PRECEDENCE'] = merged_dataset['PRECEDENCE'].map({'H': 3, 'M': 2, 'L': 1})

# Prepare the data for the fitness function
fitness_data = merged_dataset[['RANKING', 'PRE-CONDITIONS', 'PRECEDENCE', 'COMPLEXITY']].values

# Initialize and run the Particle Swarm Optimization Algorithm
instance = PSO(
    func=fitness_fn,
    dim=4,  # Updated to match the number of parameters
    lb=np.array([1, 0, 1, 1]),  # Updated lower bounds
    ub=np.array([5, 1, 3, 10]),  # Updated upper bounds
    pop=len(merged_dataset)
)

# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log the initial state
logger.info("Initial Population: %s", instance.X)

# Capture the initial state
initial_population = instance.X.copy()
initial_fitness = instance.cal_y()

initial_state = {
    'test_cases': merged_dataset.to_dict(orient='records'),
    'population': initial_population.tolist(),
    'fitness': initial_fitness.tolist()
}

# Measure execution time
start_time = time.time()
result = instance.run(max_iter=100)
end_time = time.time()
execution_time = end_time - start_time

best_genome = instance.gbest_x
best_fitness = instance.gbest_y

# Log the best results
logger.info("Best values for x: %s", best_genome)
logger.info("Best value for f(x): %s", best_fitness)
logger.info("Execution Time: %.2f seconds", execution_time)

# Create the results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(__file__), 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Apply the best genome values to the entire dataset
optimized_test_cases = merged_dataset.copy()
optimized_test_cases['Optimized'] = [
    '✔' if best_genome[i % best_genome.shape[0]] else '✘' for i in range(len(optimized_test_cases))
]

# Save results in JSON format
results = {
    'PSO': {
        'initial_state': {
            'test_cases': initial_state['test_cases'],
            'population': initial_state['population'],
            'fitness': initial_state['fitness']
        },
        'optimized_state': {
            'test_cases': optimized_test_cases.to_dict(orient='records'),
            'best_genome': best_genome.tolist(),
            'best_fitness': best_fitness,
            'execution_time': execution_time
        }
    }
}

json_path = os.path.join(results_dir, 'PSO_result.json')
with open(json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

# Save results in CSV and TSV formats
csv_path = os.path.join(results_dir, 'PSO_result.csv')
optimized_test_cases.to_csv(csv_path, index=False)

tsv_path = os.path.join(results_dir, 'PSO_result.tsv')
optimized_test_cases.to_csv(tsv_path, sep='\t', index=False)

print("Best values for x: ", best_genome)
print("Best value for f(x):", best_fitness)
print("Execution Time: {:.2f} seconds".format(execution_time))
