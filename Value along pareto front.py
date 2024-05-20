import numpy as np
from deap import algorithms, base, creator, tools

# Define the objectives
def f1(x):
    return 2 * x**3 - 2 * x - 1

def f2(x):
    return 2 * x**2 + 3 * x + 8

# Define the problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", lambda ind: (f1(ind[0]), f2(ind[0])))
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# Genetic Algorithm parameters
population_size = 100
generations = 100
crossover_prob = 0.7
mutation_prob = 0.2

pop = toolbox.population(n=population_size)
algorithms.eaMuPlusLambda(pop, toolbox, population_size, population_size, crossover_prob, mutation_prob, generations, verbose=False)

# Find the Pareto front
front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

# Find the individual with the smallest f2 value in the Pareto front
smallest_f2 = min(front, key=lambda x: f2(x[0]))

# Print the result
print("Function value of f2 at the end point of the Pareto front where f2 is smallest:", round(f2(smallest_f2[0]), 3))
