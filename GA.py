from typing import Tuple 
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass

budget = 5000

pop_size = 20         ## number of individuals in the population
tournament_size = 3   ## how many individuals compete in each tournament
crossover_rate = 0.8  ## probability of crossover
mutation_rate = 0.02  ## probability to flip each bit (about 1/n)

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

# Tournament selection: pick a few random individuals, return the best one
def select_parent(fitness):
    candidates = np.random.choice(len(fitness), tournament_size, replace=False)
    best = candidates[0]
    for c in candidates:
        if fitness[c] > fitness[best]:
            best = c
    return best



# Uniform crossover: each bit comes from parent 1 or parent 2 (coin flip)
def crossover(p1, p2):
    child = p1.copy()
    for j in range(len(p1)):
        if np.random.rand() < 0.5:
            child[j] = p2[j]
    return child

# Bit-flip mutation: flip each bit with probability mutation_rate
def mutate(x):
    for j in range(len(x)):
        if np.random.rand() < mutation_rate:
            x[j] = 1 - x[j]
    return x

def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO) -> None:
    # initial_pop = ... make sure you randomly create the first population
    n = problem.meta_data.n_variables                  # NEW
    population = []                                    # NEW
    fitness = []                                       # NEW
    for i in range(pop_size):                              # NEW
        x = np.random.randint(0, 2, n)                 # NEW
        population.append(x)                           # NEW
        fitness.append(problem(x))                     # NEW
    print("best of first population:", max(fitness)) 
    
      # NEW (test only)
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # please implement the mutation, crossover, selection here
        # .....
        # this is how you evaluate one solution `x`
        # f = problem(x)

        # Elitism: the best individual goes to the new population unchanged
        best = np.argmax(fitness)
        new_population = [population[best].copy()]
        new_fitness = [fitness[best]]

        # Fill the rest of the new population with children
        while len(new_population) < pop_size and problem.state.evaluations < budget:
            p1 = population[select_parent(fitness)]
            p2 = population[select_parent(fitness)]
            if np.random.rand() < crossover_rate:
                child = crossover(p1, p2)
            else:
                child = p1.copy()
            child = mutate(child)
            new_population.append(child)
            new_fitness.append(problem(child))

        population = new_population
        fitness = new_fitness

    print("best found:", max(fitness))
    # no return value needed


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger
    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F23)
        F23.reset()
    _logger.close()
