# One max Problem
# 
# [0 1 0 0 0 1 1 0 1 1 1 0 0 1]
# {1,3} {4,1,2,5} {3,4,1,3} {3,1}

import random

POPULATION_SIZE = 100
GENOME_LENGTH = 20
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7
GENERATIONS = 200

def random_genome(length):
    return [random.randint(a=0,b=1) for _ in range(length)]

def init_population(population_size, genome_length):
    return [random_genome(genome_length) for _ in range(population_size)]

def fitness(genome):
    return sum(genome)

def select_parent(population, fitness_values):
    total_fitness = sum(fitness_values)
    pick = random.uniform(a=0, b=total_fitness)
    current = 0
    for individual, fitness_value in zip(population, fitness_values):
        current += fitness_value
        if current > pick:
            return individual
        
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(a=1, b=len(parent1)-1)
        return parent1[:crossover_point] + parent2[crossover_point:], parent2[:crossover_point] + parent1[crossover_point:]
    else:
        return parent1, parent2
    
def mutate(genome):
    for i in range(len(genome)):
        if random.random() < MUTATION_RATE:
            genome[i] = abs(genome[i] - i)
    return genome

def genetic_algorithm():
    population = init_population(POPULATION_SIZE, GENOME_LENGTH)

    for generation in range(GENERATIONS):
        fitness_values = [fitness(genome) for genome in population]

        new_population = []

        for _ in range(POPULATION_SIZE // 2):
            parent1 = select_parent(population, fitness_values)
            parent2 = select_parent(population, fitness_values)
            
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1), mutate(offspring2)])

        population = new_population

        fitness_values = [fitness(genome) for genome in population]
        best_fitness = max(fitness_values)

        print(f"Generation {generation}: Best Fitness = {best_fitness}")
    
    best_index = fitness_values.index(max(fitness_values))
    best_solution = population[best_index]
    print(f'Best Solution: {best_solution}')
    print(f'Best Fitness: {fitness(best_solution)}')


if __name__ ==  '__main__':
    genetic_algorithm()

