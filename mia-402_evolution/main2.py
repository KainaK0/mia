import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

# ==========================================
# 1. DATA SETUP
# ==========================================

# Resource Pool (Capacity & Cost)
# Format: {ID: {'type': 'R/N', 'capacity': int, 'cost': float}}
# R = Renewable (per hour), N = Non-Renewable (per unit)
RESOURCES = {
    # Labor
    'Mech': {'type': 'R', 'capacity': 4, 'cost': 50},
    'Welder': {'type': 'R', 'capacity': 3, 'cost': 60},
    'Elec': {'type': 'R', 'capacity': 3, 'cost': 55},
    'Instru': {'type': 'R', 'capacity': 2, 'cost': 65},
    'Scaff': {'type': 'R', 'capacity': 5, 'cost': 40},
    'CraneOp': {'type': 'R', 'capacity': 2, 'cost': 70},
    # Equipment
    'TeleCrane': {'type': 'R', 'capacity': 1, 'cost': 200},
    'CraneTruck': {'type': 'R', 'capacity': 2, 'cost': 150},
    'Scaffolding': {'type': 'R', 'capacity': 10, 'cost': 20},
    # Parts (Capacity here represents Max Stock, Cost is per unit)
    'PumpPart': {'type': 'N', 'capacity': 5, 'cost': 1000},
    'GearboxPart': {'type': 'N', 'capacity': 2, 'cost': 2500},
}

# Activity List
# Format: {ID: {'duration': int, 'preds': [IDs], 'reqs': {ResID: Qty}}}
ACTIVITIES = {
    1: {'duration': 4, 'preds': [], 'reqs': {'Mech': 2, 'PumpPart': 1}},
    2: {'duration': 2, 'preds': [1], 'reqs': {'CraneOp': 1, 'CraneTruck': 1}},
    3: {'duration': 3, 'preds': [1], 'reqs': {'Elec': 1, 'Instru': 1}},
    4: {'duration': 6, 'preds': [2], 'reqs': {'Welder': 2, 'Scaff': 1, 'Scaffolding': 4}},
    5: {'duration': 8, 'preds': [2, 3], 'reqs': {'Mech': 3, 'TeleCrane': 1, 'GearboxPart': 1}},
    6: {'duration': 5, 'preds': [4, 5], 'reqs': {'Mech': 2, 'Elec': 1}}
}
NUM_ACTIVITIES = len(ACTIVITIES)

# ==========================================
# 2. THE DECODER (Serial SGS)
# ==========================================

def calculate_schedule(individual):
    """
    Decodes the priority list (individual) into a valid schedule.
    Returns: (Makespan, Total Cost)
    """
    
    # 1. Initialization
    # Tracks when each resource is free. A simple timeline approach.
    # For optimization, we use a dictionary mapping time -> available_amount
    # But for simplicity/clarity, we will scan time slots.
    
    schedule = {} # stores {act_id: {'start': t, 'end': t}}
    completed = set()
    
    # Resource usage tracking
    # R_timeline: {time_slot: {res_id: used_qty}}
    r_timeline = {}
    
    # Cost Accumulators
    total_cost = 0
    
    # Calculate Non-Renewable Costs (Fixed) immediately
    # And check global stock constraints (Simple penalty if exceeded)
    stock_usage = {k: 0 for k, v in RESOURCES.items() if v['type'] == 'N'}
    
    for act_id in range(1, NUM_ACTIVITIES + 1):
        for res, qty in ACTIVITIES[act_id]['reqs'].items():
            if RESOURCES[res]['type'] == 'N':
                stock_usage[res] += qty
                total_cost += qty * RESOURCES[res]['cost']
    
    # Check if stock limits exceeded (Penalty)
    for res, used in stock_usage.items():
        if used > RESOURCES[res]['capacity']:
            return 99999, 9999999 # Heavy penalty for invalid solution

    # 2. Scheduling Loop
    # We must respect the Priority List (individual), BUT we can only schedule
    # tasks whose predecessors are finished.
    
    # Copy priority list to not modify original
    priority_list = list(individual)
    
    while len(completed) < NUM_ACTIVITIES:
        # Find candidates: Tasks in priority list whose predecessors are done
        # and are not yet scheduled.
        candidate = None
        
        # We iterate through the priority list to find the FIRST eligible task
        for task_id in priority_list:
            if task_id in completed:
                continue
                
            preds = ACTIVITIES[task_id]['preds']
            if all(p in completed for p in preds):
                candidate = task_id
                break
        
        if candidate is None:
            # This should not happen if graph is DAG.
            break
            
        # 3. Find Earliest Start Time for Candidate
        # Start constraint 1: After all predecessors finish
        if not ACTIVITIES[candidate]['preds']:
            min_start = 0
        else:
            min_start = max([schedule[p]['end'] for p in ACTIVITIES[candidate]['preds']])
            
        duration = ACTIVITIES[candidate]['duration']
        reqs = ACTIVITIES[candidate]['reqs']
        
        # Scan forward from min_start to find a window where resources are available
        t = min_start
        scheduled = False
        
        while not scheduled:
            # Check availability for duration [t, t + duration]
            feasible = True
            
            for time_step in range(t, t + duration):
                current_usage = r_timeline.get(time_step, {})
                
                for res, qty in reqs.items():
                    if RESOURCES[res]['type'] == 'R':
                        used = current_usage.get(res, 0)
                        if used + qty > RESOURCES[res]['capacity']:
                            feasible = False
                            break
                if not feasible:
                    break
            
            if feasible:
                # Book the resources
                for time_step in range(t, t + duration):
                    if time_step not in r_timeline:
                        r_timeline[time_step] = {}
                    for res, qty in reqs.items():
                        if RESOURCES[res]['type'] == 'R':
                            r_timeline[time_step][res] = r_timeline[time_step].get(res, 0) + qty
                            # Add Variable Cost (Hourly Rate * Qty)
                            total_cost += (RESOURCES[res]['cost'] * qty)
                
                # Record Schedule
                schedule[candidate] = {'start': t, 'end': t + duration}
                completed.add(candidate)
                scheduled = True
            else:
                t += 1 # Try next time slot
                
    # Calculate Makespan
    makespan = max([s['end'] for s in schedule.values()])
    
    return makespan, total_cost

# ==========================================
# 3. GA SETUP (NSGA-II)
# ==========================================

# Create Fitness Function (Minimizing Makespan, Minimizing Cost)
# Weights are (-1.0, -1.0) because DEAP maximizes by default.
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Attribute generator: Random permutation of Activity IDs
def random_permutation():
    return random.sample(range(1, NUM_ACTIVITIES + 1), NUM_ACTIVITIES)

toolbox.register("indices", random_permutation)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register Operators
toolbox.register("evaluate", calculate_schedule)
# Ordered Crossover (CX) preserves permutation validity
toolbox.register("mate", tools.cxOrdered) 
# Shuffle Mutation (swaps two indices)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
# NSGA-II Selection
toolbox.register("select", tools.selNSGA2)

# ==========================================
# 4. MAIN LOOP
# ==========================================

def main():
    random.seed(42)
    
    # Hyperparameters
    POP_SIZE = 50
    NGEN = 20
    CXPB = 0.9  # Crossover probability
    MUTPB = 0.1 # Mutation probability
    
    pop = toolbox.population(n=POP_SIZE)
    
    # Evaluate initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        
    # Assign crowding distance (required for NSGA-II)
    pop = toolbox.select(pop, len(pop))

    print(f"Starting Evolution with {NGEN} generations...")
    
    for gen in range(1, NGEN + 1):
        # Create offspring (Tournament selection -> Crossover -> Mutation)
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values
        
        for ind in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(ind)
                del ind.fitness.values
                
        # Evaluate offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Select the next generation (Elitism via NSGA-II)
        pop = toolbox.select(pop + offspring, POP_SIZE)
        
    # ==========================================
    # 5. RESULTS
    # ==========================================
    
    # Extract Pareto Front (Non-dominated solutions)
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    print("\n--- Best Solutions (Pareto Front) ---")
    print(f"{'Solution':<15} | {'Makespan (hrs)':<15} | {'Cost ($)':<15}")
    print("-" * 50)
    
    unique_sols = set()
    for ind in pareto_front:
        m, c = ind.fitness.values
        if (m, c) not in unique_sols:
            print(f"{str(ind):<15} | {m:<15.0f} | {c:<15.0f}")
            unique_sols.add((m, c))

if __name__ == "__main__":
    main()