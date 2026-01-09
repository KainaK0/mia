import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pymoo imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.repair import Repair
from pymoo.optimize import minimize

# ==========================================
# 1. AUTOMATIC DATA GENERATOR (Kept Intact)
# ==========================================
class MRCPSPDataGenerator:
    """
    Generates a synthetic Multi-Mode Resource Constrained Project Scheduling Problem.
    """
    def __init__(self, num_tasks=100):
        self.num_tasks = num_tasks
        np.random.seed(42) 
        
        # --- A. Define Resource Pools ---
        base_cap = int(5 + np.log(num_tasks) * 2) 
        self.R_capacity = np.array([base_cap * 2, base_cap, int(base_cap/2), 4]) 
        self.R_cost = np.array([10, 20, 50, 100]) 
        
        self.N_limit = np.array([num_tasks * 50, num_tasks * 100]) 
        self.N_cost = np.array([5, 10]) 

        # --- B. Generate Project Network (DAG) ---
        self.successors = {i: [] for i in range(num_tasks)}
        self.predecessors = {i: [] for i in range(num_tasks)}
        
        for i in range(1, num_tasks - 1):
            limit = min(i, 3) 
            num_preds = np.random.randint(1, limit + 1) 
            preds = np.random.choice(range(i), size=num_preds, replace=False)
            
            for p in preds:
                self.successors[p].append(i)
                self.predecessors[i].append(p)
                
        for i in range(num_tasks - 1):
            if not self.successors[i]:
                self.successors[i].append(num_tasks - 1)
                self.predecessors[num_tasks - 1].append(i)

        # --- C. Generate Execution Modes ---
        self.modes = {}
        self.modes[0] = [[0] * 7]
        self.modes[num_tasks - 1] = [[0] * 7]
        
        for t in range(1, num_tasks - 1):
            base_dur = np.random.randint(5, 20)
            modes = []
            
            # MODE 1: Fast & Expensive
            dur_1 = max(1, int(base_dur * 0.6))
            r_1 = [np.random.randint(0, cap) for cap in self.R_capacity]
            n_1 = [np.random.randint(10, 20) for _ in self.N_limit]
            modes.append([dur_1, *r_1, *n_1])
            
            # MODE 2: Normal
            dur_2 = base_dur
            r_2 = [int(r * 0.7) for r in r_1]
            n_2 = [int(n * 0.8) for n in n_1]
            modes.append([dur_2, *r_2, *n_2])
            
            # MODE 3: Slow & Cheap
            dur_3 = int(base_dur * 1.5)
            r_3 = [int(r * 0.4) for r in r_1]
            n_3 = [int(n * 0.5) for n in n_1]
            modes.append([dur_3, *r_3, *n_3])
            
            self.modes[t] = modes

# ==========================================
# 2. REPAIR OPERATOR (Kept Intact)
# ==========================================
class ModeRepair(Repair):
    def __init__(self, project_data):
        self.project_data = project_data
        super().__init__()

    def _do(self, problem, X, **kwargs):
        n = self.project_data.num_tasks
        
        for i in range(len(X)):
            mode_genes = X[i, n:].astype(int)
            
            # Renewable Check
            for t in range(n):
                max_m = len(self.project_data.modes[t])
                m_idx = mode_genes[t] % max_m
                
                req = np.array(self.project_data.modes[t][m_idx][1:5])
                
                if np.any(req > self.project_data.R_capacity):
                    for alt in range(max_m):
                        alt_req = np.array(self.project_data.modes[t][alt][1:5])
                        if np.all(alt_req <= self.project_data.R_capacity):
                            mode_genes[t] = alt
                            X[i, n + t] = alt
                            break
        return X

# ==========================================
# 3. DECODER (SSGS) (Kept Intact)
# ==========================================
def calculate_metrics(priorities, mode_indices, data):
    n = data.num_tasks
    prio_map = {i: priorities[i] for i in range(n)}
    
    unscheduled = set(range(n))
    scheduled = []
    finish_times = {i: 0 for i in range(n)}
    timeline = {}
    
    total_cost = 0.0
    for t in range(n):
        m = int(mode_indices[t]) % len(data.modes[t])
        nr_usage = np.array(data.modes[t][m][5:])
        total_cost += np.sum(nr_usage * data.N_cost)

    loop_guard = 0
    while unscheduled:
        loop_guard += 1
        if loop_guard > 10000: return 1e6, 1e6 # Failsafe
        
        ready = [t for t in unscheduled if all(p in scheduled for p in data.predecessors[t])]
        if not ready: break
        
        task = min(ready, key=lambda t: prio_map[t])
        
        es = max([finish_times[p] for p in data.predecessors[task]]) if data.predecessors[task] else 0
        
        m = int(mode_indices[task]) % len(data.modes[task])
        dat = data.modes[task][m]
        dur = dat[0]
        req_r = np.array(dat[1:5])
        
        t_start = es
        is_scheduled = False
        
        while not is_scheduled and t_start < 5000:
            valid = True
            if dur > 0:
                for t in range(t_start, t_start + dur):
                    current = timeline.get(t, np.zeros(4))
                    if np.any(current + req_r > data.R_capacity):
                        valid = False; break
            
            if valid:
                if dur > 0:
                    for t in range(t_start, t_start + dur):
                        if t not in timeline: timeline[t] = np.zeros(4)
                        timeline[t] += req_r
                        total_cost += np.sum(req_r * data.R_cost)
                
                finish_times[task] = t_start + dur
                scheduled.append(task)
                unscheduled.remove(task)
                is_scheduled = True
            else:
                t_start += 1
        
        if not is_scheduled: return 1e6, 1e6 
                
    return finish_times[n-1], total_cost

# ==========================================
# 4. PROBLEM DEFINITIONS
# ==========================================

# --- A. Multi-Objective Problem (For NSGA-II) ---
class MRCPSP(ElementwiseProblem):
    def __init__(self, project_data):
        self.project_data = project_data 
        n = project_data.num_tasks
        super().__init__(n_var=2*n, n_obj=2, xl=0, xu=100)

    def _evaluate(self, x, out, *args, **kwargs):
        n = self.project_data.num_tasks
        prio = x[:n]
        modes = x[n:]
        makespan, cost = calculate_metrics(prio, modes, self.project_data)
        out["F"] = [makespan, cost]

# --- B. Single-Objective Problem (For Standard GA) ---
class MRCPSP_SO(ElementwiseProblem):
    def __init__(self, project_data):
        self.project_data = project_data
        n = project_data.num_tasks
        # n_obj=1 for Single Objective GA
        super().__init__(n_var=2*n, n_obj=1, xl=0, xu=100)

    def _evaluate(self, x, out, *args, **kwargs):
        n = self.project_data.num_tasks
        prio = x[:n]
        modes = x[n:]
        makespan, cost = calculate_metrics(prio, modes, self.project_data)
        
        # Weighted Sum: Combine Makespan and Cost into one fitness value
        # Normalization factor (0.1) roughly balances the scale between Time and Cost
        fitness = makespan + (cost * 0.1)
        out["F"] = [fitness]

# ==========================================
# 5. VISUALIZATION FUNCTION
# ==========================================
def plot_convergence(res_nsga2, res_ga, data):
    """
    Plots the evolution of Makespan and Cost generation by generation.
    Saves the output to 'convergence_comparison.png'.
    """
    # --- 1. Extract NSGA-II History ---
    n_gen_nsga2 = len(res_nsga2.history)
    min_makespan_nsga2 = []
    min_cost_nsga2 = []

    for algo in res_nsga2.history:
        pop = algo.pop
        F = pop.get("F")
        # Filter out penalties (infeasible solutions > 1e5)
        valid_indices = np.where(F[:, 0] < 1e5)
        
        if valid_indices[0].size > 0:
            # We take the best makespan and best cost in the ENTIRE population
            # independently (they might belong to different solutions)
            min_makespan_nsga2.append(np.min(F[valid_indices, 0]))
            min_cost_nsga2.append(np.min(F[valid_indices, 1]))
        else:
            # Fallback if generation has no valid solutions
            last_ms = min_makespan_nsga2[-1] if min_makespan_nsga2 else 1e5
            last_co = min_cost_nsga2[-1] if min_cost_nsga2 else 1e5
            min_makespan_nsga2.append(last_ms)
            min_cost_nsga2.append(last_co)

    # --- 2. Extract GA History ---
    n_gen_ga = len(res_ga.history)
    ga_makespan = []
    
    for algo in res_ga.history:
        # GA in pymoo stores the best individual of generation in algo.opt
        opt = algo.opt[0]
        X = opt.X
        
        # We must decode X to get actual Makespan (since GA optimizes weighted Fitness)
        n = data.num_tasks
        prio = X[:n]
        modes = X[n:]
        ms, _ = calculate_metrics(prio, modes, data)
        ga_makespan.append(ms)

    # --- 3. Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Evolution of Makespan
    ax1.plot(range(n_gen_nsga2), min_makespan_nsga2, label="NSGA-II (Best Makespan)", color="blue", linewidth=2)
    ax1.plot(range(n_gen_ga), ga_makespan, label="Standard GA (Best Makespan)", color="red", linestyle="--", linewidth=2)
    ax1.set_title("Evolution of Makespan")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Makespan (Hours)")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Plot 2: Evolution of Cost (NSGA-II only)
    ax2.plot(range(n_gen_nsga2), min_cost_nsga2, label="NSGA-II (Best Cost)", color="green", linewidth=2)
    ax2.set_title("Evolution of Cost (NSGA-II)")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Total Cost ($)")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    # Save to file
    filename = "convergence_comparison.png"
    plt.savefig(filename)
    print(f"\n[Success] Plot saved to '{filename}'")
    plt.close()

# ==========================================
# 6. EXECUTION & COMPARISON
# ==========================================
def main():
    NUM_ACTIVITIES = 150 
    
    print(f"--- Generating Shared Data for {NUM_ACTIVITIES} activities ---")
    data = MRCPSPDataGenerator(NUM_ACTIVITIES)
    print(f"Graph Created: {sum(len(v) for v in data.successors.values())} edges.")
    
    # ---------------------------------------------------------
    # ALGORITHM 1: NSGA-II (Multi-Objective)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("1. Running NSGA-II (Multi-Objective)...")
    print("="*50)
    
    problem_mo = MRCPSP(data)
    
    algorithm_nsga2 = NSGA2(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=TwoPointCrossover(prob=0.9), 
        mutation=PolynomialMutation(prob=0.1, eta=20),
        repair=ModeRepair(data),
        eliminate_duplicates=True
    )
    
    start_time_nsga2 = time.time()
    # ENABLE SAVE_HISTORY=TRUE to track generations
    res_nsga2 = minimize(problem_mo, algorithm_nsga2, ('n_gen', 50), seed=1, verbose=False, save_history=True)
    end_time_nsga2 = time.time()
    
    time_nsga2 = end_time_nsga2 - start_time_nsga2
    print(f"NSGA-II Finished in: {time_nsga2:.4f} seconds")

    # ---------------------------------------------------------
    # ALGORITHM 2: Standard GA (Single-Objective)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("2. Running Standard GA (Single-Objective Weighted)...")
    print("="*50)
    
    problem_so = MRCPSP_SO(data)
    
    algorithm_ga = GA(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=TwoPointCrossover(prob=0.9), 
        mutation=PolynomialMutation(prob=0.1, eta=20), 
        repair=ModeRepair(data),
        eliminate_duplicates=True
    )
    
    start_time_ga = time.time()
    # ENABLE SAVE_HISTORY=TRUE to track generations
    res_ga = minimize(problem_so, algorithm_ga, ('n_gen', 50), seed=1, verbose=False, save_history=True)
    end_time_ga = time.time()
    
    time_ga = end_time_ga - start_time_ga
    print(f"Standard GA Finished in: {time_ga:.4f} seconds")
    
    # ---------------------------------------------------------
    # FINAL COMPARISON
    # ---------------------------------------------------------
    print("\n" + "#"*50)
    print(f"{'ALGORITHM COMPARISON':^50}")
    print("#"*50)
    print(f"{'Algorithm':<20} | {'Execution Time (s)':<20} | {'Best Solution Found'}")
    print("-" * 75)
    
    # Get best NSGA-II solution (Picking lowest makespan)
    if res_nsga2.F is not None:
        best_nsga2_idx = np.argmin(res_nsga2.F[:, 0]) 
        best_nsga2_val = f"Time: {res_nsga2.F[best_nsga2_idx][0]:.0f}, Cost: {res_nsga2.F[best_nsga2_idx][1]:.0f}"
    else:
        best_nsga2_val = "Infeasible"

    # Get best GA solution
    if res_ga.F is not None:
        best_ga_val = f"Fitness: {res_ga.F[0]:.2f}"
    else:
        best_ga_val = "Infeasible"
        
    print(f"{'NSGA-II':<20} | {time_nsga2:<20.4f} | {best_nsga2_val}")
    print(f"{'Standard GA':<20} | {time_ga:<20.4f} | {best_ga_val}")
    print("#"*50)

    # ---------------------------------------------------------
    # PLOT EVOLUTION
    # ---------------------------------------------------------
    print("\nPlotting convergence graphs...")
    plot_convergence(res_nsga2, res_ga, data)

if __name__ == "__main__":
    main()