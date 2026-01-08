import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.repair import Repair
from pymoo.optimize import minimize

# ==========================================
# 1. AUTOMATIC DATA GENERATOR
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
# 1. DATA GENERATION
# ==========================================
class MRCPSPData:
    def __init__(self):
        self.num_tasks = 12  # 0 to 11
        
        # Renewable Resources (R1, R2)
        self.R_capacity = np.array([9, 4]) 
        self.R_cost = np.array([5, 6])     
        
        # Non-Renewable Resources (NR1, NR2)
        self.N_limit = np.array([29, 40])
        self.N_cost = np.array([2, 3])     

        self.successors = {
            0: [1, 2, 3], 1: [4, 5], 2: [9, 10], 3: [8],
            4: [6, 7], 5: [9], 6: [9], 7: [8],
            8: [11], 9: [11], 10: [11], 11: []
        }
        
        self.predecessors = {i: [] for i in range(self.num_tasks)}
        for p, children in self.successors.items():
            for c in children:
                self.predecessors[c].append(p)

        # Mode Data: TaskID -> [[Dur, R1, R2, NR1, NR2], ...]
        self.modes = {
            0:  [[0, 0, 0, 0, 0]], 
            1:  [[3, 6, 0, 9, 0], [9, 5, 0, 8, 0], [10, 0, 6, 5, 7]],
            2:  [[1, 0, 4, 0, 0], [2, 1, 7, 0, 0], [5, 0, 4, 0, 0]],
            3:  [[3, 10, 0, 0, 7], [5, 7, 0, 2, 0], [8, 6, 0, 0, 7]], 
            4:  [[4, 0, 9, 8, 0], [6, 2, 0, 0, 7], [10, 0, 5, 5, 5]],
            5:  [[2, 2, 0, 0, 0], [2, 0, 8, 0, 0], [2, 2, 0, 0, 1]],
            6:  [[4, 5, 0, 10, 0], [2, 0, 7, 0, 0], [5, 0, 10, 0, 10]],
            7:  [[4, 6, 0, 0, 1], [10, 3, 0, 10, 0], [10, 4, 0, 0, 1]],
            8:  [[2, 2, 0, 0, 0], [7, 1, 0, 0, 8], [10, 1, 0, 0, 7]],
            9:  [[1, 4, 0, 0, 0], [1, 0, 2, 0, 0], [4, 0, 0, 0, 0]],
            10: [[9, 0, 2, 0, 0], [9, 4, 1, 0, 0], [10, 0, 1, 0, 0]],
            11: [[0, 0, 0, 0, 0]]
        }

# ==========================================
# 2. REPAIR OPERATOR (CRITICAL FIX HERE)
# ==========================================
class ModeRepair(Repair):
    def __init__(self, project_data):
        self.project_data = project_data
        super().__init__()

    def _do(self, problem, X, **kwargs):
        n_tasks = self.project_data.num_tasks
        n_modes_start = n_tasks 
        
        # Iterate over each individual row in the matrix
        for i in range(len(X)):
            mode_genes = X[i, n_modes_start:].astype(int)
            
            # --- FIX PART A: Repair Impossible Modes (Renewable Capacity) ---
            # Some modes in the paper require 10 resource units, but capacity is 9.
            # We must swap these out immediately or SSGS hangs.
            for t in range(n_tasks):
                max_modes = len(self.project_data.modes[t])
                m_idx = mode_genes[t] % max_modes
                
                # Check if this mode fits in the pipe (R1 <= 9, R2 <= 4)
                mode_r = np.array(self.project_data.modes[t][m_idx][1:3])
                if np.any(mode_r > self.project_data.R_capacity):
                    # Force switch to a valid mode (Generic fallback to mode 0 or search)
                    # Simple fix: Scan for ANY valid mode
                    for alt_m in range(max_modes):
                        alt_r = np.array(self.project_data.modes[t][alt_m][1:3])
                        if np.all(alt_r <= self.project_data.R_capacity):
                            mode_genes[t] = alt_m # Found a valid one
                            X[i, n_modes_start + t] = alt_m
                            break
            
            # --- FIX PART B: Repair Budget (Non-Renewable) ---
            # Recalculate usage after renewable repair
            current_usage = np.zeros(len(self.project_data.N_limit))
            actual_mode_indices = np.zeros(n_tasks, dtype=int)

            for t in range(n_tasks):
                max_modes = len(self.project_data.modes[t])
                m_idx = mode_genes[t] % max_modes
                actual_mode_indices[t] = m_idx
                current_usage += self.project_data.modes[t][m_idx][3:]

            # Greedy Budget Repair
            for r_idx in range(len(self.project_data.N_limit)):
                while current_usage[r_idx] > self.project_data.N_limit[r_idx]:
                    best_reduction = 0
                    best_task = -1
                    best_new_mode = -1
                    
                    for t in range(1, n_tasks - 1):
                        current_m = actual_mode_indices[t]
                        current_cost = self.project_data.modes[t][current_m][3 + r_idx]
                        
                        for alt_m in range(len(self.project_data.modes[t])):
                            if alt_m == current_m: continue
                            
                            # Ensure the alternative is ALSO renewable-feasible
                            alt_r = np.array(self.project_data.modes[t][alt_m][1:3])
                            if np.any(alt_r > self.project_data.R_capacity):
                                continue 

                            alt_cost = self.project_data.modes[t][alt_m][3 + r_idx]
                            reduction = current_cost - alt_cost
                            
                            if reduction > 0 and reduction > best_reduction:
                                best_reduction = reduction
                                best_task = t
                                best_new_mode = alt_m
                    
                    if best_task != -1:
                        old_m = actual_mode_indices[best_task]
                        old_res = self.project_data.modes[best_task][old_m][3:]
                        new_res = self.project_data.modes[best_task][best_new_mode][3:]
                        
                        current_usage = current_usage - old_res + new_res
                        actual_mode_indices[best_task] = best_new_mode
                        X[i, n_modes_start + best_task] = best_new_mode
                    else:
                        break 
        return X

# ==========================================
# 3. DECODER (Updated with Safety Valve)
# ==========================================
def calculate_makespan_cost(priority_list, mode_list, data):
    n_tasks = data.num_tasks
    priorities = {i: priority_list[i] for i in range(n_tasks)}
    unscheduled = set(range(n_tasks))
    scheduled = []
    
    finish_times = {i: 0 for i in range(n_tasks)}
    r_timeline = {} 
    
    total_cost = 0
    for t in range(n_tasks):
        m_idx = int(mode_list[t]) % len(data.modes[t])
        total_cost += np.sum(data.modes[t][m_idx][3:] * data.N_cost)

    loop_safety = 0 # Prevent infinite loops
    max_loops = 5000 

    while unscheduled:
        loop_safety += 1
        if loop_safety > max_loops:
            # Emergency exit: Return Huge Penalty if scheduler hangs
            return 1e5, 1e5

        eligible = [t for t in unscheduled 
                    if all(p in scheduled for p in data.predecessors[t])]
        
        if not eligible: break 

        next_task = min(eligible, key=lambda x: priorities[x])
        
        preds = data.predecessors[next_task]
        es = max([finish_times[p] for p in preds]) if preds else 0
        
        m_idx = int(mode_list[next_task]) % len(data.modes[next_task])
        m_data = data.modes[next_task][m_idx]
        dur, req_r = m_data[0], np.array(m_data[1:3])
        
        # FAILSAFE: If Mode requires more than Capacity, task is impossible.
        if np.any(req_r > data.R_capacity):
            return 1e6, 1e6 # Return High Penalty

        current_time = es
        is_scheduled = False
        
        # Horizon check
        while not is_scheduled and current_time < 200:
            feasible = True
            if dur > 0:
                for t in range(current_time, current_time + dur):
                    usage = r_timeline.get(t, np.zeros(2))
                    if np.any(usage + req_r > data.R_capacity):
                        feasible = False; break
            
            if feasible:
                if dur > 0:
                    for t in range(current_time, current_time + dur):
                        if t not in r_timeline: r_timeline[t] = np.zeros(2)
                        r_timeline[t] += req_r
                        total_cost += np.sum(req_r * data.R_cost)
                
                finish_times[next_task] = current_time + dur
                scheduled.append(next_task)
                unscheduled.remove(next_task)
                is_scheduled = True
            else:
                current_time += 1
                
        if not is_scheduled:
            # If we hit horizon 200 and still can't schedule, return penalty
            return 1e5, 1e5
                
    return finish_times[11], total_cost

# ==========================================
# 4. PROBLEM DEFINITION
# ==========================================
class MRCPSP(ElementwiseProblem):
    def __init__(self, project_data):
        self.project_data = project_data 
        n = project_data.num_tasks
        
        super().__init__(
            n_var=2 * n, 
            n_obj=2, 
            n_ieq_constr=0, 
            xl=0, 
            xu=100
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n = self.project_data.num_tasks
        priorities = x[:n]
        mode_genes = x[n:]
        
        modes = []
        for i in range(n):
            n_modes = len(self.project_data.modes[i])
            modes.append(int(mode_genes[i]) % n_modes)
            
        makespan, cost = calculate_makespan_cost(priorities, modes, self.project_data)
        out["F"] = [makespan, cost]

# ==========================================
# 5. EXECUTION
# ==========================================
def run_mnsga2():

    NUM_ACTIVITIES = 150 
    
    print(f"--- Generating Data for {NUM_ACTIVITIES} activities ---")
    data = MRCPSPDataGenerator(NUM_ACTIVITIES)

    data = MRCPSPData()
    problem = MRCPSP(data)
    
    # We suppress strict typing warnings for Pylance using 'type: ignore'
    # because we are passing valid subclasses that Pylance doesn't expect.
    algorithm = NSGA2(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=TwoPointCrossover(prob=0.9), # type: ignore
        mutation=PolynomialMutation(prob=0.1, eta=20), # type: ignore
        repair=ModeRepair(data),
        eliminate_duplicates=True
    )

    print("Running MNSGA-II on MRCPSP Case Study...")
    res = minimize(problem, algorithm, ('n_gen', 50), seed=1, verbose=True)

    print("\n--- Final Results (Pareto Front) ---")
    print(f"{'Makespan':<15} | {'Cost':<15}")
    print("-" * 35)
    
    if res.F is not None:
        unique_F = np.unique(res.F, axis=0)
        unique_F = unique_F[unique_F[:, 0].argsort()]
        for f in unique_F:
            # Filter out penalty values
            if f[0] < 1e5:
                print(f"{f[0]:<15.1f} | {f[1]:<15.1f}")
    else:
        print("No solutions found.")

if __name__ == "__main__":
    run_mnsga2()