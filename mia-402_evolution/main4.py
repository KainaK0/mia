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
# 1. COMPLEX DATA GENERATION (30 Tasks)
# ==========================================
class MRCPSPData:
    def __init__(self):
        # Increased to 30 Tasks (0 to 29)
        self.num_tasks = 30
        
        # Renewable Resources (R1, R2, R3, R4)
        # Added two more resource types (e.g., Specialized Machinery, Inspectors)
        self.R_capacity = np.array([12, 8, 6, 4]) 
        self.R_cost = np.array([5, 8, 12, 15])     
        
        # Non-Renewable Resources (NR1, NR2) - Tighter Budget relative to scale
        self.N_limit = np.array([150, 200])
        self.N_cost = np.array([2, 4])     

        # --- Complex Precedence Logic (DAG) ---
        # 0 is Start, 29 is End
        self.successors = {
            0: [1, 2, 3, 4],       # Parallel start
            1: [5, 6],
            2: [7],
            3: [8, 9],
            4: [10],
            5: [11, 12],
            6: [12],               # Convergence
            7: [13],
            8: [13, 14],
            9: [15],
            10: [15, 16],
            11: [17],
            12: [17, 18],
            13: [18, 19],
            14: [19],
            15: [20],
            16: [20, 21],
            17: [22],
            18: [22, 23],
            19: [23, 24],
            20: [24, 25],
            21: [25],
            22: [26],
            23: [26, 27],
            24: [27],
            25: [27, 28],
            26: [29],
            27: [29],
            28: [29],
            29: []
        }
        
        self.predecessors = {i: [] for i in range(self.num_tasks)}
        for p, children in self.successors.items():
            for c in children:
                self.predecessors[c].append(p)

        # --- Mode Data Generation ---
        # Format: TaskID -> [[Dur, R1, R2, R3, R4, NR1, NR2], ...]
        # 3 Modes per task: Fast/Expensive vs Slow/Cheap vs Balanced
        self.modes = {}
        
        # Dummy Start/End
        self.modes[0] = [[0, 0,0,0,0, 0,0]]
        self.modes[29] = [[0, 0,0,0,0, 0,0]]

        np.random.seed(42) # Deterministic for reproducibility
        
        for t in range(1, 29):
            task_modes = []
            # Base duration/resource need for this task
            base_dur = np.random.randint(3, 15)
            
            # Mode 1: Fast & Expensive (High Renewable usage)
            m1_dur = max(2, base_dur - 3)
            m1_R = np.random.randint(0, 5, size=4) # R1..R4
            m1_N = np.random.randint(5, 15, size=2) # NR1..NR2
            task_modes.append([m1_dur, *m1_R, *m1_N])
            
            # Mode 2: Slow & Cheap (Low Renewable usage)
            m2_dur = base_dur + 5
            m2_R = np.maximum(0, m1_R - 2)
            m2_N = np.maximum(0, m1_N - 4)
            task_modes.append([m2_dur, *m2_R, *m2_N])
            
            # Mode 3: Balanced
            m3_dur = base_dur
            m3_R = np.maximum(0, m1_R - 1)
            m3_N = np.maximum(0, m1_N - 2)
            task_modes.append([m3_dur, *m3_R, *m3_N])
            
            self.modes[t] = task_modes

# ==========================================
# 2. REPAIR OPERATOR (Scalable)
# ==========================================
class ModeRepair(Repair):
    def __init__(self, project_data):
        self.project_data = project_data
        super().__init__()

    def _do(self, problem, X, **kwargs):
        n_tasks = self.project_data.num_tasks
        n_modes_start = n_tasks 
        R_cap = self.project_data.R_capacity
        N_lim = self.project_data.N_limit

        for i in range(len(X)):
            mode_genes = X[i, n_modes_start:].astype(int)
            
            # --- FIX PART A: Renewable Capacity Feasibility ---
            for t in range(n_tasks):
                max_modes = len(self.project_data.modes[t])
                m_idx = mode_genes[t] % max_modes
                
                # Check R1..R4 capacity
                # Mode structure: [Dur, R1, R2, R3, R4, NR1, NR2] -> Renewable is indices 1:5
                mode_r = np.array(self.project_data.modes[t][m_idx][1:5])
                
                if np.any(mode_r > R_cap):
                    # Force switch to valid mode
                    valid_found = False
                    for alt_m in range(max_modes):
                        alt_r = np.array(self.project_data.modes[t][alt_m][1:5])
                        if np.all(alt_r <= R_cap):
                            mode_genes[t] = alt_m
                            X[i, n_modes_start + t] = alt_m
                            valid_found = True
                            break
            
            # --- FIX PART B: Budget Feasibility ---
            current_usage = np.zeros(len(N_lim))
            actual_mode_indices = np.zeros(n_tasks, dtype=int)

            for t in range(n_tasks):
                max_modes = len(self.project_data.modes[t])
                m_idx = mode_genes[t] % max_modes
                actual_mode_indices[t] = m_idx
                # NR is indices 5:7
                current_usage += self.project_data.modes[t][m_idx][5:]

            for r_idx in range(len(N_lim)):
                while current_usage[r_idx] > N_lim[r_idx]:
                    best_reduction = 0
                    best_task = -1
                    best_new_mode = -1
                    
                    for t in range(1, n_tasks - 1):
                        current_m = actual_mode_indices[t]
                        # Cost index in mode array is 5 + r_idx
                        current_cost = self.project_data.modes[t][current_m][5 + r_idx]
                        
                        for alt_m in range(len(self.project_data.modes[t])):
                            if alt_m == current_m: continue
                            
                            # Check renewable feasibility first
                            alt_r = np.array(self.project_data.modes[t][alt_m][1:5])
                            if np.any(alt_r > R_cap): continue 

                            alt_cost = self.project_data.modes[t][alt_m][5 + r_idx]
                            reduction = current_cost - alt_cost
                            
                            if reduction > 0 and reduction > best_reduction:
                                best_reduction = reduction
                                best_task = t
                                best_new_mode = alt_m
                    
                    if best_task != -1:
                        old_m = actual_mode_indices[best_task]
                        old_res = self.project_data.modes[best_task][old_m][5:]
                        new_res = self.project_data.modes[best_task][best_new_mode][5:]
                        
                        current_usage = current_usage - old_res + new_res
                        actual_mode_indices[best_task] = best_new_mode
                        X[i, n_modes_start + best_task] = best_new_mode
                    else:
                        break 
        return X

# ==========================================
# 3. DECODER
# ==========================================
def calculate_makespan_cost(priority_list, mode_list, data):
    n_tasks = data.num_tasks
    priorities = {i: priority_list[i] for i in range(n_tasks)}
    unscheduled = set(range(n_tasks))
    scheduled = []
    
    finish_times = {i: 0 for i in range(n_tasks)}
    r_timeline = {} 
    
    total_cost = 0.0
    for t in range(n_tasks):
        m_idx = int(mode_list[t]) % len(data.modes[t])
        # NR cost (indices 5,6)
        nr_usage = np.array(data.modes[t][m_idx][5:])
        total_cost += np.sum(nr_usage * data.N_cost)

    loop_safety = 0 
    
    while unscheduled:
        loop_safety += 1
        if loop_safety > 10000: return 1e6, 1e6 

        eligible = [t for t in unscheduled 
                    if all(p in scheduled for p in data.predecessors[t])]
        
        if not eligible: break 

        next_task = min(eligible, key=lambda x: priorities[x])
        
        preds = data.predecessors[next_task]
        es = max([finish_times[p] for p in preds]) if preds else 0
        
        m_idx = int(mode_list[next_task]) % len(data.modes[next_task])
        m_data = data.modes[next_task][m_idx]
        dur = m_data[0]
        # Renewable is indices 1:5
        req_r = np.array(m_data[1:5]) 
        
        if np.any(req_r > data.R_capacity): return 1e6, 1e6 

        current_time = es
        is_scheduled = False
        
        while not is_scheduled and current_time < 500: # Extended horizon
            feasible = True
            if dur > 0:
                for t in range(current_time, current_time + dur):
                    usage = r_timeline.get(t, np.zeros(4))
                    if np.any(usage + req_r > data.R_capacity):
                        feasible = False; break
            
            if feasible:
                if dur > 0:
                    for t in range(current_time, current_time + dur):
                        if t not in r_timeline: r_timeline[t] = np.zeros(4)
                        r_timeline[t] += req_r
                        total_cost += np.sum(req_r * data.R_cost)
                
                finish_times[next_task] = current_time + dur
                scheduled.append(next_task)
                unscheduled.remove(next_task)
                is_scheduled = True
            else:
                current_time += 1
                
        if not is_scheduled: return 1e6, 1e6
                
    return finish_times[n_tasks-1], total_cost

# ==========================================
# 4. PROBLEM & EXECUTION
# ==========================================
class MRCPSP(ElementwiseProblem):
    def __init__(self, project_data):
        self.project_data = project_data 
        n = project_data.num_tasks
        super().__init__(n_var=2 * n, n_obj=2, n_ieq_constr=0, xl=0, xu=100)

    def _evaluate(self, x, out, *args, **kwargs):
        n = self.project_data.num_tasks
        priorities = x[:n]
        mode_genes = x[n:]
        modes = [int(mode_genes[i]) % len(self.project_data.modes[i]) for i in range(n)]
        makespan, cost = calculate_makespan_cost(priorities, modes, self.project_data)
        out["F"] = [makespan, cost]

def run_mnsga2():
    print("Initializing Complex Dataset (30 Tasks)...")
    data = MRCPSPData()
    problem = MRCPSP(data)
    
    algorithm = NSGA2(
        pop_size=150, # Increased population for larger search space
        sampling=IntegerRandomSampling(),
        crossover=TwoPointCrossover(prob=0.9), # type: ignore
        mutation=PolynomialMutation(prob=0.1, eta=20), # type: ignore
        repair=ModeRepair(data),
        eliminate_duplicates=True
    )

    print("Running MNSGA-II...")
    res = minimize(problem, algorithm, ('n_gen', 100), seed=42, verbose=True) # More generations

    print("\n--- Final Results (Pareto Front) ---")
    print(f"{'Makespan':<15} | {'Cost':<15}")
    print("-" * 35)
    
    if res.F is not None:
        unique_F = np.unique(res.F, axis=0)
        unique_F = unique_F[unique_F[:, 0].argsort()]
        for f in unique_F:
            if f[0] < 1e5:
                print(f"{f[0]:<15.1f} | {f[1]:<15.1f}")
    else:
        print("No feasible solutions found.")

if __name__ == "__main__":
    run_mnsga2()