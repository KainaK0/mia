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
# 2. REPAIR OPERATOR
# ==========================================
class ModeRepair(Repair):
    def __init__(self, project_data):
        # Renamed to project_data to avoid confusion, though Repair class is safe
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
# 3. DECODER (SSGS)
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
# 4. PROBLEM WRAPPER (FIXED)
# ==========================================
class MRCPSP(ElementwiseProblem):
    def __init__(self, project_data):
        # FIX: Rename 'self.data' to 'self.project_data' to avoid pymoo conflict
        self.project_data = project_data 
        n = project_data.num_tasks
        super().__init__(n_var=2*n, n_obj=2, xl=0, xu=100)

    def _evaluate(self, x, out, *args, **kwargs):
        # Use the new attribute name here
        n = self.project_data.num_tasks
        prio = x[:n]
        modes = x[n:]
        makespan, cost = calculate_metrics(prio, modes, self.project_data)
        out["F"] = [makespan, cost]

# ==========================================
# 5. EXECUTION
# ==========================================
def main():
    NUM_ACTIVITIES = 150 
    
    print(f"--- Generating Data for {NUM_ACTIVITIES} activities ---")
    data = MRCPSPDataGenerator(NUM_ACTIVITIES)
    
    print(f"Graph Structure: {NUM_ACTIVITIES} nodes, {sum(len(v) for v in data.successors.values())} edges.")
    print(f"Resource Caps: {data.R_capacity} (Scaled automatically)")
    
    problem = MRCPSP(data)
    
    algorithm = NSGA2(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=TwoPointCrossover(prob=0.9), # type: ignore
        mutation=PolynomialMutation(prob=0.1, eta=20), # type: ignore
        repair=ModeRepair(data),
        eliminate_duplicates=True
    )
    
    print("Running Optimization...")
    res = minimize(problem, algorithm, ('n_gen', 50), seed=1, verbose=True)
    
    print("\n--- Best Trade-off Schedules (Pareto Front) ---")
    print(f"{'Duration (Hours)':<20} | {'Total Cost ($)':<20}")
    print("-" * 45)
    
    if res.F is not None:
        unique_F = np.unique(res.F, axis=0)
        unique_F = unique_F[unique_F[:, 0].argsort()]
        for f in unique_F:
            if f[0] < 1e5:
                print(f"{f[0]:<20.1f} | {f[1]:<20.1f}")
    else:
        print("No feasible solutions found.")

if __name__ == "__main__":
    main()