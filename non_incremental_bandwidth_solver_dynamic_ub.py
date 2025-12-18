# non_incremental_bandwidth_solver_dynamic_ub.py
# 2D Bandwidth Minimization using Non-Incremental SAT with Dynamic UB Update
# Strategy: Create fresh solver for each K value, update UB = K at each iteration
# Key Innovation: UB in distance constraints is updated to current K being tested

import os
import sys
import math
import time
import gc
import ctypes
from pysat.formula import IDPool
from pysat.solvers import Glucose42, Cadical195, Solver

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Use the cutoff-optimized distance encoder
    from distance_encoder_cutoff import encode_abs_distance_cutoff, calculate_theoretical_upper_bound
    from position_constraints import encode_all_position_constraints, create_position_variables
    print("All modules loaded OK (using cutoff-optimized distance encoder)")
except ImportError as e:
    print(f"Import error: {e}")
    print("Need required modules: distance_encoder_cutoff.py, position_constraints.py")
    raise ImportError("Missing required modules")


class NonIncrementalBandwidthSolverDynamicUB:
    """
    2D Bandwidth Minimization solver using Non-Incremental SAT with Dynamic UB Update
    
    KEY INNOVATION - Dynamic UB Update:
    Unlike the standard cutoff solver that uses a fixed theoretical UB for all K values,
    this solver updates UB = K at each iteration. This means:
    
    1. When testing K=12: UB=12, encode distance constraints with UB=12
    2. When testing K=11: UB=11, encode distance constraints with UB=11 (tighter constraints)
    3. When testing K=10: UB=10, encode distance constraints with UB=10 (even tighter)
    ... and so on
    
    Benefits of Dynamic UB:
    1. Progressively tighter constraints as K decreases
    2. Fewer T variables at smaller K values (T_1 to T_K instead of T_1 to T_theoretical_ub)
    3. More mutual exclusion clauses at smaller K (more position pairs forbidden)
    4. Potentially faster SAT solving due to smaller search space
    5. Fresh solver + fresh encoding = no stale learnt clauses
    
    Strategy:
    1. Calculate initial UB = min(upper_bound, theoretical_ub)
    2. For each K value from initial_ub down to 1:
       a. Create FRESH solver
       b. Create FRESH IDPool for variables (to avoid variable ID accumulation)
       c. Set effective_ub = K (DYNAMIC UPDATE)
       d. Encode position constraints
       e. Encode distance constraints with effective_ub = K
       f. Encode bandwidth ≤ K constraints
       g. Solve and analyze result
       h. If SAT with actual bandwidth Y < K, jump to K = Y
       i. If UNSAT, optimal is previous K
       j. Delete solver and cleanup
    
    Problem: Place n vertices on n×n grid to minimize bandwidth
    bandwidth = max{|x_u - x_v| + |y_u - y_v| : (u,v) ∈ E}
    """
    
    def __init__(self, n, solver_type='glucose42'):
        self.n = n
        self.solver_type = solver_type
        
        # Base variable pool (will be recreated per K)
        self.vpool = None
        
        # Position variables for X,Y coordinates (will be recreated per K)
        self.X_vars = {}
        self.Y_vars = {}
        
        # Distance variables with dynamic UB optimization (will be recreated per K)
        self.Tx_vars = {}
        self.Ty_vars = {}
        
        self.edges = []
        self.last_model = None
        
        # Theoretical upper bound (starting point, will be dynamically reduced)
        self.theoretical_ub = calculate_theoretical_upper_bound(n)
        
        # Performance tracking
        self.cumulative_solve_time = 0.0
        self.cumulative_encode_time = 0.0
        
        print(f"Created non-incremental solver with DYNAMIC UB update: n={n}, using {solver_type}")
        print(f"Theoretical UB: {self.theoretical_ub}")
        print(f"Strategy: Fresh solver per K + DYNAMIC UB = K at each iteration")
        print(f"Key benefit: UB progressively tightens as K decreases")
    
    def set_graph_edges(self, edges):
        """Set graph edges"""
        self.edges = edges
        print(f"Graph has {len(edges)} edges")
        print(f"Dynamic UB: Distance constraints will be encoded with UB = current K")
    
    def _reset_variables(self):
        """Reset all variables for fresh encoding at new K"""
        self.vpool = IDPool()
        self.X_vars = {}
        self.Y_vars = {}
        self.Tx_vars = {}
        self.Ty_vars = {}
    
    def _create_position_variables(self):
        """Create position variables for vertices on X and Y axes"""
        self.X_vars, self.Y_vars = create_position_variables(self.n, self.vpool)
    
    def _prepare_distance_variable_prefixes(self):
        """Prepare distance variable prefixes for each edge"""
        for i, (u, v) in enumerate(self.edges):
            edge_id = f'edge_{u}_{v}'
            tx_prefix = f"Tx[{u},{v}]"
            ty_prefix = f"Ty[{u},{v}]"
            self.Tx_vars[edge_id] = {'prefix': tx_prefix, 'vars': {}}
            self.Ty_vars[edge_id] = {'prefix': ty_prefix, 'vars': {}}
    
    def _encode_position_constraints_to_solver(self, solver):
        """
        Encode position constraints directly to solver
        
        Returns encoding time and clause count for metrics
        """
        encode_start_time = time.time()
        
        constraints = encode_all_position_constraints(self.n, self.X_vars, self.Y_vars, self.vpool)
        
        clauses_added = 0
        for clause in constraints:
            solver.add_clause(clause)
            clauses_added += 1
        
        encode_time = time.time() - encode_start_time
        
        vars_after = solver.nof_vars()
        clauses_after = solver.nof_clauses()
        
        return encode_time, clauses_added, vars_after, clauses_after
    
    def _encode_distance_constraints_to_solver(self, solver, effective_ub):
        """
        Encode distance constraints with DYNAMIC UB directly to solver
        
        CRITICAL DIFFERENCE: Uses effective_ub = current K being tested,
        NOT the fixed theoretical_ub.
        
        Args:
            solver: SAT solver instance
            effective_ub: Current K value (used as UB for distance encoding)
        
        Returns encoding time and stats for metrics
        """
        encode_start_time = time.time()
        
        edges_processed = 0
        for edge_id, (u, v) in zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges):
            # X distance with DYNAMIC UB = current K
            tx_prefix = self.Tx_vars[edge_id]['prefix']
            tx_clauses, tx_vars = encode_abs_distance_cutoff(
                self.X_vars[u], self.X_vars[v],
                effective_ub, self.vpool, tx_prefix  # ← DYNAMIC UB
            )
            self.Tx_vars[edge_id]['vars'] = tx_vars
            for c in tx_clauses:
                solver.add_clause(c)
            tx_clauses.clear()
            del tx_clauses
            
            # Y distance with DYNAMIC UB = current K
            ty_prefix = self.Ty_vars[edge_id]['prefix']
            ty_clauses, ty_vars = encode_abs_distance_cutoff(
                self.Y_vars[u], self.Y_vars[v],
                effective_ub, self.vpool, ty_prefix  # ← DYNAMIC UB
            )
            self.Ty_vars[edge_id]['vars'] = ty_vars
            for c in ty_clauses:
                solver.add_clause(c)
            ty_clauses.clear()
            del ty_clauses
            
            edges_processed += 1
        
        encode_time = time.time() - encode_start_time
        
        vars_after = solver.nof_vars()
        clauses_after = solver.nof_clauses()
        
        return encode_time, edges_processed, vars_after, clauses_after
    
    def _encode_bandwidth_constraints_to_solver(self, solver, K):
        """
        Encode bandwidth <= K constraints directly to solver
        
        For each edge: (Tx<=K) ∧ (Ty<=K) ∧ (Tx>=i → Ty<=K-i)
        
        Note: Since effective_ub = K, all T variables exist up to K,
        so no need to check if variables exist beyond K.
        
        Returns encoding time and clause count for metrics
        """
        encode_start_time = time.time()
        
        clauses_added = 0
        edges_processed = 0
        
        for edge_id in self.Tx_vars:
            tx_vars = self.Tx_vars[edge_id]['vars']
            ty_vars = self.Ty_vars[edge_id]['vars']
            edges_processed += 1
            
            # Tx <= K: Since T variables are T_1 to T_K,
            # T_K means "distance >= K", so ¬T_K means "distance < K"
            # But we want distance <= K, which is "NOT (distance >= K+1)"
            # However, with effective_ub = K, we don't have T_{K+1}
            # The mutual exclusion in distance encoding already forbids distance > K
            # So bandwidth <= K is implicitly enforced!
            
            # But we still add the implication constraints for tightness:
            # Tx >= i → Ty <= K-i
            for i in range(1, K + 1):
                remaining = K - i
                if remaining >= 0:
                    # Tx >= i → Ty <= remaining
                    # Since remaining < K when i > 0, we need T_{remaining+1} to exist
                    if i in tx_vars and (remaining + 1) in ty_vars:
                        solver.add_clause([-tx_vars[i], -ty_vars[remaining + 1]])
                        clauses_added += 1
                    
                    # Ty >= i → Tx <= remaining (symmetry)
                    if i in ty_vars and (remaining + 1) in tx_vars:
                        solver.add_clause([-ty_vars[i], -tx_vars[remaining + 1]])
                        clauses_added += 1
        
        encode_time = time.time() - encode_start_time
        
        vars_after = solver.nof_vars()
        clauses_after = solver.nof_clauses()
        
        return encode_time, clauses_added, vars_after, clauses_after
    
    def _create_solver(self):
        """Create fresh SAT solver instance"""
        if self.solver_type == 'glucose42':
            return Glucose42()
        elif self.solver_type == 'cadical195':
            return Cadical195()
        else:
            print(f"Unknown solver '{self.solver_type}', using Glucose42")
            return Glucose42()
    
    def _extract_positions_from_model(self, model):
        """
        Extract vertex positions from SAT solution
        
        Optimized: Use set for O(1) lookup
        """
        posset = {lit for lit in model if lit > 0}
        
        positions = {}
        for v in range(1, self.n + 1):
            for pos in range(1, self.n + 1):
                if self.X_vars[v][pos-1] in posset:
                    positions[f'X_{v}'] = pos
                    break
            
            for pos in range(1, self.n + 1):
                if self.Y_vars[v][pos-1] in posset:
                    positions[f'Y_{v}'] = pos
                    break
        
        return positions
    
    def _calculate_bandwidth(self, positions):
        """Calculate bandwidth from vertex positions"""
        max_distance = 0
        edge_distances = []
        
        for u, v in self.edges:
            x_u = positions.get(f'X_{u}', 0)
            y_u = positions.get(f'Y_{u}', 0)
            x_v = positions.get(f'X_{v}', 0)
            y_v = positions.get(f'Y_{v}', 0)
            
            distance = abs(x_u - x_v) + abs(y_u - y_v)
            max_distance = max(max_distance, distance)
            edge_distances.append((u, v, distance))
        
        return max_distance, edge_distances
    
    def extract_actual_bandwidth(self, model):
        """Extract actual bandwidth from SAT solution without printing details"""
        positions = self._extract_positions_from_model(model)
        bandwidth, _ = self._calculate_bandwidth(positions)
        return bandwidth
    
    def solve_with_dynamic_ub(self, upper_bound):
        """
        Main non-incremental SAT solving with DYNAMIC UB update
        
        CRITICAL INNOVATION: At each K iteration, we set effective_ub = K
        
        Strategy:
        1. Start from min(upper_bound, theoretical_ub)
        2. For each K from start down to 1:
           a. Create fresh solver AND fresh IDPool
           b. Set effective_ub = K (DYNAMIC UPDATE)
           c. Create fresh position variables
           d. Encode position constraints
           e. Encode distance constraints with effective_ub = K
           f. Encode bandwidth ≤ K constraints
           g. Solve
           h. If SAT with actual bandwidth Y < K, jump to K = Y
           i. If UNSAT, optimal is previous K
           j. Delete solver and cleanup
        """
        print(f"\nNON-INCREMENTAL SAT OPTIMIZATION WITH DYNAMIC UB")
        print(f"Strategy: Fresh solver per K + DYNAMIC UB = K at each iteration")
        print(f"Theoretical UB (starting point): {self.theoretical_ub}")
        print(f"Solver: {self.solver_type.upper()} (fresh instance per K)")
        
        # Apply theoretical upper bound cutoff initially
        current_k = min(upper_bound, self.theoretical_ub)
        
        if current_k < upper_bound:
            print(f"Initial cutoff applied: Adjusted upper bound from {upper_bound} to {current_k}")
        
        optimal_k = None
        solver_stats = {
            'total_solves': 0,
            'sat_results': 0,
            'unsat_results': 0,
            'smart_jumps': 0,
            'total_encoding_time': 0.0,
            'initial_ub_cutoff_benefit': upper_bound - current_k if upper_bound > current_k else 0
        }
        
        while current_k >= 1:
            print(f"\n{'='*60}")
            print(f"Testing K = {current_k}")
            print(f"{'='*60}")
            
            # CRITICAL: Dynamic UB = current K
            effective_ub = current_k
            print(f"  DYNAMIC UB = {effective_ub} (same as K being tested)")
            
            solver_stats['total_solves'] += 1
            
            # Step 1: Reset all variables for fresh encoding
            print(f"  Resetting variables for fresh encoding...")
            self._reset_variables()
            
            # Step 2: Create fresh position variables
            print(f"  Creating fresh position variables...")
            self._create_position_variables()
            
            # Step 3: Prepare distance variable prefixes
            self._prepare_distance_variable_prefixes()
            
            # Step 4: Create fresh solver
            print(f"  Creating fresh solver...")
            solver = self._create_solver()
            
            # Step 5: Encode all constraints with DYNAMIC UB
            k_encode_start = time.time()
            
            # 5a. Position constraints
            print(f"  Encoding position constraints...")
            pos_time, pos_clauses, pos_vars, pos_cls = self._encode_position_constraints_to_solver(solver)
            print(f"    Position: {pos_clauses} clauses, {pos_time:.3f}s")
            
            # 5b. Distance constraints with DYNAMIC UB = K
            print(f"  Encoding distance constraints with DYNAMIC UB = {effective_ub}...")
            dist_time, dist_edges, dist_vars, dist_cls = self._encode_distance_constraints_to_solver(solver, effective_ub)
            print(f"    Distance: {dist_edges} edges, {dist_time:.3f}s")
            print(f"    T variables per edge: {2 * effective_ub} (T_1 to T_{effective_ub} for X and Y)")
            
            # 5c. Bandwidth constraints for this K
            print(f"  Encoding bandwidth constraints for K={current_k}...")
            bw_time, bw_clauses, bw_vars, bw_cls = self._encode_bandwidth_constraints_to_solver(solver, current_k)
            print(f"    Bandwidth: {bw_clauses} clauses, {bw_time:.3f}s")
            
            total_encode_time = time.time() - k_encode_start
            solver_stats['total_encoding_time'] += total_encode_time
            self.cumulative_encode_time += total_encode_time
            
            print(f"  Total encoding time for K={current_k}: {total_encode_time:.3f}s")
            print(f"  Final solver stats: vars={bw_vars}, clauses={bw_cls}")
            
            # METRICS for position (per K)
            print(f"METRIC: k={current_k}_position_encode_time_s={pos_time:.6f}")
            print(f"METRIC: k={current_k}_position_solver_vars={pos_vars}")
            print(f"METRIC: k={current_k}_position_solver_clauses={pos_cls}")
            
            # METRICS for distance (per K) - with dynamic UB info
            print(f"METRIC: k={current_k}_distance_encode_time_s={dist_time:.6f}")
            print(f"METRIC: k={current_k}_distance_solver_vars={dist_vars}")
            print(f"METRIC: k={current_k}_distance_solver_clauses={dist_cls}")
            print(f"METRIC: k={current_k}_effective_ub={effective_ub}")
            print(f"METRIC: k={current_k}_t_vars_per_edge={2 * effective_ub}")
            
            # METRICS for bandwidth (per K)
            print(f"METRIC: k={current_k}_bandwidth_encode_time_s={bw_time:.6f}")
            print(f"METRIC: k={current_k}_bandwidth_solver_vars={bw_vars}")
            print(f"METRIC: k={current_k}_bandwidth_solver_clauses={bw_cls}")
            
            # METRICS for total (per K)
            print(f"METRIC: k={current_k}_total_encode_time_s={total_encode_time:.6f}")
            print(f"METRIC: cumulative_encode_time_s={self.cumulative_encode_time:.6f}")
            
            # Step 6: Solve
            print(f"  Solving with fresh solver...")
            solve_start = time.time()
            result = solver.solve()
            solve_time = time.time() - solve_start
            self.cumulative_solve_time += solve_time
            
            print(f"  Solve time: {solve_time:.3f}s")
            
            if result:
                # SAT - found solution
                solver_stats['sat_results'] += 1
                print(f"SATISFIABLE: K = {current_k}")
                
                # Extract model and calculate actual bandwidth
                model = solver.get_model()
                self.last_model = model
                actual_bandwidth = self.extract_actual_bandwidth(model)
                
                print(f"  Actual bandwidth from solution: {actual_bandwidth}")
                
                # METRICS: best-so-far and SAT info
                best_k_now = actual_bandwidth
                print(f"METRIC: best_k_so_far={best_k_now}")
                print(f"METRIC: sat_at_k={current_k}")
                print(f"METRIC: sat_solve_time_s={solve_time:.6f}")
                print(f"METRIC: cumulative_solve_time_s={self.cumulative_solve_time:.6f}")
                
                # Smart jumping based on actual bandwidth
                if actual_bandwidth < current_k:
                    optimal_k = actual_bandwidth
                    jump_to = actual_bandwidth - 1
                    solver_stats['smart_jumps'] += 1
                    print(f"  Smart jump: K {current_k} → {jump_to} (actual bandwidth: {actual_bandwidth})")
                    current_k = max(jump_to, 1)
                else:
                    optimal_k = current_k
                    current_k -= 1
                
            else:
                # UNSAT - K is too small
                solver_stats['unsat_results'] += 1
                print(f"UNSATISFIABLE: K = {current_k}")
                print(f"  METRIC: unsat_at_k={current_k}")
                print(f"  METRIC: unsat_solve_time_s={solve_time:.6f}")
                print(f"Optimal bandwidth found: {optimal_k}")
            
            # Step 7: Cleanup solver
            solver.delete()
            del solver
            
            # Force garbage collection
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (OSError, AttributeError):
                pass
            
            # Break if UNSAT
            if not result:
                break
        
        # Final results
        print(f"\nNON-INCREMENTAL SAT OPTIMIZATION WITH DYNAMIC UB COMPLETE")
        print(f"="*60)
        print(f"Final optimal bandwidth: {optimal_k}")
        print(f"Theoretical upper bound: {self.theoretical_ub}")
        print(f"Initial UB cutoff benefit: Eliminated {solver_stats['initial_ub_cutoff_benefit']} impossible K values")
        print(f"Dynamic UB benefit: Progressively tighter constraints at each K")
        print(f"Solver statistics:")
        print(f"  Total solve calls: {solver_stats['total_solves']}")
        print(f"  SAT results: {solver_stats['sat_results']}")
        print(f"  UNSAT results: {solver_stats['unsat_results']}")
        print(f"  Smart jumps: {solver_stats['smart_jumps']}")
        print(f"  Total encoding time: {solver_stats['total_encoding_time']:.3f}s")
        print(f"  Total solve time: {self.cumulative_solve_time:.3f}s")
        print(f"  Solver: {self.solver_type.upper()} (fresh per K + dynamic UB)")
        print(f"="*60)
        
        return optimal_k
    
    def solve_bandwidth_optimization(self, start_k=None, end_k=None):
        """
        Main solve function with non-incremental SAT and DYNAMIC UB
        
        1. Calculate theoretical upper bound as starting point
        2. Use non-incremental SAT (fresh solver per K) for optimization
        3. Apply DYNAMIC UB = K at each iteration for maximum constraint tightening
        """
        if start_k is None:
            start_k = 1
        if end_k is None:
            end_k = self.theoretical_ub
        
        # Apply theoretical UB cutoff initially
        effective_end_k = min(end_k, self.theoretical_ub)
        
        print(f"\n" + "="*80)
        print(f"2D BANDWIDTH OPTIMIZATION - NON-INCREMENTAL SAT WITH DYNAMIC UB")
        print(f"Graph: {self.n} nodes, {len(self.edges)} edges")
        print(f"Strategy: Fresh solver per K + DYNAMIC UB = K at each iteration")
        print(f"Theoretical UB (starting point): {self.theoretical_ub}")
        print(f"Testing range: K = {start_k} to {effective_end_k}")
        if effective_end_k < end_k:
            print(f"Initial cutoff applied: Original end_k {end_k} → {effective_end_k}")
        print(f"="*80)
        
        # Phase 1: Analysis
        print(f"\nPhase 1: Strategy analysis")
        print(f"Theoretical UB = {self.theoretical_ub}")
        print(f"Starting K = {effective_end_k}")
        print(f"Key innovation: At each K, encode distance constraints with UB = K")
        print(f"Benefit: Progressively fewer T variables and tighter mutual exclusions")
        
        # Phase 2: Solve with dynamic UB
        print(f"\nPhase 2: Non-incremental SAT optimization with DYNAMIC UB")
        optimal_k = self.solve_with_dynamic_ub(effective_end_k)
        
        return optimal_k


def test_dynamic_ub_solver():
    """Test the dynamic UB solver on some small graphs"""
    print("=== DYNAMIC UB SOLVER TESTS ===")
    
    # Triangle
    print(f"\n" + "="*50)
    print(f"Test 1: Triangle (Non-Incremental + Dynamic UB)")
    print(f"="*50)
    
    n1 = 3
    edges1 = [(1, 2), (2, 3), (1, 3)]
    
    solver1 = NonIncrementalBandwidthSolverDynamicUB(n1, 'glucose42')
    solver1.set_graph_edges(edges1)
    
    optimal1 = solver1.solve_bandwidth_optimization(start_k=1, end_k=4)
    print(f"Triangle result: {optimal1}")
    
    # Path
    print(f"\n" + "="*50)
    print(f"Test 2: Path (Non-Incremental + Dynamic UB)")
    print(f"="*50)
    
    n2 = 4
    edges2 = [(1, 2), (2, 3), (3, 4)]
    
    solver2 = NonIncrementalBandwidthSolverDynamicUB(n2, 'cadical195')
    solver2.set_graph_edges(edges2)
    
    optimal2 = solver2.solve_bandwidth_optimization(start_k=1, end_k=6)
    print(f"Path result: {optimal2}")
    
    # Cycle
    print(f"\n" + "="*50)
    print(f"Test 3: Cycle (Non-Incremental + Dynamic UB)")
    print(f"="*50)
    
    n3 = 5
    edges3 = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
    
    solver3 = NonIncrementalBandwidthSolverDynamicUB(n3, 'glucose42')
    solver3.set_graph_edges(edges3)
    
    optimal3 = solver3.solve_bandwidth_optimization(start_k=1, end_k=8)
    print(f"Cycle result: {optimal3}")
    
    # Performance summary
    print(f"\n" + "="*80)
    print(f"DYNAMIC UB SOLVER RESULTS SUMMARY")
    print(f"="*80)
    print(f"Triangle (3 nodes): {optimal1}")
    print(f"Path (4 nodes): {optimal2}")
    print(f"Cycle (5 nodes): {optimal3}")
    print(f"="*80)
    print(f"Strategy: Fresh solver per K + DYNAMIC UB = K at each iteration")
    print(f"Key benefits:")
    print(f"  1. Progressively tighter constraints as K decreases")
    print(f"  2. Fewer T variables at smaller K (T_1 to T_K instead of T_1 to T_theoretical_ub)")
    print(f"  3. More mutual exclusion clauses at smaller K")
    print(f"  4. Fresh solver = clean memory, no stale learnt clauses")
    print(f"="*80)


def compare_fixed_vs_dynamic_ub(n, edges, name):
    """Compare fixed UB vs dynamic UB approaches"""
    print(f"\n{'='*60}")
    print(f"COMPARISON: Fixed UB vs Dynamic UB for {name}")
    print(f"{'='*60}")
    
    theoretical_ub = calculate_theoretical_upper_bound(n)
    print(f"n = {n}, edges = {len(edges)}, theoretical UB = {theoretical_ub}")
    
    print(f"\nFixed UB approach (standard cutoff):")
    print(f"  UB is always = {theoretical_ub} for ALL K values")
    print(f"  T variables per edge per axis: {theoretical_ub}")
    print(f"  Total T variables: {len(edges) * 2 * theoretical_ub}")
    
    print(f"\nDynamic UB approach (this solver):")
    print(f"  UB = K at each iteration")
    print(f"  Example at K=2: T variables per edge per axis = 2")
    print(f"  Example at K=1: T variables per edge per axis = 1")
    print(f"  Total T variables DECREASE as K decreases")
    
    print(f"\nBenefit calculation:")
    for k in range(theoretical_ub, 0, -1):
        fixed_t_vars = len(edges) * 2 * theoretical_ub
        dynamic_t_vars = len(edges) * 2 * k
        reduction = (fixed_t_vars - dynamic_t_vars) / fixed_t_vars * 100 if fixed_t_vars > 0 else 0
        print(f"  At K={k}: Fixed has {fixed_t_vars} T-vars, Dynamic has {dynamic_t_vars} T-vars ({reduction:.1f}% reduction)")


if __name__ == '__main__':
    """
    Command line usage: python non_incremental_bandwidth_solver_dynamic_ub.py [mtx_file] [solver]
    
    Arguments:
        mtx_file: Name of MTX file (searches in mtx/group 1/, mtx/group 2/, mtx/group 3/, and mtx/regular/)
        solver: SAT solver to use (glucose42 or cadical195, default: glucose42)
    
    Examples:
        python non_incremental_bandwidth_solver_dynamic_ub.py 8.jgl009.mtx glucose42
        python non_incremental_bandwidth_solver_dynamic_ub.py 1.ash85.mtx cadical195  
        python non_incremental_bandwidth_solver_dynamic_ub.py 3.bcsstk01.mtx
        python non_incremental_bandwidth_solver_dynamic_ub.py  # Run test mode
    
    KEY INNOVATION: UB = K at each iteration (dynamic update)
    """
    
    if len(sys.argv) >= 2:
        # MTX file mode
        mtx_file = sys.argv[1]
        solver_type = sys.argv[2] if len(sys.argv) >= 3 else 'glucose42'
        
        print("=" * 80)
        print("2D BANDWIDTH OPTIMIZATION - NON-INCREMENTAL SAT WITH DYNAMIC UB")
        print("=" * 80)
        print(f"File: {mtx_file}")
        print(f"Solver: {solver_type.upper()}")
        print(f"Strategy: Fresh solver per K + DYNAMIC UB = K at each iteration")
        print(f"Key innovation: UB progressively tightens as K decreases")
        
        # Search for file in common locations
        if not os.path.exists(mtx_file):
            search_paths = [
                mtx_file,
                f"mtx/{mtx_file}",
                f"mtx/group 1/{mtx_file}",
                f"mtx/group 2/{mtx_file}",
                f"mtx/group 3/{mtx_file}",
                f"mtx/regular/{mtx_file}",
                f"sample_mtx_datasets/{mtx_file}",
                f"mtx/{mtx_file}.mtx",
                f"mtx/group 1/{mtx_file}.mtx", 
                f"mtx/group 2/{mtx_file}.mtx",
                f"mtx/group 3/{mtx_file}.mtx",
                f"mtx/regular/{mtx_file}.mtx",
                f"sample_mtx_datasets/{mtx_file}.mtx"
            ]
            
            found_file = None
            for path in search_paths:
                if os.path.exists(path):
                    found_file = path
                    break
            
            if found_file is None:
                print(f"Error: File '{mtx_file}' not found")
                print("Searched in:")
                for path in search_paths:
                    print(f"  {path}")
                
                print(f"\nAvailable files in mtx/group 1/:")
                group1_path = "mtx/group 1"
                if os.path.exists(group1_path):
                    files = [f for f in os.listdir(group1_path) if f.endswith('.mtx')]
                    for f in sorted(files):
                        print(f"  {f}")
                
                print(f"\nAvailable files in mtx/group 2/:")
                group2_path = "mtx/group 2"
                if os.path.exists(group2_path):
                    files = [f for f in os.listdir(group2_path) if f.endswith('.mtx')]
                    for f in sorted(files):
                        print(f"  {f}")
                
                print(f"\nAvailable files in mtx/group 3/:")
                group3_path = "mtx/group 3"
                if os.path.exists(group3_path):
                    files = [f for f in os.listdir(group3_path) if f.endswith('.mtx')]
                    for f in sorted(files):
                        print(f"  {f}")

                print(f"\nAvailable files in mtx/regular/:")
                regular_path = "mtx/regular"
                if os.path.exists(regular_path):
                    files = [f for f in os.listdir(regular_path) if f.endswith('.mtx')]
                    for f in sorted(files):
                        print(f"  {f}")

                print(f"\nUsage examples:")
                print(f"  python non_incremental_bandwidth_solver_dynamic_ub.py 8.jgl009.mtx glucose42")
                print(f"  python non_incremental_bandwidth_solver_dynamic_ub.py 1.ash85.mtx cadical195")
                print(f"  python non_incremental_bandwidth_solver_dynamic_ub.py 3.bcsstk01.mtx")
                sys.exit(1)
            
            mtx_file = found_file
        
        # Parse MTX file
        def parse_mtx_file(filename):
            """Parse MTX file and return n, edges"""
            print(f"Reading MTX file: {os.path.basename(filename)}")
            
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
            except FileNotFoundError:
                print(f"Error: File {filename} not found")
                return None, None
            
            header_found = False
            edges_set = set()
            n = 0
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                if line.startswith('%'):
                    continue
                
                if not header_found:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            n = int(parts[0])
                            m = int(parts[1]) if len(parts) > 1 else 0
                            print(f"Graph: {n} vertices, {m} edges (from header)")
                            header_found = True
                            continue
                    except ValueError:
                        continue
                
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        u, v = int(parts[0]), int(parts[1])
                        
                        if u != v:
                            edges_set.add((min(u, v), max(u, v)))
                
                except ValueError:
                    continue
            
            edges = list(edges_set)
            print(f"Loaded: {n} vertices, {len(edges)} edges")
            return n, edges
        
        # Parse graph
        n, edges = parse_mtx_file(mtx_file)
        if n is None or edges is None:
            print("Failed to parse MTX file")
            sys.exit(1)
        
        # Solve with dynamic UB
        print(f"\nSolving 2D bandwidth minimization with Non-Incremental SAT + DYNAMIC UB...")
        print(f"Problem: {n} vertices on {n}×{n} grid")
        print(f"Strategy: Fresh solver per K + UB = K at each iteration")
        print(f"Using: {solver_type.upper()}")
        
        solver = NonIncrementalBandwidthSolverDynamicUB(n, solver_type)
        solver.set_graph_edges(edges)
        
        start_time = time.time()
        optimal_bandwidth = solver.solve_bandwidth_optimization()
        solve_time = time.time() - start_time
        
        # Results
        print(f"\n" + "="*80)
        print(f"FINAL NON-INCREMENTAL SAT + DYNAMIC UB RESULTS")
        print(f"="*80)
        
        if optimal_bandwidth is not None:
            print(f"Optimal bandwidth: {optimal_bandwidth}")
            print(f"Total time: {solve_time:.2f}s")
            print(f"Graph: {n} vertices, {len(edges)} edges")
            print(f"Theoretical UB (starting point): {solver.theoretical_ub}")
            print(f"Strategy: Fresh solver per K + DYNAMIC UB = K")
            print(f"Solver: {solver_type.upper()}")
            print(f"Status: SUCCESS")
        else:
            print(f"No solution found")
            print(f"Total time: {solve_time:.2f}s")
            print(f"Status: FAILED")
        
        print(f"="*80)
        
    else:
        # Test mode
        print("=" * 80)
        print("2D BANDWIDTH OPTIMIZATION - NON-INCREMENTAL SAT + DYNAMIC UB TEST MODE")
        print("=" * 80)
        print("Usage: python non_incremental_bandwidth_solver_dynamic_ub.py [mtx_file] [solver]")
        print()
        print("KEY INNOVATION: UB = K at each iteration (dynamic update)")
        print()
        print("Running built-in test cases...")
        
        # Run tests
        test_dynamic_ub_solver()
        
        # Run comparison
        print(f"\n" + "="*80)
        print(f"FIXED UB vs DYNAMIC UB COMPARISON")
        print(f"="*80)
        
        compare_fixed_vs_dynamic_ub(5, [(1,2), (2,3), (3,4), (4,5), (1,5)], "Cycle-5")
        compare_fixed_vs_dynamic_ub(8, [(i, i+1) for i in range(1, 8)], "Path-8")
        
        print(f"\n" + "="*80)
        print(f"KEY FEATURES OF DYNAMIC UB APPROACH")
        print(f"="*80)
        print(f"1. DYNAMIC UB: UB = K at each iteration (not fixed theoretical UB)")
        print(f"2. Progressive Tightening: Constraints get tighter as K decreases")
        print(f"3. Variable Reduction: Fewer T variables at smaller K")
        print(f"4. Fresh Encoding: New solver + new variables per K")
        print(f"5. Smart Jumping: Use actual bandwidth to skip impossible K values")
        print(f"6. Memory Efficient: No accumulated state between K values")
        print(f"="*80)
