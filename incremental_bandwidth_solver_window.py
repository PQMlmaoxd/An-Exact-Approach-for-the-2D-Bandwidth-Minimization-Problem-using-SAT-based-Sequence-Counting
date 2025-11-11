# incremental_bandwidth_solver_window.py
# 2D Bandwidth Minimization using Incremental SAT with Window-Based Distance Encoding
# Strategy: Keep solver alive, monotonically add tightening constraints as K decreases
# Uses window-based distance encoding with lightweight T variables for maximum efficiency

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
    # Use the window-based distance encoder
    from distance_encoder_window import encode_abs_distance_window_cutoff, k_aware_reverse_clauses
    from distance_encoder_cutoff import calculate_theoretical_upper_bound
    from position_constraints import encode_all_position_constraints, create_position_variables
    print("All modules loaded OK (using window-based distance encoder)")
except ImportError as e:
    print(f"Import error: {e}")
    print("Need required modules: distance_encoder_window.py, distance_encoder_cutoff.py, position_constraints.py")
    raise ImportError("Missing required modules")


class IncrementalBandwidthSolverWindow:
    """
    2D Bandwidth Minimization solver using Incremental SAT with Window-Based Distance Encoding
    
    Key Features:
    1. Window-Based Encoding: O(n·UB) complexity instead of O(n²) for distance constraints
    2. Lightweight T variables: Only creates T variables up to UB
    3. Optional K-aware reverse clauses: Enhanced propagation for incremental solving
    4. Monotone strengthening: Persistent solver with learnt clause reuse
    5. Per-edge unique prefixes: Prevents variable conflicts
    6. Streaming approach: Memory-efficient clause generation
    
    Window-Based Distance Encoding:
    - Instead of encoding all O(n²) position pairs, use window constraints
    - For each position i: if u at i, then v must be in [i-UB, i+UB]
    - T_d indicates distance ≥ d (for d = 1..UB)
    - Forward links: (U_i ∧ V_{i-d}) → T_d establishes exact distance
    - Monotonicity: T_{d+1} → T_d ensures proper ordering
    
    Strategy:
    1. Calculate theoretical UB for window size
    2. Keep one solver alive for entire optimization process
    3. Add base constraints (position, window-based distance) once at start
    4. For each K value, only add tightening bandwidth constraints
    5. Use actual bandwidth from SAT models to jump to better K values
    6. Leverage learnt clauses across all K values for maximum performance
    
    Problem: Place n vertices on n×n grid to minimize bandwidth
    bandwidth = max{|x_u - x_v| + |y_u - y_v| : (u,v) ∈ E}
    """
    
    def __init__(self, n, solver_type='glucose42', use_k_aware=True, symmetric_window=True):
        self.n = n
        self.solver_type = solver_type
        self.vpool = IDPool()
        
        # Position variables for X,Y coordinates
        self.X_vars = {}  # X_vars[v][pos] = variable for vertex v at X position pos
        self.Y_vars = {}  # Y_vars[v][pos] = variable for vertex v at Y position pos
        
        # Distance variables with window encoding
        self.Tx_vars = {}  # T variables for X distances (per edge, up to UB only)
        self.Ty_vars = {}  # T variables for Y distances (per edge, up to UB only)
        
        self.edges = []
        self.last_model = None  # Store last successful SAT model
        
        # Window-based encoding configuration
        self.theoretical_ub = calculate_theoretical_upper_bound(n)
        self.use_k_aware = use_k_aware  # Use K-aware reverse clauses
        self.symmetric_window = symmetric_window  # Use symmetric window constraints
        
        # Incremental SAT state
        self.persistent_solver = None
        self.base_constraints_added = False
        self.current_k_constraints = set()  # Track which K constraints we've added
        
        print(f"Created incremental window solver: n={n}, using {solver_type}")
        print(f"Theoretical UB: {self.theoretical_ub}")
        print(f"Window size: {self.theoretical_ub} (per axis)")
        print(f"Strategy: Monotone strengthening + window-based encoding")
        print(f"K-aware clauses: {'Enabled' if use_k_aware else 'Disabled'}")
        print(f"Symmetric window: {'Enabled' if symmetric_window else 'Disabled'}")
        print(f"Benefits: O(n·UB) complexity, reduced clauses, faster propagation")
    
    def set_graph_edges(self, edges):
        """Set graph edges"""
        self.edges = edges
        print(f"Graph has {len(edges)} edges")
        print(f"Each edge will use window constraints with UB={self.theoretical_ub}")
    
    def create_position_variables(self):
        """Create position variables for vertices on X and Y axes"""
        self.X_vars, self.Y_vars = create_position_variables(self.n, self.vpool)
        print(f"Created position variables: {self.n} vertices × 2 axes × {self.n} positions = {2 * self.n * self.n} variables")
    
    def create_distance_variables(self):
        """
        Prepare distance variable prefixes with window-based encoding
        
        Note: Actual T variables will be created during encoding.
        This just prepares the prefixes.
        """
        print(f"Preparing distance variable prefixes with window-based encoding...")
        
        for i, (u, v) in enumerate(self.edges):
            edge_id = f'edge_{u}_{v}'
            
            # Create unique prefixes for this edge to prevent variable conflicts
            tx_prefix = f"Tx[{u},{v}]"
            ty_prefix = f"Ty[{u},{v}]"
            
            # Store prefixes for later use
            self.Tx_vars[edge_id] = {'prefix': tx_prefix, 'vars': {}}
            self.Ty_vars[edge_id] = {'prefix': ty_prefix, 'vars': {}}
        
        print(f"Prepared {len(self.edges)} edges for distance variable creation")
        print(f"Each edge will get T variables for distances 1 to {self.theoretical_ub}")
        print(f"Expected T variables per edge: {2 * self.theoretical_ub} (X + Y)")
        print(f"Total expected T variables: {len(self.edges) * 2 * self.theoretical_ub}")
    
    def encode_position_constraints(self):
        """
        Position constraints: each vertex gets exactly one position on each axis
        
        STREAMING APPROACH: Add clauses directly to solver instead of accumulating in memory
        """
        print(f"Encoding position constraints...")
        encode_start_time = time.time()
        
        # Generate constraints and stream directly to solver
        constraints = encode_all_position_constraints(self.n, self.X_vars, self.Y_vars, self.vpool)
        
        # Stream position clauses directly to solver
        clauses_added = 0
        for clause in constraints:
            self.persistent_solver.add_clause(clause)
            clauses_added += 1
        
        encode_time = time.time() - encode_start_time
        print(f"Generated {clauses_added} position constraint clauses")
        print(f"Position constraints encode time: {encode_time:.3f}s")
        print(f"Memory optimization: Position clauses streamed directly to solver")
        
        # Get solver stats after position constraints
        vars_after = self.persistent_solver.nof_vars()
        clauses_after = self.persistent_solver.nof_clauses()
        print(f"  Solver stats after position: vars={vars_after}, clauses={clauses_after}")
        
        # METRICS
        print(f"METRIC: position_encode_time_s={encode_time:.6f}")
        print(f"METRIC: position_solver_vars={vars_after}")
        print(f"METRIC: position_solver_clauses={clauses_after}")
        
        return clauses_added
    
    def encode_distance_constraints(self):
        """
        Encode distance constraints using WINDOW-BASED encoding
        
        Key difference from cutoff encoding:
        - Uses window constraints: (¬U_i ∨ V_{i-UB} ∨ ... ∨ V_{i+UB})
        - More efficient propagation with O(n·UB) complexity
        - Optional symmetric windows for bidirectional constraints
        
        STREAMING APPROACH: Generate and stream clauses directly to solver
        """
        print("Encoding distance constraints with WINDOW-BASED encoding...")
        
        encode_start_time = time.time()
        
        for edge_id, (u, v) in zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges):
            # X distance with window encoding
            tx_prefix = self.Tx_vars[edge_id]['prefix']
            tx_clauses, tx_vars = encode_abs_distance_window_cutoff(
                self.X_vars[u], self.X_vars[v],
                self.theoretical_ub, self.vpool, tx_prefix,
                add_T=True,
                add_monotonic=True,
                add_base_samepos=True,
                symmetric_window=self.symmetric_window
            )
            self.Tx_vars[edge_id]['vars'] = tx_vars
            
            # Stream X distance clauses to solver
            for c in tx_clauses:
                self.persistent_solver.add_clause(c)
            tx_clauses.clear()
            del tx_clauses
            
            # Y distance with window encoding
            ty_prefix = self.Ty_vars[edge_id]['prefix']
            ty_clauses, ty_vars = encode_abs_distance_window_cutoff(
                self.Y_vars[u], self.Y_vars[v],
                self.theoretical_ub, self.vpool, ty_prefix,
                add_T=True,
                add_monotonic=True,
                add_base_samepos=True,
                symmetric_window=self.symmetric_window
            )
            self.Ty_vars[edge_id]['vars'] = ty_vars
            
            # Stream Y distance clauses to solver
            for c in ty_clauses:
                self.persistent_solver.add_clause(c)
            ty_clauses.clear()
            del ty_clauses
        
        encode_time = time.time() - encode_start_time
        
        # Get solver stats after distance constraints
        vars_after = self.persistent_solver.nof_vars()
        clauses_after = self.persistent_solver.nof_clauses()
        
        print(f"Distance encoding time: {encode_time:.3f}s")
        print(f"Solver stats after distance: vars={vars_after}, clauses={clauses_after}")
        print(f"Window encoding: O(n·UB) = O({self.n}·{self.theoretical_ub}) per edge")
        
        # METRICS
        print(f"METRIC: distance_encode_time_s={encode_time:.6f}")
        print(f"METRIC: distance_solver_vars={vars_after}")
        print(f"METRIC: distance_solver_clauses={clauses_after}")
        
        return encode_time
    
    def _create_solver(self):
        """Create SAT solver instance"""
        if self.solver_type == 'glucose42':
            return Glucose42()
        elif self.solver_type == 'cadical195':
            return Cadical195()
        else:
            print(f"Unknown solver '{self.solver_type}', using Glucose42")
            return Glucose42()
    
    def _initialize_persistent_solver(self):
        """
        Initialize persistent solver with base constraints using window-based encoding
        
        Base constraints include:
        - Position constraints (each vertex gets one position)
        - Window-based distance constraints (O(n·UB) complexity)
        
        These constraints are K-independent and added once.
        Uses streaming approach to minimize peak RAM usage.
        """
        if self.persistent_solver is not None:
            print("Persistent solver already initialized")
            return
        
        print(f"\nInitializing persistent solver with WINDOW-BASED encoding...")
        print(f"Using {self.solver_type.upper()} with incremental interface")
        print(f"Window size (UB): {self.theoretical_ub}")
        print(f"Encoding complexity: O(n·UB) = O({self.n}·{self.theoretical_ub}) per edge")
        print(f"Streaming approach: add clauses immediately to reduce peak RAM")
        
        self.persistent_solver = self._create_solver()
        
        # Stream position constraints
        print(f"  Adding position constraints...")
        initialization_start_time = time.time()
        total_position_clauses = self.encode_position_constraints()
        print(f"    Position: {total_position_clauses} clauses")
        
        # Stream distance constraints with window encoding
        print(f"  Adding window-based distance constraints...")
        _ = self.encode_distance_constraints()
        
        total_initialization_time = time.time() - initialization_start_time
        print(f"  Total base constraints encoding time: {total_initialization_time:.3f}s")
        
        # Force garbage collection and memory trim
        print(f"  Forcing garbage collection and memory trim...")
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
        
        self.base_constraints_added = True
        print(f"Persistent solver initialized with window-based encoding")
        print(f"Base constraints added and memory optimized")
        print(f"Window encoding benefit: O(n·UB) instead of O(n²) complexity")
        
        # Final solver stats
        print(f"  Solver stats after base: "
              f"vars={self.persistent_solver.nof_vars()}, "
              f"clauses={self.persistent_solver.nof_clauses()}")
    
    def encode_bandwidth_constraints_for_k(self, K):
        """
        Encode bandwidth <= K constraints for incremental addition
        
        Returns only the NEW tightening clauses for this specific K.
        Uses monotone strengthening: never removes constraints, only adds.
        
        Optional: Add K-aware reverse clauses for enhanced propagation.
        
        For each edge: (Tx<=K) ∧ (Ty<=K) ∧ (Tx>=i → Ty<=K-i)
        """
        if K in self.current_k_constraints:
            print(f"  K={K} constraints already added (monotone strengthening)")
            return []
        
        # Apply UB cutoff
        effective_k = min(K, self.theoretical_ub)
        if effective_k != K:
            print(f"  K={K} capped at theoretical UB={self.theoretical_ub}")
        
        new_clauses = []
        edges_processed = 0
        
        print(f"  Encoding NEW bandwidth constraints for K={effective_k}...")
        encode_start_time = time.time()
        
        for edge_id in self.Tx_vars:
            tx_vars = self.Tx_vars[edge_id]['vars']  # Dict: d -> var_id
            ty_vars = self.Ty_vars[edge_id]['vars']  # Dict: d -> var_id
            edges_processed += 1
            
            # Tx <= K (i.e., not Tx >= K+1)
            if (effective_k + 1) in tx_vars:
                clause = [-tx_vars[effective_k + 1]]
                new_clauses.append(clause)
            
            # Ty <= K (i.e., not Ty >= K+1)
            if (effective_k + 1) in ty_vars:
                clause = [-ty_vars[effective_k + 1]]
                new_clauses.append(clause)
            
            # Implication constraints: Tx >= i → Ty <= K-i
            for i in range(1, effective_k + 1):
                remaining = effective_k - i
                if remaining >= 0:
                    # Tx >= i → Ty <= remaining
                    if i in tx_vars and (remaining + 1) in ty_vars:
                        clause = [-tx_vars[i], -ty_vars[remaining + 1]]
                        new_clauses.append(clause)
                    
                    # Ty >= i → Tx <= remaining (symmetry)
                    if i in ty_vars and (remaining + 1) in tx_vars:
                        clause = [-ty_vars[i], -tx_vars[remaining + 1]]
                        new_clauses.append(clause)
            
            # Optional: Add K-aware reverse clauses for this edge
            if self.use_k_aware and effective_k < self.theoretical_ub:
                # K-aware for X distance
                tx_prefix = self.Tx_vars[edge_id]['prefix']
                k_clauses_x = k_aware_reverse_clauses(
                    self.X_vars[self.edges[edges_processed - 1][0]],
                    self.X_vars[self.edges[edges_processed - 1][1]],
                    effective_k,
                    self.theoretical_ub,
                    self.vpool,
                    tx_prefix,
                    symmetric=self.symmetric_window
                )
                new_clauses.extend(k_clauses_x)
                
                # K-aware for Y distance
                ty_prefix = self.Ty_vars[edge_id]['prefix']
                k_clauses_y = k_aware_reverse_clauses(
                    self.Y_vars[self.edges[edges_processed - 1][0]],
                    self.Y_vars[self.edges[edges_processed - 1][1]],
                    effective_k,
                    self.theoretical_ub,
                    self.vpool,
                    ty_prefix,
                    symmetric=self.symmetric_window
                )
                new_clauses.extend(k_clauses_y)
        
        # Mark this K as processed
        self.current_k_constraints.add(K)
        
        encode_time = time.time() - encode_start_time
        print(f"  Generated {len(new_clauses)} NEW bandwidth clauses for {edges_processed} edges, K={effective_k}")
        if self.use_k_aware:
            print(f"  Includes K-aware reverse clauses for enhanced propagation")
        print(f"  Bandwidth constraints (K={effective_k}) encode time: {encode_time:.3f}s")
        
        return new_clauses
    
    def _extract_positions_from_model(self, model):
        """Extract vertex positions from SAT solution"""
        posset = {lit for lit in model if lit > 0}
        
        positions = {}
        for v in range(1, self.n + 1):
            # Find X position
            for pos in range(1, self.n + 1):
                if self.X_vars[v][pos-1] in posset:
                    positions[f'X_{v}'] = pos
                    break
            
            # Find Y position
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
    
    def solve_with_incremental_sat(self, upper_bound):
        """
        Main incremental SAT solving with window-based encoding
        
        Strategy:
        1. Apply theoretical UB for window size
        2. Start from min(upper_bound, theoretical_ub), test K values going down
        3. If SAT at K with actual bandwidth Y < K, jump to K = Y-1
        4. If UNSAT at K, optimal is previous K
        5. Use persistent solver + learnt clauses for maximum efficiency
        6. Leverage window-based encoding for faster propagation
        """
        print(f"\nINCREMENTAL SAT OPTIMIZATION WITH WINDOW-BASED ENCODING")
        print(f"Strategy: Monotone strengthening + window-based distance encoding")
        print(f"Window size (UB): {self.theoretical_ub}")
        print(f"Solver: {self.solver_type.upper()} with persistent incremental interface")
        
        # Initialize persistent solver with base constraints
        self._initialize_persistent_solver()
        
        # Apply theoretical upper bound
        current_k = min(upper_bound, self.theoretical_ub)
        
        if current_k < upper_bound:
            print(f"UB applied: Adjusted upper bound from {upper_bound} to {current_k}")
        
        optimal_k = None
        solver_stats = {
            'total_solves': 0,
            'sat_results': 0,
            'unsat_results': 0,
            'smart_jumps': 0,
            'clauses_added': 0,
            'ub_cutoff_benefit': upper_bound - current_k if upper_bound > current_k else 0
        }
        
        # Cumulative solve time tracking
        if not hasattr(self, '_cum_solve_time'):
            self._cum_solve_time = 0.0
        
        while current_k >= 1:
            print(f"\nTesting K = {current_k}")
            solver_stats['total_solves'] += 1
            
            # Add bandwidth constraints for this K
            print(f"  Adding constraints for K={current_k}...")
            bandwidth_clauses = self.encode_bandwidth_constraints_for_k(current_k)
            
            # Stream K-specific constraints to persistent solver
            for clause in bandwidth_clauses:
                self.persistent_solver.add_clause(clause)
            
            # Clear bandwidth clauses to reduce memory
            clause_count = len(bandwidth_clauses)
            solver_stats['clauses_added'] += clause_count
            bandwidth_clauses.clear()
            del bandwidth_clauses
            
            # Solver stats after adding K constraints
            print(f"  Solver stats now: "
                  f"vars={self.persistent_solver.nof_vars()}, "
                  f"clauses={self.persistent_solver.nof_clauses()}")
            
            # Solve with current constraints
            print(f"  Solving with persistent solver...")
            solve_start = time.time()
            result = self.persistent_solver.solve()
            solve_time = time.time() - solve_start
            
            print(f"  Solve time: {solve_time:.3f}s")
            
            if result:
                # SAT - found solution
                solver_stats['sat_results'] += 1
                print(f"SATISFIABLE: K = {current_k}")
                
                # Extract model and calculate actual bandwidth
                model = self.persistent_solver.get_model()
                self.last_model = model
                actual_bandwidth = self.extract_actual_bandwidth(model)
                
                print(f"  Actual bandwidth from solution: {actual_bandwidth}")
                
                # METRICS
                best_k_now = actual_bandwidth
                print(f"METRIC: best_k_so_far={best_k_now}")
                print(f"METRIC: sat_at_k={current_k}")
                print(f"METRIC: sat_solve_time_s={solve_time:.6f}")
                
                # Cumulative solve time
                self._cum_solve_time += solve_time
                print(f"METRIC: cumulative_solve_time_s={self._cum_solve_time:.6f}")
                
                # Verify bandwidth
                if actual_bandwidth > self.theoretical_ub:
                    print(f"  WARNING: Bandwidth {actual_bandwidth} > theoretical UB {self.theoretical_ub}")
                
                # Smart jumping
                if actual_bandwidth < current_k:
                    optimal_k = current_k
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
                print(f"Optimal bandwidth found: {optimal_k}")
                break
        
        # Final results
        print(f"\nINCREMENTAL SAT OPTIMIZATION WITH WINDOW-BASED ENCODING COMPLETE")
        print(f"="*60)
        print(f"Final optimal bandwidth: {optimal_k}")
        print(f"Window size (UB): {self.theoretical_ub}")
        print(f"Solver statistics:")
        print(f"  Total solve calls: {solver_stats['total_solves']}")
        print(f"  SAT results: {solver_stats['sat_results']}")
        print(f"  UNSAT results: {solver_stats['unsat_results']}")
        print(f"  Smart jumps: {solver_stats['smart_jumps']}")
        print(f"  Total clauses added: {solver_stats['clauses_added']}")
        print(f"  Solver: {self.solver_type.upper()} (persistent + window encoding)")
        print(f"  K-aware clauses: {'Enabled' if self.use_k_aware else 'Disabled'}")
        print(f"="*60)
        
        return optimal_k
    
    def cleanup_solver(self):
        """Clean up persistent solver"""
        if self.persistent_solver is not None:
            print(f"Cleaning up persistent solver...")
            self.persistent_solver.delete()
            self.persistent_solver = None
            self.base_constraints_added = False
            self.current_k_constraints.clear()
    
    def solve_bandwidth_optimization(self, start_k=None, end_k=None):
        """
        Main solve function with incremental SAT and window-based encoding
        
        1. Calculate theoretical upper bound for window size
        2. Use incremental SAT with monotone strengthening
        3. Apply window-based encoding for O(n·UB) complexity
        """
        if start_k is None:
            start_k = 1
        if end_k is None:
            end_k = self.theoretical_ub
        
        # Apply theoretical UB
        effective_end_k = min(end_k, self.theoretical_ub)
        
        print(f"\n" + "="*80)
        print(f"2D BANDWIDTH OPTIMIZATION - INCREMENTAL SAT WITH WINDOW ENCODING")
        print(f"Graph: {self.n} nodes, {len(self.edges)} edges")
        print(f"Strategy: Monotone strengthening + window-based distance encoding")
        print(f"Window size (UB): {self.theoretical_ub}")
        print(f"Testing range: K = {start_k} to {effective_end_k}")
        if effective_end_k < end_k:
            print(f"UB applied: Original end_k {end_k} → {effective_end_k}")
        print(f"K-aware clauses: {'Enabled' if self.use_k_aware else 'Disabled'}")
        print(f"Symmetric window: {'Enabled' if self.symmetric_window else 'Disabled'}")
        print(f"="*80)
        
        try:
            # Phase 1: Theoretical upper bound analysis
            print(f"\nPhase 1: Theoretical upper bound analysis")
            print(f"Theoretical UB = {self.theoretical_ub}")
            print(f"Using UB = {effective_end_k}")
            print(f"Window encoding: O(n·UB) = O({self.n}·{self.theoretical_ub}) per edge")
            
            # Phase 2: Incremental SAT optimization with window encoding
            print(f"\nPhase 2: Incremental SAT optimization with window-based encoding")
            optimal_k = self.solve_with_incremental_sat(effective_end_k)
            
            return optimal_k
            
        finally:
            # Always cleanup solver
            self.cleanup_solver()


def test_incremental_window_solver():
    """Test the incremental window solver on some small graphs"""
    print("=== INCREMENTAL WINDOW SOLVER TESTS ===")
    
    # Triangle
    print(f"\n" + "="*50)
    print(f"Test 1: Triangle (Incremental SAT + Window Encoding)")
    print(f"="*50)
    
    n1 = 3
    edges1 = [(1, 2), (2, 3), (1, 3)]
    
    solver1 = IncrementalBandwidthSolverWindow(n1, 'glucose42', use_k_aware=False)
    solver1.set_graph_edges(edges1)
    solver1.create_position_variables()
    solver1.create_distance_variables()
    
    optimal1 = solver1.solve_bandwidth_optimization(start_k=1, end_k=4)
    print(f"Triangle result: {optimal1}")
    
    # Path
    print(f"\n" + "="*50)
    print(f"Test 2: Path (Incremental SAT + Window Encoding)")
    print(f"="*50)
    
    n2 = 4
    edges2 = [(1, 2), (2, 3), (3, 4)]
    
    solver2 = IncrementalBandwidthSolverWindow(n2, 'cadical195', use_k_aware=True)
    solver2.set_graph_edges(edges2)
    solver2.create_position_variables()
    solver2.create_distance_variables()
    
    optimal2 = solver2.solve_bandwidth_optimization(start_k=1, end_k=6)
    print(f"Path result: {optimal2}")
    
    # Cycle
    print(f"\n" + "="*50)
    print(f"Test 3: Cycle (Incremental SAT + Window Encoding)")
    print(f"="*50)
    
    n3 = 5
    edges3 = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
    
    solver3 = IncrementalBandwidthSolverWindow(n3, 'glucose42', use_k_aware=True, symmetric_window=True)
    solver3.set_graph_edges(edges3)
    solver3.create_position_variables()
    solver3.create_distance_variables()
    
    optimal3 = solver3.solve_bandwidth_optimization(start_k=1, end_k=8)
    print(f"Cycle result: {optimal3}")
    
    # Results summary
    print(f"\n" + "="*80)
    print(f"INCREMENTAL SAT WITH WINDOW ENCODING RESULTS SUMMARY")
    print(f"="*80)
    print(f"Triangle (3 nodes): {optimal1}")
    print(f"Path (4 nodes): {optimal2}")
    print(f"Cycle (5 nodes): {optimal3}")
    print(f"="*80)
    print(f"Strategy: Monotone strengthening + window-based encoding")
    print(f"Benefits: O(n·UB) complexity, learnt clauses reuse, smart jumping,")
    print(f"          enhanced propagation, reduced clause count")
    print(f"="*80)


if __name__ == '__main__':
    """
    Command line usage: python incremental_bandwidth_solver_window.py [mtx_file] [solver] [--k-aware] [--symmetric]
    
    Arguments:
        mtx_file: Name of MTX file
        solver: SAT solver to use (glucose42 or cadical195, default: glucose42)
        --k-aware: Enable K-aware reverse clauses (optional)
        --symmetric: Enable symmetric window constraints (optional)
    
    Examples:
        python incremental_bandwidth_solver_window.py 8.jgl009.mtx glucose42
        python incremental_bandwidth_solver_window.py 1.ash85.mtx cadical195 --k-aware
        python incremental_bandwidth_solver_window.py 3.bcsstk01.mtx glucose42 --k-aware --symmetric
        python incremental_bandwidth_solver_window.py  # Run test mode
    """
    
    # Check if MTX file provided
    if len(sys.argv) >= 2:
        # MTX file mode
        mtx_file = sys.argv[1]
        solver_type = 'glucose42'
        use_k_aware = False
        symmetric_window = False
        
        # Parse arguments
        for i in range(2, len(sys.argv)):
            arg = sys.argv[i]
            if arg in ['glucose42', 'cadical195']:
                solver_type = arg
            elif arg == '--k-aware':
                use_k_aware = True
            elif arg == '--symmetric':
                symmetric_window = True
        
        print("=" * 80)
        print("2D BANDWIDTH OPTIMIZATION - INCREMENTAL SAT WITH WINDOW ENCODING")
        print("=" * 80)
        print(f"File: {mtx_file}")
        print(f"Solver: {solver_type.upper()}")
        print(f"Strategy: Monotone strengthening + window-based distance encoding")
        print(f"K-aware clauses: {'Enabled' if use_k_aware else 'Disabled'}")
        print(f"Symmetric window: {'Enabled' if symmetric_window else 'Disabled'}")
        
        # Search for file
        if not os.path.exists(mtx_file):
            search_paths = [
                mtx_file,
                f"mtx/{mtx_file}",
                f"mtx/group 1/{mtx_file}",
                f"mtx/group 2/{mtx_file}",
                f"mtx/group 3/{mtx_file}",
                f"mtx/regular/{mtx_file}",
                f"mtx/{mtx_file}.mtx",
                f"mtx/group 1/{mtx_file}.mtx",
                f"mtx/group 2/{mtx_file}.mtx",
                f"mtx/group 3/{mtx_file}.mtx",
                f"mtx/regular/{mtx_file}.mtx"
            ]
            
            found_file = None
            for path in search_paths:
                if os.path.exists(path):
                    found_file = path
                    break
            
            if found_file is None:
                print(f"Error: File '{mtx_file}' not found")
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
            
            for line in lines:
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
        
        # Solve with window-based encoding
        print(f"\nSolving 2D bandwidth minimization with Window-Based Encoding...")
        print(f"Problem: {n} vertices on {n}×{n} grid")
        print(f"Strategy: Incremental SAT + window-based distance encoding")
        print(f"Using: {solver_type.upper()}")
        
        solver = IncrementalBandwidthSolverWindow(n, solver_type, use_k_aware, symmetric_window)
        solver.set_graph_edges(edges)
        solver.create_position_variables()
        solver.create_distance_variables()
        
        start_time = time.time()
        optimal_bandwidth = solver.solve_bandwidth_optimization()
        solve_time = time.time() - start_time
        
        # Results
        print(f"\n" + "="*80)
        print(f"FINAL INCREMENTAL SAT + WINDOW ENCODING RESULTS")
        print(f"="*80)
        
        if optimal_bandwidth is not None:
            print(f"Optimal bandwidth: {optimal_bandwidth}")
            print(f"Total solve time: {solve_time:.2f}s")
            print(f"Graph: {n} vertices, {len(edges)} edges")
            print(f"Window size (UB): {solver.theoretical_ub}")
            print(f"Strategy: Incremental + window-based encoding")
            print(f"Solver: {solver_type.upper()}")
            print(f"K-aware: {'Yes' if use_k_aware else 'No'}")
            print(f"Symmetric: {'Yes' if symmetric_window else 'No'}")
            print(f"Status: SUCCESS")
        else:
            print(f"No solution found")
            print(f"Total solve time: {solve_time:.2f}s")
            print(f"Status: FAILED")
        
        print(f"="*80)
        
    else:
        # Test mode
        print("=" * 80)
        print("2D BANDWIDTH OPTIMIZATION - INCREMENTAL SAT + WINDOW ENCODING TEST MODE")
        print("=" * 80)
        print("Usage: python incremental_bandwidth_solver_window.py [mtx_file] [solver] [--k-aware] [--symmetric]")
        print()
        print("Running built-in window-based encoding test cases...")
        
        test_incremental_window_solver()
        
        print(f"\n" + "="*80)
        print(f"KEY FEATURES OF WINDOW-BASED ENCODING")
        print(f"="*80)
        print(f"1. Window Constraints: O(n·UB) instead of O(n²) complexity")
        print(f"2. Lightweight T Variables: Only up to theoretical UB")
        print(f"3. Enhanced Propagation: Faster unit propagation with window structure")
        print(f"4. K-aware Clauses: Optional tighter constraints for each K (--k-aware)")
        print(f"5. Symmetric Windows: Bidirectional constraints for stronger propagation (--symmetric)")
        print(f"6. Incremental Solving: Learnt clauses reused across K values")
        print(f"7. Smart Jumping: Skip impossible K values based on SAT solutions")
        print(f"="*80)
