# incremental_bandwidth_solver_cutoff.py
# 2D Bandwidth Minimization using Incremental SAT with UB Cutoff Optimization
# Strategy: Keep solver alive, monotonically add tightening constraints as K decreases
# Uses optimized distance encoding with theoretical UB cutoff for maximum efficiency

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
    # Use the new cutoff-optimized distance encoder
    from distance_encoder_cutoff import encode_abs_distance_cutoff, calculate_theoretical_upper_bound
    from position_constraints import encode_all_position_constraints, create_position_variables
    print("All modules loaded OK (using cutoff-optimized distance encoder)")
except ImportError as e:
    print(f"Import error: {e}")
    print("Need required modules: distance_encoder_cutoff.py, position_constraints.py")
    raise ImportError("Missing required modules")


class IncrementalBandwidthSolverCutoff:
    """
    2D Bandwidth Minimization solver using Incremental SAT with UB Cutoff Optimization
    
    Key Improvements over standard incremental solver:
    1. UB Cutoff Optimization: Uses theoretical upper bound to eliminate impossible distance assignments
    2. Lightweight T variables: Only creates T variables up to UB, not full range 1..n-1
    3. Direct mutual exclusion: Efficient 2-literal clauses for distance > UB constraints
    4. Per-edge unique prefixes: Prevents variable conflicts when using global vpool
    5. Optimized clause generation: O(n × UB) instead of O(n²) complexity
    
    Strategy:
    1. Calculate theoretical UB = min{2⌈(√(2n-1)-1)/2⌉, 2⌈√(n/2)⌉-1}
    2. Keep one solver alive for entire optimization process  
    3. Add base constraints (position, distance with UB cutoff) once at start
    4. For each K value, only add tightening bandwidth constraints
    5. Use actual bandwidth from SAT models to jump to better K values
    6. Leverage learnt clauses across all K values for maximum performance
    
    Problem: Place n vertices on n×n grid to minimize bandwidth
    bandwidth = max{|x_u - x_v| + |y_u - y_v| : (u,v) ∈ E}
    """
    
    def __init__(self, n, solver_type='glucose42'):
        self.n = n
        self.solver_type = solver_type
        self.vpool = IDPool()
        
        # Position variables for X,Y coordinates
        self.X_vars = {}  # X_vars[v][pos] = variable for vertex v at X position pos
        self.Y_vars = {}  # Y_vars[v][pos] = variable for vertex v at Y position pos
        
        # Distance variables with cutoff optimization
        self.Tx_vars = {}  # T variables for X distances (per edge, up to UB only)
        self.Ty_vars = {}  # T variables for Y distances (per edge, up to UB only)
        
        self.edges = []
        self.last_model = None  # Store last successful SAT model
        
        # Cutoff optimization
        self.theoretical_ub = calculate_theoretical_upper_bound(n)
        
        # Incremental SAT state
        self.persistent_solver = None
        self.base_constraints_added = False
        self.current_k_constraints = set()  # Track which K constraints we've added
        
        print(f"Created incremental cutoff solver: n={n}, using {solver_type}")
        print(f"Theoretical UB: {self.theoretical_ub}")
        print(f"Strategy: Monotone strengthening + UB cutoff optimization")
        print(f"Benefits: Reduced variables, faster encoding, early infeasible elimination")
    
    def set_graph_edges(self, edges):
        """Set graph edges"""
        self.edges = edges
        print(f"Graph has {len(edges)} edges")
        print(f"Each edge will use distance constraints up to UB={self.theoretical_ub}")
    
    def create_position_variables(self):
        """Create position variables for vertices on X and Y axes"""
        self.X_vars, self.Y_vars = create_position_variables(self.n, self.vpool)
        print(f"Created position variables: {self.n} vertices × 2 axes × {self.n} positions = {2 * self.n * self.n} variables")
    
    def create_distance_variables(self):
        """
        Create T variables for edge distances with UB cutoff optimization
        
        Key improvements:
        - Only creates T variables up to theoretical UB (not n-1)
        - Uses unique prefixes per edge to prevent conflicts
        - Significantly reduces total variable count
        """
        total_tx_vars = 0
        total_ty_vars = 0
        
        print(f"Creating distance variables with UB cutoff optimization...")
        
        for i, (u, v) in enumerate(self.edges):
            edge_id = f'edge_{u}_{v}'
            
            # Create unique prefixes for this edge to prevent variable conflicts
            tx_prefix = f"Tx[{u},{v}]"
            ty_prefix = f"Ty[{u},{v}]"
            
            # Variables will be created by encode_abs_distance_cutoff
            # We just store the prefixes for later use
            self.Tx_vars[edge_id] = {'prefix': tx_prefix, 'vars': {}}
            self.Ty_vars[edge_id] = {'prefix': ty_prefix, 'vars': {}}
        
        print(f"Prepared {len(self.edges)} edges for distance variable creation")
        print(f"Each edge will get T variables for distances 1 to {self.theoretical_ub}")
        print(f"Expected T variables per edge: {2 * self.theoretical_ub} (X + Y)")
        print(f"Total expected T variables: {len(self.edges) * 2 * self.theoretical_ub}")
        print(f"Reduction vs standard encoding: ~{len(self.edges) * 2 * (self.n - 1 - self.theoretical_ub)} fewer variables")
    
    def encode_position_constraints(self):
        """
        Position constraints: each vertex gets exactly one position on each axis
        Each position can have at most one vertex
        Uses Sequential Counter encoding for O(n²) complexity
        
        STREAMING APPROACH: Add clauses directly to solver instead of accumulating in memory
        """
        print(f"Encoding position constraints...")
        encode_start_time = time.time()
        
        # Generate constraints and stream directly to solver
        constraints = encode_all_position_constraints(self.n, self.X_vars, self.Y_vars, self.vpool)
        
        # Stream position clauses directly to solver (no intermediate storage)
        clauses_added = 0
        for clause in constraints:
            self.persistent_solver.add_clause(clause)
            clauses_added += 1
        
        encode_time = time.time() - encode_start_time
        print(f"Generated {clauses_added} position constraint clauses")
        print(f"Position constraints encode time: {encode_time:.3f}s")
        print(f"Memory optimization: Position clauses streamed directly to solver")
        
        # Add solver stats after position constraints
        vars_after = self.persistent_solver.nof_vars()
        clauses_after = self.persistent_solver.nof_clauses()
        print(f"  Solver stats after position: vars={vars_after}, clauses={clauses_after}")
        
        # >>> THÊM 3 DÒNG METRIC CHUẨN HOÁ
        print(f"METRIC: position_encode_time_s={encode_time:.6f}")
        print(f"METRIC: position_solver_vars={vars_after}")
        print(f"METRIC: position_solver_clauses={clauses_after}")
        
        # Return count only, not clause list
        return clauses_added
    
    def encode_distance_constraints(self, current_k: int):
        """
        Encode distance constraints with K-specific UB cutoff optimization.
        
        CRITICAL IMPROVEMENT: Uses effective_ub = min(current_k, theoretical_ub) to optimize
        for the specific K being tested, not the global theoretical UB.
        
        Args:
            current_k: Current bandwidth being tested (used for K-specific optimization)
        
        Returns:
            Encoding time in seconds
        """
        print("Encoding distance constraints with K-specific UB cutoff optimization...")
        
        # CRITICAL FIX: Use K-specific UB for optimal performance
        effective_ub = min(current_k, self.theoretical_ub)
        
        print(f"  K-specific UB: {effective_ub} (K={current_k}, theoretical={self.theoretical_ub})")
        if effective_ub < self.theoretical_ub:
            reduction = ((self.theoretical_ub - effective_ub) / self.theoretical_ub * 100)
            print(f"  Optimization: Using UB={effective_ub} instead of {self.theoretical_ub} ({reduction:.1f}% reduction)")
        
        # time start
        t0 = time.time()

        for edge_id, (u, v) in zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges):
            # X distance with K-specific UB
            tx_prefix = self.Tx_vars[edge_id]['prefix']
            tx_clauses, tx_vars = encode_abs_distance_cutoff(
                self.X_vars[u], self.X_vars[v],
                effective_ub, self.vpool, tx_prefix  # ← FIXED: use effective_ub
            )
            self.Tx_vars[edge_id]['vars'] = tx_vars
            for c in tx_clauses:
                self.persistent_solver.add_clause(c)
            tx_clauses.clear(); del tx_clauses

            # Y distance with K-specific UB
            ty_prefix = self.Ty_vars[edge_id]['prefix']
            ty_clauses, ty_vars = encode_abs_distance_cutoff(
                self.Y_vars[u], self.Y_vars[v],
                effective_ub, self.vpool, ty_prefix  # ← FIXED: use effective_ub
            )
            self.Ty_vars[edge_id]['vars'] = ty_vars
            for c in ty_clauses:
                self.persistent_solver.add_clause(c)
            ty_clauses.clear(); del ty_clauses

        enc_time = time.time() - t0

        # absolute stats from solver (no deltas)
        vars_after = self.persistent_solver.nof_vars()
        cls_after  = self.persistent_solver.nof_clauses()

        print(f"Distance encoding time: {enc_time:.3f}s")
        print(f"Solver stats after distance: vars={vars_after}, clauses={cls_after}")

        # Metrics
        print(f"METRIC: distance_encode_time_s={enc_time:.6f}")
        print(f"METRIC: distance_solver_vars={vars_after}")
        print(f"METRIC: distance_solver_clauses={cls_after}")
        print(f"METRIC: effective_ub_used={effective_ub}")

        return enc_time
    
    def _create_solver(self):
        """Create SAT solver instance"""
        if self.solver_type == 'glucose42':
            return Glucose42()
        elif self.solver_type == 'cadical195':
            return Cadical195()
        else:
            print(f"Unknown solver '{self.solver_type}', using Glucose42")
            return Glucose42()
    
    def _initialize_persistent_solver(self, initial_k: int):
        """
        Initialize persistent solver with base constraints including K-specific UB cutoff optimization.
        
        CRITICAL CHANGE: Now accepts initial_k parameter to optimize distance encoding
        for the specific K being tested, not just the global theoretical UB.
        
        Base constraints include:
        - Position constraints (each vertex gets one position)
        - Distance constraints with K-specific UB cutoff (eliminates distance > min(K, theoretical_ub))
        
        These constraints are K-specific and added once at the start of optimization.
        Uses streaming approach to minimize peak RAM usage.
        
        Args:
            initial_k: Initial K value for optimization (used to set effective UB)
        """
        if self.persistent_solver is not None:
            print("Persistent solver already initialized")
            return
        
        effective_ub = min(initial_k, self.theoretical_ub)
        
        print(f"\nInitializing persistent solver with K-specific UB cutoff optimization...")
        print(f"Using {self.solver_type.upper()} with incremental interface")
        print(f"Initial K: {initial_k}")
        print(f"Theoretical UB: {self.theoretical_ub}")
        print(f"Effective UB for encoding: {effective_ub}")
        print(f"Streaming approach: add clauses immediately to reduce peak RAM")
        
        self.persistent_solver = self._create_solver()
        
        # Stream position constraints (add directly, no intermediate list)
        print(f"  Adding position constraints...")
        initialization_start_time = time.time()
        total_position_clauses = self.encode_position_constraints()
        print(f"    Position: {total_position_clauses} clauses")
        
        # Stream distance constraints with K-specific UB cutoff
        print(f"  Adding distance constraints with K-specific UB cutoff...")
        _ = self.encode_distance_constraints(initial_k)  # ← FIXED: pass initial_k
        
        total_initialization_time = time.time() - initialization_start_time
        print(f"  Total base constraints encoding time: {total_initialization_time:.3f}s")
        
        # Force garbage collection and memory trim to reduce peak RAM
        print(f"  Forcing garbage collection and memory trim...")
        gc.collect()
        try:
            # Try to trim memory on Linux systems
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            # Not available on all systems, silently continue
            pass
        
        self.base_constraints_added = True
        print(f"Persistent solver initialized with K-specific UB cutoff optimization")
        print(f"Base constraints added and memory optimized")
        print(f"K-specific cutoff benefit: Eliminated all distance assignments > {effective_ub}")
        
        # Add final solver stats after all base constraints
        print(f"  Solver stats after base: "
              f"vars={self.persistent_solver.nof_vars()}, "
              f"clauses={self.persistent_solver.nof_clauses()}")
    
    def encode_bandwidth_constraints_for_k(self, K):
        """
        Encode bandwidth <= K constraints for incremental addition with cutoff optimization
        
        Returns only the NEW tightening clauses for this specific K.
        Uses monotone strengthening: never removes constraints, only adds.
        
        Key improvement: Only works with T variables up to min(K, theoretical_ub),
        significantly reducing constraint complexity for small K values.
        
        For each edge: (Tx<=K) ∧ (Ty<=K) ∧ (Tx>=i → Ty<=K-i)
        """
        if K in self.current_k_constraints:
            print(f"  K={K} constraints already added (monotone strengthening)")
            return []
        
        # Apply UB cutoff: no point in adding constraints for K > theoretical_ub
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
                    # Equivalent: ¬Tx_i ∨ ¬Ty_{remaining+1}
                    if i in tx_vars and (remaining + 1) in ty_vars:
                        clause = [-tx_vars[i], -ty_vars[remaining + 1]]
                        new_clauses.append(clause)
                    
                    # Ty >= i → Tx <= remaining (symmetry)
                    # Equivalent: ¬Ty_i ∨ ¬Tx_{remaining+1}
                    if i in ty_vars and (remaining + 1) in tx_vars:
                        clause = [-ty_vars[i], -tx_vars[remaining + 1]]
                        new_clauses.append(clause)
        
        # Mark this K as processed
        self.current_k_constraints.add(K)
        
        encode_time = time.time() - encode_start_time
        print(f"  Generated {len(new_clauses)} NEW bandwidth clauses for {edges_processed} edges, K={effective_k}")
        print(f"  Bandwidth constraints (K={effective_k}) encode time: {encode_time:.3f}s")
        print(f"  Cutoff benefit: Working with T-vars up to UB={self.theoretical_ub} only")
        
        return new_clauses
    
    def _extract_positions_from_model(self, model):
        """
        Extract vertex positions from SAT solution
        
        Optimized: Use set for O(1) lookup instead of O(m) list operations.
        Complexity: O(m + n²) instead of O(n²m) where m = |model|, n = vertices
        """
        # Create set of positive literals for O(1) lookup
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
    
    def _print_solution_details(self, positions, edge_distances, bandwidth, K):
        """Show solution details (summary only)"""
        print(f"Solution summary:")
        print(f"  Vertices placed: {len([v for v in range(1, self.n + 1) if f'X_{v}' in positions])}/{self.n}")
        print(f"  Edges evaluated: {len(edge_distances)}")
        print(f"  Bandwidth: {bandwidth} (constraint: {K})")
        print(f"  Valid: {'Yes' if bandwidth <= K else 'No'}")
        print(f"  Within theoretical UB: {'Yes' if bandwidth <= self.theoretical_ub else 'No'}")
    
    def extract_and_verify_solution(self, model, K):
        """Extract solution and check if it satisfies K constraint"""
        print(f"--- Verifying solution for K={K} ---")
        
        positions = self._extract_positions_from_model(model)
        bandwidth, edge_distances = self._calculate_bandwidth(positions)
        self._print_solution_details(positions, edge_distances, bandwidth, K)
        
        return bandwidth <= K, bandwidth  # Return both validity and actual bandwidth
    
    def extract_actual_bandwidth(self, model):
        """Extract actual bandwidth from SAT solution without printing details"""
        positions = self._extract_positions_from_model(model)
        bandwidth, _ = self._calculate_bandwidth(positions)
        return bandwidth
    
    def solve_with_incremental_sat(self, upper_bound):
        """
        Main incremental SAT solving with monotone strengthening and K-specific UB cutoff optimization
        
        Strategy:
        1. Apply K-specific UB cutoff from the start (effective_ub = min(K, theoretical_ub))
        2. Start from min(upper_bound, theoretical_ub), test K values going down
        3. If SAT at K with actual bandwidth Y < K, jump to K = Y-1  
        4. If UNSAT at K, optimal is previous K
        5. Use persistent solver + learnt clauses for maximum efficiency
        6. Leverage K-specific UB cutoff to eliminate impossible assignments early
        
        CRITICAL CHANGE: Now passes initial_k to solver initialization to optimize
        distance encoding for the specific K being tested, not just the global theoretical UB.
        """
        print(f"\nINCREMENTAL SAT OPTIMIZATION WITH K-SPECIFIC UB CUTOFF")
        print(f"Strategy: Monotone strengthening + K-specific UB cutoff optimization")
        print(f"Theoretical UB: {self.theoretical_ub}")
        print(f"Solver: {self.solver_type.upper()} with persistent incremental interface")
        
        # Apply theoretical upper bound cutoff immediately
        current_k = min(upper_bound, self.theoretical_ub)
        
        if current_k < upper_bound:
            print(f"UB cutoff applied: Adjusted upper bound from {upper_bound} to {current_k}")
        
        # Initialize persistent solver with K-specific base constraints
        # CRITICAL FIX: Pass initial K to optimize distance encoding
        self._initialize_persistent_solver(current_k)  # ← FIXED: pass initial_k
        
        optimal_k = None
        solver_stats = {
            'total_solves': 0,
            'sat_results': 0,
            'unsat_results': 0,
            'smart_jumps': 0,
            'clauses_added': 0,
            'ub_cutoff_benefit': upper_bound - current_k if upper_bound > current_k else 0,
            'effective_ub_used': min(current_k, self.theoretical_ub),
            'k_specific_optimization': True
        }
        
        print(f"K-specific cutoff benefit: Distance encoding optimized for max distance = {min(current_k, self.theoretical_ub)}")
        
        while current_k >= 1:
            print(f"\nTesting K = {current_k}")
            solver_stats['total_solves'] += 1
            
            # Add bandwidth constraints for this K (monotone strengthening)
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
            
            # Add solver stats after adding K constraints
            print(f"  Solver stats now: "
                  f"vars={self.persistent_solver.nof_vars()}, "
                  f"clauses={self.persistent_solver.nof_clauses()}")
            
            # Get solver statistics
            solve_start = time.time()
            
            # Solve with current constraints (incremental)
            print(f"  Solving with persistent solver...")
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
                
                # === METRIC: best-so-far ===
                # Lấy best K mới (ở đây chính là actual_bandwidth)
                best_k_now = actual_bandwidth
                # In metric để runner có thể parse liên tục
                print(f"METRIC: best_k_so_far={best_k_now}")
                print(f"METRIC: sat_at_k={current_k}")
                print(f"METRIC: sat_solve_time_s={solve_time:.6f}")
                
                # Cumulative solve time tracking
                if not hasattr(self, '_cum_solve_time'):
                    self._cum_solve_time = 0.0
                self._cum_solve_time += solve_time
                print(f"METRIC: cumulative_solve_time_s={self._cum_solve_time:.6f}")
                
                # Verify bandwidth is within theoretical bounds
                if actual_bandwidth > self.theoretical_ub:
                    print(f"  WARNING: Bandwidth {actual_bandwidth} > theoretical UB {self.theoretical_ub}")
                
                # Smart jumping based on actual bandwidth
                if actual_bandwidth < current_k:
                    optimal_k = actual_bandwidth
                    jump_to = actual_bandwidth - 1
                    solver_stats['smart_jumps'] += 1
                    print(f"  Smart jump: K {current_k} → {jump_to} (actual bandwidth: {actual_bandwidth})")
                    current_k = max(jump_to, 1)
                else:
                    optimal_k = current_k  # Current K is feasible
                    current_k -= 1  # Try smaller K
                
            else:
                # UNSAT - K is too small
                solver_stats['unsat_results'] += 1
                print(f"UNSATISFIABLE: K = {current_k}")
                print(f"Optimal bandwidth found: {optimal_k}")
                break
        
        # Final results
        print(f"\nINCREMENTAL SAT OPTIMIZATION WITH UB CUTOFF COMPLETE")
        print(f"="*60)
        print(f"Final optimal bandwidth: {optimal_k}")
        print(f"Theoretical upper bound: {self.theoretical_ub}")
        print(f"UB cutoff benefit: Eliminated {solver_stats['ub_cutoff_benefit']} impossible K values")
        print(f"Solver statistics:")
        print(f"  Total solve calls: {solver_stats['total_solves']}")
        print(f"  SAT results: {solver_stats['sat_results']}")
        print(f"  UNSAT results: {solver_stats['unsat_results']}")
        print(f"  Smart jumps: {solver_stats['smart_jumps']}")
        print(f"  Total clauses added: {solver_stats['clauses_added']}")
        print(f"  Solver: {self.solver_type.upper()} (persistent + UB cutoff)")
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
        Main solve function with incremental SAT and UB cutoff optimization
        
        1. Calculate theoretical upper bound for cutoff optimization
        2. Use incremental SAT with monotone strengthening for optimization
        3. Apply UB cutoff to eliminate impossible assignments early
        """
        if start_k is None:
            start_k = 1
        if end_k is None:
            # Use theoretical upper bound with cutoff optimization
            end_k = self.theoretical_ub
        
        # Apply theoretical UB cutoff immediately
        effective_end_k = min(end_k, self.theoretical_ub)
        
        print(f"\n" + "="*80)
        print(f"2D BANDWIDTH OPTIMIZATION - INCREMENTAL SAT WITH UB CUTOFF")
        print(f"Graph: {self.n} nodes, {len(self.edges)} edges")
        print(f"Strategy: Monotone strengthening + UB cutoff optimization")
        print(f"Theoretical UB: {self.theoretical_ub}")
        print(f"Testing range: K = {start_k} to {effective_end_k}")
        if effective_end_k < end_k:
            print(f"UB cutoff applied: Original end_k {end_k} → {effective_end_k}")
        print(f"="*80)
        
        try:
            # Phase 1: Theoretical upper bound analysis
            print(f"\nPhase 1: Theoretical upper bound analysis with cutoff")
            print(f"Theoretical UB = {self.theoretical_ub}")
            print(f"Using UB = {effective_end_k} (with cutoff optimization)")
            print(f"Benefit: All distance assignments > {self.theoretical_ub} eliminated at encoding")
            
            # Phase 2: Incremental SAT optimization with cutoff
            print(f"\nPhase 2: Incremental SAT optimization with UB cutoff")
            optimal_k = self.solve_with_incremental_sat(effective_end_k)
            
            return optimal_k
            
        finally:
            # Always cleanup solver
            self.cleanup_solver()


def test_incremental_cutoff_solver():
    """Test the incremental cutoff solver on some small graphs"""
    print("=== INCREMENTAL CUTOFF SOLVER TESTS ===")
    
    # Triangle
    print(f"\n" + "="*50)
    print(f"Test 1: Triangle (Incremental SAT + UB Cutoff)")
    print(f"="*50)
    
    n1 = 3
    edges1 = [(1, 2), (2, 3), (1, 3)]
    
    solver1 = IncrementalBandwidthSolverCutoff(n1, 'glucose42')
    solver1.set_graph_edges(edges1)
    solver1.create_position_variables()
    solver1.create_distance_variables()
    
    optimal1 = solver1.solve_bandwidth_optimization(start_k=1, end_k=4)
    print(f"Triangle result: {optimal1}")
    
    # Path
    print(f"\n" + "="*50)
    print(f"Test 2: Path (Incremental SAT + UB Cutoff)")
    print(f"="*50)
    
    n2 = 4
    edges2 = [(1, 2), (2, 3), (3, 4)]
    
    solver2 = IncrementalBandwidthSolverCutoff(n2, 'cadical195')
    solver2.set_graph_edges(edges2)
    solver2.create_position_variables()
    solver2.create_distance_variables()
    
    optimal2 = solver2.solve_bandwidth_optimization(start_k=1, end_k=6)
    print(f"Path result: {optimal2}")
    
    # Cycle
    print(f"\n" + "="*50)
    print(f"Test 3: Cycle (Incremental SAT + UB Cutoff)")
    print(f"="*50)
    
    n3 = 5
    edges3 = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
    
    solver3 = IncrementalBandwidthSolverCutoff(n3, 'glucose42')
    solver3.set_graph_edges(edges3)
    solver3.create_position_variables()
    solver3.create_distance_variables()
    
    optimal3 = solver3.solve_bandwidth_optimization(start_k=1, end_k=8)
    print(f"Cycle result: {optimal3}")
    
    # Performance comparison summary
    print(f"\n" + "="*80)
    print(f"INCREMENTAL SAT WITH UB CUTOFF RESULTS SUMMARY")
    print(f"="*80)
    print(f"Triangle (3 nodes): {optimal1}")
    print(f"Path (4 nodes): {optimal2}")
    print(f"Cycle (5 nodes): {optimal3}")
    print(f"="*80)
    print(f"Strategy: Monotone strengthening + UB cutoff optimization")
    print(f"Benefits: Learnt clauses reuse, no solver restarts, smart jumping, UB cutoff,")
    print(f"          reduced variables, faster encoding, early infeasible elimination")
    print(f"="*80)


def compare_solvers_efficiency():
    """Compare efficiency between standard and cutoff solvers"""
    print(f"\n" + "="*80)
    print(f"SOLVER EFFICIENCY COMPARISON")
    print(f"="*80)
    
    test_graphs = [
        (3, [(1, 2), (2, 3), (1, 3)], "Triangle"),
        (4, [(1, 2), (2, 3), (3, 4)], "Path"),
        (5, [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)], "Cycle"),
    ]
    
    for n, edges, name in test_graphs:
        print(f"\n{name} (n={n}):")
        
        # Theoretical analysis
        ub = calculate_theoretical_upper_bound(n)
        standard_t_vars = len(edges) * 2 * (n - 1)  # Estimate for standard
        cutoff_t_vars = len(edges) * 2 * ub         # Actual for cutoff
        
        print(f"  Theoretical UB: {ub}")
        print(f"  Standard encoding T-vars (estimated): {standard_t_vars}")
        print(f"  Cutoff encoding T-vars: {cutoff_t_vars}")
        
        if standard_t_vars > 0:
            reduction = (standard_t_vars - cutoff_t_vars) / standard_t_vars * 100
            print(f"  Variable reduction: {reduction:.1f}%")
        
        # Clause complexity comparison
        standard_clauses_estimate = n * n * len(edges) * 2  # Rough estimate
        cutoff_clauses_estimate = n * ub * len(edges) * 2   # Optimized estimate
        
        print(f"  Standard encoding clauses (estimated): {standard_clauses_estimate}")
        print(f"  Cutoff encoding clauses (estimated): {cutoff_clauses_estimate}")
        
        if standard_clauses_estimate > 0:
            clause_reduction = (standard_clauses_estimate - cutoff_clauses_estimate) / standard_clauses_estimate * 100
            print(f"  Clause reduction: {clause_reduction:.1f}%")


if __name__ == '__main__':
    """
    Command line usage: python incremental_bandwidth_solver_cutoff.py [mtx_file] [solver]
    
    Arguments:
        mtx_file: Name of MTX file (searches in mtx/group 1/, mtx/group 2/, and mtx/group 3/)
        solver: SAT solver to use (glucose42 or cadical195, default: glucose42)
    
    Examples:
        python incremental_bandwidth_solver_cutoff.py 8.jgl009.mtx glucose42
        python incremental_bandwidth_solver_cutoff.py 1.ash85.mtx cadical195  
        python incremental_bandwidth_solver_cutoff.py 3.bcsstk01.mtx
        python incremental_bandwidth_solver_cutoff.py 1.ck104.mtx
        python incremental_bandwidth_solver_cutoff.py  # Run test mode
        
    Available MTX files:
        Group 1: 1.bcspwr01.mtx, 2.bcspwr02.mtx, 3.bcsstk01.mtx, 4.can___24.mtx,
                 5.fidap005.mtx, 6.fidapm05.mtx, 7.ibm32.mtx, 8.jgl009.mtx, 
                 9.jgl011.mtx, 10.lap_25.mtx, 11.pores_1.mtx, 12.rgg010.mtx
        Group 2: 1.ash85.mtx
        Group 3: Various larger matrices including ck104.mtx, bcsstk04.mtx, etc.
    """
    
    # Check if MTX file provided
    if len(sys.argv) >= 2:
        # MTX file mode
        mtx_file = sys.argv[1]
        solver_type = sys.argv[2] if len(sys.argv) >= 3 else 'glucose42'
        
        print("=" * 80)
        print("2D BANDWIDTH OPTIMIZATION - INCREMENTAL SAT WITH UB CUTOFF")
        print("=" * 80)
        print(f"File: {mtx_file}")
        print(f"Solver: {solver_type.upper()}")
        print(f"Strategy: Monotone strengthening + UB cutoff optimization")
        print(f"Features: Reduced variables, faster encoding, early infeasible elimination")
        
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
                print(f"  python incremental_bandwidth_solver_cutoff.py 8.jgl009.mtx glucose42")
                print(f"  python incremental_bandwidth_solver_cutoff.py 1.ash85.mtx cadical195")
                print(f"  python incremental_bandwidth_solver_cutoff.py 3.bcsstk01.mtx")
                sys.exit(1)
            
            mtx_file = found_file
        
        # Parse MTX file
        def parse_mtx_file(filename):
            """
            Parse MTX file and return n, edges
            
            Handles MatrixMarket format:
            - Comments and metadata parsing
            - Self-loop removal  
            - Undirected graph processing only
            - Error handling for malformed files
            """
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
                
                # Skip comments
                if line.startswith('%'):
                    continue
                
                # Parse header
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
                
                # Parse edge
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        u, v = int(parts[0]), int(parts[1])
                        
                        # Skip self-loops
                        if u != v:
                            # Add both directions for undirected graph
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
        
        # Solve bandwidth problem with incremental SAT and UB cutoff
        print(f"\nSolving 2D bandwidth minimization with Incremental SAT + UB Cutoff...")
        print(f"Problem: {n} vertices on {n}×{n} grid")
        print(f"Strategy: Monotone strengthening + UB cutoff optimization")
        print(f"Using: {solver_type.upper()}")
        
        solver = IncrementalBandwidthSolverCutoff(n, solver_type)
        solver.set_graph_edges(edges)
        solver.create_position_variables()
        solver.create_distance_variables()
        
        start_time = time.time()
        optimal_bandwidth = solver.solve_bandwidth_optimization()
        solve_time = time.time() - start_time
        
        # Results
        print(f"\n" + "="*80)
        print(f"FINAL INCREMENTAL SAT + UB CUTOFF RESULTS")
        print(f"="*80)
        
        if optimal_bandwidth is not None:
            print(f"Optimal bandwidth: {optimal_bandwidth}")
            print(f"Total solve time: {solve_time:.2f}s")
            print(f"Graph: {n} vertices, {len(edges)} edges")
            print(f"Theoretical UB: {solver.theoretical_ub}")
            print(f"Strategy: Monotone strengthening + UB cutoff")
            print(f"Solver: {solver_type.upper()} (persistent + cutoff)")
            print(f"Status: SUCCESS")
        else:
            print(f"No solution found")
            print(f"Total solve time: {solve_time:.2f}s")
            print(f"Status: FAILED")
        
        print(f"="*80)
        
    else:
        # Test mode - run incremental cutoff test cases
        print("=" * 80)
        print("2D BANDWIDTH OPTIMIZATION - INCREMENTAL SAT + UB CUTOFF TEST MODE")
        print("=" * 80)
        print("Usage: python incremental_bandwidth_solver_cutoff.py [mtx_file] [solver]")
        print()
        print("Arguments:")
        print("  mtx_file: Name of MTX file (searches in mtx/group 1/, mtx/group 2/, and mtx/group 3/, and mtx/regular/)")
        print("  solver: SAT solver to use (glucose42 or cadical195, default: glucose42)")
        print()
        print("Examples:")
        print("  python incremental_bandwidth_solver_cutoff.py 8.jgl009.mtx glucose42")
        print("  python incremental_bandwidth_solver_cutoff.py 1.ash85.mtx cadical195")
        print("  python incremental_bandwidth_solver_cutoff.py 3.bcsstk01.mtx")
        print("  python incremental_bandwidth_solver_cutoff.py 1.ck104.mtx")
        print()
        print("Features:")
        print("  - Monotone strengthening: persistent solver with learnt clause reuse")
        print("  - UB cutoff optimization: theoretical upper bound eliminates impossible assignments")
        print("  - Reduced variables: T variables only up to UB, not full range 1..n-1")
        print("  - Faster encoding: O(n × UB) instead of O(n²) complexity")
        print("  - Smart jumping: use actual bandwidth to skip impossible K values")
        print("  - Exact solving: no symmetry breaking for maximum accuracy")
        print()
        print("Available MTX files:")
        print("  Group 1: 1.bcspwr01.mtx, 2.bcspwr02.mtx, 3.bcsstk01.mtx, 4.can___24.mtx,")
        print("           5.fidap005.mtx, 6.fidapm05.mtx, 7.ibm32.mtx, 8.jgl009.mtx,")
        print("           9.jgl011.mtx, 10.lap_25.mtx, 11.pores_1.mtx, 12.rgg010.mtx")
        print("  Group 2: 1.ash85.mtx")
        print("  Group 3: Various larger matrices including ck104.mtx, bcsstk04.mtx, etc.")
        print("  Regular: Various regular matrices")
        print()
        print("Running built-in incremental SAT + UB cutoff test cases...")
        
        # Run tests
        test_incremental_cutoff_solver()
        
        # Run efficiency comparison
        compare_solvers_efficiency()
        
        print(f"\n" + "="*80)
        print(f"KEY BENEFITS OF UB CUTOFF OPTIMIZATION")
        print(f"="*80)
        print(f"1. Reduced Variables: T variables only up to theoretical UB")
        print(f"2. Faster Encoding: O(n × UB) instead of O(n²) complexity")
        print(f"3. Early Elimination: Impossible distance assignments removed at encoding")
        print(f"4. Memory Efficiency: Significantly fewer clauses and variables")
        print(f"5. Pipeline Compatibility: T variables still available for bandwidth ≤ K checks")
        print(f"6. Theoretical Soundness: Based on proven upper bound formulas")
        print(f"="*80)
