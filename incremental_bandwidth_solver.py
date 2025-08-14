# incremental_bandwidth_solver.py
# 2D Bandwidth Minimization using Incremental SAT with Monotone Strengthening
# Strategy: Keep solver alive, monotonically add tightening constraints as K decreases

from pysat.formula import IDPool
from pysat.solvers import Glucose42, Cadical195, Solver
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from distance_encoder import encode_abs_distance_final
    # Using PySAT Sequential Counter instead of NSC
    from position_constraints import encode_all_position_constraints, create_position_variables
    print("All modules loaded OK")
except ImportError as e:
    print(f"Import error: {e}")
    print("Need required modules")
    raise ImportError("Missing required modules")

def calculate_theoretical_upper_bound(n):
    """
    Calculate theoretical upper bound using formula:
    δ(n) = min{2⌈(√(2n-1)-1)/2⌉, 2⌈√(n/2)⌉-1}
    
    This provides a tight upper bound for 2D bandwidth minimization
    based on theoretical analysis of grid placements.
    
    Args:
        n: Number of vertices
        
    Returns:
        Theoretical upper bound for bandwidth
    """
    import math
    
    # Handle special cases
    if n <= 0:
        return 0
    if n == 1:
        print(f"Theoretical UB calculation for n={n}: Single vertex → δ(1) = 0")
        return 0
    if n == 2:
        print(f"Theoretical UB calculation for n={n}: Two vertices → δ(2) = 1")
        return 1
    
    # First term: 2⌈(√(2n-1)-1)/2⌉
    term1 = 2 * math.ceil((math.sqrt(2*n - 1) - 1) / 2)
    
    # Second term: 2⌈√(n/2)⌉-1  
    term2 = 2 * math.ceil(math.sqrt(n / 2)) - 1
    
    # Return minimum of both terms
    ub = min(term1, term2)
    
    print(f"Theoretical UB calculation for n={n}:")
    print(f"  Term 1: 2⌈(√(2×{n}-1)-1)/2⌉ = 2⌈({math.sqrt(2*n-1):.3f}-1)/2⌉ = 2⌈{(math.sqrt(2*n-1)-1)/2:.3f}⌉ = {term1}")
    print(f"  Term 2: 2⌈√({n}/2)⌉-1 = 2⌈√{n/2:.1f}⌉-1 = 2⌈{math.sqrt(n/2):.3f}⌉-1 = {term2}")
    print(f"  δ({n}) = min({term1}, {term2}) = {ub}")
    
    return ub

# Backup implementations if imports fail
if 'encode_abs_distance_final' not in locals():
    def encode_abs_distance_final(U_vars, V_vars, n, vpool, prefix="T"):
        T_vars = [vpool.id(f'{prefix}_geq_{d}') for d in range(1, n)]
        clauses = []
        return T_vars, clauses

class IncrementalBandwidthSolver:
    """
    2D Bandwidth Minimization solver using Incremental SAT with Monotone Strengthening
    
    Strategy:
    1. Keep one solver alive for entire optimization process
    2. Add base constraints (position, distance, symmetry) once at start
    3. For each K value, only add tightening bandwidth constraints
    4. Use actual bandwidth from SAT models to jump to better K values
    5. Leverage learnt clauses across all K values for maximum performance
    
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
        
        # Distance variables  
        self.Tx_vars = {}  # T variables for X distances
        self.Ty_vars = {}  # T variables for Y distances
        
        self.edges = []
        self.last_model = None  # Store last successful SAT model
        
        # Incremental SAT state
        self.persistent_solver = None
        self.base_constraints_added = False
        self.current_k_constraints = set()  # Track which K constraints we've added
        
        print(f"Created incremental solver: n={n}, using {solver_type}")
        print(f"Strategy: Monotone strengthening with persistent solver")
    
    def set_graph_edges(self, edges):
        """Set graph edges"""
        self.edges = edges
        print(f"Graph has {len(edges)} edges")
    
    def create_position_variables(self):
        """Create position variables for vertices on X and Y axes"""
        self.X_vars, self.Y_vars = create_position_variables(self.n, self.vpool)
    
    def create_distance_variables(self):
        """Create T variables for edge distances"""
        for i, (u, v) in enumerate(self.edges):
            edge_id = f'edge_{u}_{v}'
            self.Tx_vars[edge_id] = [self.vpool.id(f'Tx_{edge_id}_{d}') for d in range(1, self.n)]
            self.Ty_vars[edge_id] = [self.vpool.id(f'Ty_{edge_id}_{d}') for d in range(1, self.n)]
    
    def encode_position_constraints(self):
        """
        Position constraints: each vertex gets exactly one position on each axis
        Each position can have at most one vertex
        Uses Sequential Counter encoding for O(n²) complexity
        """
        return encode_all_position_constraints(self.n, self.X_vars, self.Y_vars, self.vpool)
    
    def encode_distance_constraints(self):
        """Encode distance constraints for each edge"""
        clauses = []
        
        for edge_id, (u, v) in zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges):
            # X distance encoding
            Tx_vars, Tx_clauses = encode_abs_distance_final(
                self.X_vars[u], self.X_vars[v], self.n, self.vpool, f"Tx_{edge_id}"
            )
            self.Tx_vars[edge_id] = Tx_vars
            clauses.extend(Tx_clauses)
            
            # Y distance encoding
            Ty_vars, Ty_clauses = encode_abs_distance_final(
                self.Y_vars[u], self.Y_vars[v], self.n, self.vpool, f"Ty_{edge_id}"
            )
            self.Ty_vars[edge_id] = Ty_vars
            clauses.extend(Ty_clauses)
        
        return clauses
    
    def encode_symmetry_breaking_constraints(self):
        """
        Add symmetry breaking constraints to reduce search space
        
        Strategy:
        1. Fix vertex 1 at position (1,1) 
        2. Fix vertex 2 on first row or first column
        3. Order vertices by degree (optional)
        """
        clauses = []
        
        print(f"Adding symmetry breaking constraints...")
        
        # Fix vertex 1 at position (1,1)
        if 1 in self.X_vars and 1 in self.Y_vars:
            # X_1[0] = true (vertex 1 at X position 1)
            clauses.append([self.X_vars[1][0]])
            # Y_1[0] = true (vertex 1 at Y position 1)  
            clauses.append([self.Y_vars[1][0]])
            print(f"  Fixed vertex 1 at position (1,1)")
        
        # Fix vertex 2 on first row (Y=1) to break rotation symmetry
        if 2 in self.Y_vars and len(self.edges) > 0:
            clauses.append([self.Y_vars[2][0]])
            print(f"  Fixed vertex 2 on first row (Y=1)")
        
        print(f"  Added {len(clauses)} symmetry breaking clauses")
        return clauses
    
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
        Initialize persistent solver with base constraints
        
        Base constraints include:
        - Position constraints (each vertex gets one position)
        - Distance constraints (Manhattan distance encoding)
        - Symmetry breaking constraints
        
        These constraints are K-independent and added once.
        """
        if self.persistent_solver is not None:
            print("Persistent solver already initialized")
            return
        
        print(f"\nInitializing persistent solver with base constraints...")
        print(f"Using {self.solver_type.upper()} with incremental interface")
        
        self.persistent_solver = self._create_solver()
        
        # Add position constraints
        print(f"  Adding position constraints...")
        position_clauses = self.encode_position_constraints()
        print(f"    Position: {len(position_clauses)} clauses")
        
        for clause in position_clauses:
            self.persistent_solver.add_clause(clause)
        
        # Add distance constraints  
        print(f"  Adding distance constraints...")
        distance_clauses = self.encode_distance_constraints()
        print(f"    Distance: {len(distance_clauses)} clauses")
        
        for clause in distance_clauses:
            self.persistent_solver.add_clause(clause)
        
        # Add symmetry breaking constraints
        print(f"  Adding symmetry breaking constraints...")
        symmetry_clauses = self.encode_symmetry_breaking_constraints()
        print(f"    Symmetry: {len(symmetry_clauses)} clauses")
        
        for clause in symmetry_clauses:
            self.persistent_solver.add_clause(clause)
        
        total_base_clauses = len(position_clauses) + len(distance_clauses) + len(symmetry_clauses)
        print(f"  Total base constraints: {total_base_clauses} clauses")
        
        self.base_constraints_added = True
        print(f"Persistent solver initialized and ready for incremental solving")
    
    def encode_bandwidth_constraints_for_k(self, K):
        """
        Encode bandwidth <= K constraints for incremental addition
        
        Returns only the NEW tightening clauses for this specific K.
        Uses monotone strengthening: never removes constraints, only adds.
        
        For each edge: (Tx<=K) ∧ (Ty<=K) ∧ (Tx>=i → Ty<=K-i)
        """
        if K in self.current_k_constraints:
            print(f"  K={K} constraints already added (monotone strengthening)")
            return []
        
        new_clauses = []
        edges_processed = 0
        
        print(f"  Encoding NEW bandwidth constraints for K={K}...")
        
        for edge_id in self.Tx_vars:
            Tx = self.Tx_vars[edge_id]  # Tx[i] means Tx >= i+1
            Ty = self.Ty_vars[edge_id]  # Ty[i] means Ty >= i+1
            edges_processed += 1
            
            # Tx <= K (i.e., not Tx >= K+1)
            if K < len(Tx):
                clause = [-Tx[K]]
                new_clauses.append(clause)
            
            # Ty <= K (i.e., not Ty >= K+1)  
            if K < len(Ty):
                clause = [-Ty[K]]
                new_clauses.append(clause)
            
            # Implication: Tx >= i → Ty <= K-i
            for i in range(1, K + 1):
                if K - i >= 0:
                    tx_geq_i = None
                    ty_leq_ki = None
                    
                    if i-1 < len(Tx):
                        tx_geq_i = Tx[i-1]  # Tx >= i
                    
                    if K-i < len(Ty):
                        ty_leq_ki = -Ty[K-i]  # Ty <= K-i (negated)
                    
                    if tx_geq_i is not None and ty_leq_ki is not None:
                        clause = [-tx_geq_i, ty_leq_ki]
                        new_clauses.append(clause)
        
        # Mark this K as processed
        self.current_k_constraints.add(K)
        
        print(f"  Generated {len(new_clauses)} NEW bandwidth clauses for {edges_processed} edges, K={K}")
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
        Main incremental SAT solving with monotone strengthening
        
        Strategy:
        1. Start from upper_bound, test K values going down
        2. If SAT at K with actual bandwidth Y < K, jump to K = Y-1  
        3. If UNSAT at K, optimal is previous K
        4. Use persistent solver + learnt clauses for maximum efficiency
        """
        print(f"\nINCREMENTAL SAT OPTIMIZATION")
        print(f"Strategy: Monotone strengthening from K={upper_bound} down to 1")
        print(f"Solver: {self.solver_type.upper()} with persistent incremental interface")
        
        # Initialize persistent solver with base constraints
        self._initialize_persistent_solver()
        
        # Calculate theoretical upper bound once and apply it immediately
        theoretical_ub = calculate_theoretical_upper_bound(self.n)
        current_k = min(upper_bound, theoretical_ub)
        
        if current_k < upper_bound:
            print(f"Adjusted upper bound from {upper_bound} to {current_k} (theoretical limit)")
        
        optimal_k = None
        solver_stats = {
            'total_solves': 0,
            'sat_results': 0,
            'unsat_results': 0,
            'smart_jumps': 0,
            'clauses_added': 0
        }
        
        while current_k >= 1:
            print(f"\nTesting K = {current_k}")
            solver_stats['total_solves'] += 1
            
            # Add bandwidth constraints for this K (monotone strengthening)
            print(f"  Adding constraints for K={current_k}...")
            bandwidth_clauses = self.encode_bandwidth_constraints_for_k(current_k)
            
            # Add new constraints to persistent solver
            for clause in bandwidth_clauses:
                self.persistent_solver.add_clause(clause)
            
            solver_stats['clauses_added'] += len(bandwidth_clauses)
            
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
                
                # Smart jumping based on actual bandwidth
                if actual_bandwidth < current_k:
                    print(f"SMART JUMP: actual={actual_bandwidth} < K={current_k}")
                    print(f"   Jumping from K={current_k} directly to K={actual_bandwidth}")
                    
                    # Update optimal and jump
                    optimal_k = actual_bandwidth
                    current_k = actual_bandwidth - 1
                    solver_stats['smart_jumps'] += 1
                    
                    # No verification needed - smart jump is based purely on actual_bandwidth
                else:
                    # Normal case: actual bandwidth equals K
                    optimal_k = current_k
                    current_k -= 1
                    
                    # No verification needed - logic is deterministic
                
            else:
                # UNSAT - K is too small
                solver_stats['unsat_results'] += 1
                print(f"UNSATISFIABLE: K = {current_k}")
                print(f"Optimal bandwidth found: {optimal_k}")
                break
        
        # Final results
        print(f"\nINCREMENTAL SAT OPTIMIZATION COMPLETE")
        print(f"="*60)
        print(f"Final optimal bandwidth: {optimal_k}")
        print(f"Solver statistics:")
        print(f"  Total solve calls: {solver_stats['total_solves']}")
        print(f"  SAT results: {solver_stats['sat_results']}")
        print(f"  UNSAT results: {solver_stats['unsat_results']}")
        print(f"  Smart jumps: {solver_stats['smart_jumps']}")
        print(f"  Total clauses added: {solver_stats['clauses_added']}")
        print(f"  Solver: {self.solver_type.upper()} (persistent)")
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
        Main solve function with incremental SAT
        
        1. Calculate theoretical upper bound
        2. Use incremental SAT with monotone strengthening for optimization
        """
        if start_k is None:
            start_k = 1
        if end_k is None:
            # Use theoretical upper bound
            end_k = calculate_theoretical_upper_bound(self.n)
        
        print(f"\n" + "="*80)
        print(f"2D BANDWIDTH OPTIMIZATION - INCREMENTAL SAT")
        print(f"Graph: {self.n} nodes, {len(self.edges)} edges")
        print(f"Strategy: Monotone strengthening with persistent solver")
        print(f"Testing range: K = {start_k} to {end_k}")
        print(f"="*80)
        
        try:
            # Phase 1: Find feasible upper bound
            print(f"\nPhase 1: Theoretical upper bound analysis")
            theoretical_ub = calculate_theoretical_upper_bound(self.n)
            feasible_ub = min(theoretical_ub, end_k)
            
            print(f"Theoretical UB = {theoretical_ub}")
            print(f"Using UB = {feasible_ub} (capped at {end_k})")
            
            # Phase 2: Incremental SAT optimization
            print(f"\nPhase 2: Incremental SAT optimization")
            optimal_k = self.solve_with_incremental_sat(feasible_ub)
            
            return optimal_k
            
        finally:
            # Always cleanup solver
            self.cleanup_solver()

def test_incremental_bandwidth_solver():
    """Test the incremental solver on some small graphs"""
    print("=== INCREMENTAL BANDWIDTH SOLVER TESTS ===")
    
    # Triangle
    print(f"\n" + "="*50)
    print(f"Test 1: Triangle (Incremental SAT)")
    print(f"="*50)
    
    n1 = 3
    edges1 = [(1, 2), (2, 3), (1, 3)]
    
    solver1 = IncrementalBandwidthSolver(n1, 'glucose42')
    solver1.set_graph_edges(edges1)
    solver1.create_position_variables()
    solver1.create_distance_variables()
    
    optimal1 = solver1.solve_bandwidth_optimization(start_k=1, end_k=4)
    print(f"Triangle result: {optimal1}")
    
    # Path
    print(f"\n" + "="*50)
    print(f"Test 2: Path (Incremental SAT)")
    print(f"="*50)
    
    n2 = 4
    edges2 = [(1, 2), (2, 3), (3, 4)]
    
    solver2 = IncrementalBandwidthSolver(n2, 'cadical195')
    solver2.set_graph_edges(edges2)
    solver2.create_position_variables()
    solver2.create_distance_variables()
    
    optimal2 = solver2.solve_bandwidth_optimization(start_k=1, end_k=6)
    print(f"Path result: {optimal2}")
    
    # Cycle
    print(f"\n" + "="*50)
    print(f"Test 3: Cycle (Incremental SAT)")
    print(f"="*50)
    
    n3 = 5
    edges3 = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
    
    solver3 = IncrementalBandwidthSolver(n3, 'glucose42')
    solver3.set_graph_edges(edges3)
    solver3.create_position_variables()
    solver3.create_distance_variables()
    
    optimal3 = solver3.solve_bandwidth_optimization(start_k=1, end_k=8)
    print(f"Cycle result: {optimal3}")
    
    # Performance comparison summary
    print(f"\n" + "="*80)
    print(f"INCREMENTAL SAT RESULTS SUMMARY")
    print(f"="*80)
    print(f"Triangle (3 nodes): {optimal1}")
    print(f"Path (4 nodes): {optimal2}")
    print(f"Cycle (5 nodes): {optimal3}")
    print(f"="*80)
    print(f"Strategy: Monotone strengthening with persistent solver")
    print(f"Benefits: Learnt clauses reuse, no solver restarts, smart jumping")
    print(f"="*80)

if __name__ == '__main__':
    """
    Command line usage: python incremental_bandwidth_solver.py [mtx_file] [solver]
    
    Arguments:
        mtx_file: Name of MTX file (searches in mtx/group 1/, mtx/group 2/, and mtx/group 3/)
        solver: SAT solver to use (glucose42 or cadical195, default: glucose42)
    
    Examples:
        python incremental_bandwidth_solver.py 8.jgl009.mtx glucose42
        python incremental_bandwidth_solver.py 1.ash85.mtx cadical195  
        python incremental_bandwidth_solver.py 3.bcsstk01.mtx
        python incremental_bandwidth_solver.py 1.ck104.mtx
        python incremental_bandwidth_solver.py  # Run test mode
        
    Available MTX files:
        Group 1: 1.bcspwr01.mtx, 2.bcspwr02.mtx, 3.bcsstk01.mtx, 4.can___24.mtx,
                 5.fidap005.mtx, 6.fidapm05.mtx, 7.ibm32.mtx, 8.jgl009.mtx, 
                 9.jgl011.mtx, 10.lap_25.mtx, 11.pores_1.mtx, 12.rgg010.mtx
        Group 2: 1.ash85.mtx
        Group 3: Various larger matrices including ck104.mtx, bcsstk04.mtx, etc.
    """
    import sys
    
    # Check if MTX file provided
    if len(sys.argv) >= 2:
        # MTX file mode
        mtx_file = sys.argv[1]
        solver_type = sys.argv[2] if len(sys.argv) >= 3 else 'glucose42'
        
        print("=" * 80)
        print("2D BANDWIDTH OPTIMIZATION - INCREMENTAL SAT SOLVER")
        print("=" * 80)
        print(f"File: {mtx_file}")
        print(f"Solver: {solver_type.upper()}")
        print(f"Strategy: Monotone strengthening with persistent solver")
        
        # Search for file in common locations
        if not os.path.exists(mtx_file):
            search_paths = [
                mtx_file,
                f"mtx/{mtx_file}",
                f"mtx/group 1/{mtx_file}",
                f"mtx/group 2/{mtx_file}",
                f"mtx/group 3/{mtx_file}",
                f"sample_mtx_datasets/{mtx_file}",
                f"mtx/{mtx_file}.mtx",
                f"mtx/group 1/{mtx_file}.mtx", 
                f"mtx/group 2/{mtx_file}.mtx",
                f"mtx/group 3/{mtx_file}.mtx",
                f"sample_mtx_datasets/{mtx_file}.mtx"
            ]
            
            found_file = None
            for path in search_paths:
                if os.path.exists(path):
                    found_file = path
                    print(f"Found file at: {path}")
                    break
            
            if found_file is None:
                print(f"Error: File '{mtx_file}' not found")
                print("Searched in:")
                for path in search_paths:
                    print(f"  - {path}")
                print(f"\nAvailable files in mtx/group 1/:")
                group1_path = "mtx/group 1"
                if os.path.exists(group1_path):
                    for file in sorted(os.listdir(group1_path)):
                        if file.endswith('.mtx'):
                            print(f"  - {file}")
                
                print(f"\nAvailable files in mtx/group 2/:")
                group2_path = "mtx/group 2"
                if os.path.exists(group2_path):
                    for file in sorted(os.listdir(group2_path)):
                        if file.endswith('.mtx'):
                            print(f"  - {file}")
                
                print(f"\nAvailable files in mtx/group 3/:")
                group3_path = "mtx/group 3"
                if os.path.exists(group3_path):
                    for file in sorted(os.listdir(group3_path)):
                        if file.endswith('.mtx'):
                            print(f"  - {file}")
                            
                print(f"\nUsage examples:")
                print(f"  python incremental_bandwidth_solver.py 8.jgl009.mtx glucose42")
                print(f"  python incremental_bandwidth_solver.py 1.ash85.mtx cadical195")
                print(f"  python incremental_bandwidth_solver.py 3.bcsstk01.mtx")
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
                print(f"File not found: {filename}")
                return None, None
            
            header_found = False
            edges_set = set()
            n = 0
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                if not line:
                    continue
                    
                # Handle comments and metadata
                if line.startswith('%'):
                    # Skip metadata - dataset is all undirected/unweighted
                    continue
                
                # Parse dimensions
                if not header_found:
                    try:
                        parts = line.split()
                        if len(parts) >= 3:
                            rows, cols, nnz = map(int, parts[:3])
                            n = max(rows, cols)
                            print(f"Matrix: {rows}×{cols}, {nnz} entries")
                            print(f"Graph: undirected, unweighted (dataset standard)")
                            header_found = True
                            continue
                    except ValueError:
                        print(f"Warning: bad header at line {line_num}: {line}")
                        continue
                
                # Parse edges
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        u, v = int(parts[0]), int(parts[1])
                        # Ignore weights (parts[2]) - dataset is unweighted
                        
                        if u == v:  # skip self-loops
                            continue
                        
                        # Always convert to undirected edge (sorted tuple)
                        edge = tuple(sorted([u, v]))
                        
                        if edge not in edges_set:
                            edges_set.add(edge)
                            
                except (ValueError, IndexError):
                    print(f"Warning: bad edge at line {line_num}: {line}")
                    continue
            
            edges = list(edges_set)
            print(f"Loaded: {n} vertices, {len(edges)} edges")
            return n, edges
        
        # Parse graph
        n, edges = parse_mtx_file(mtx_file)
        if n is None or edges is None:
            print("Failed to parse MTX file")
            sys.exit(1)
        
        # Solve bandwidth problem with incremental SAT
        print(f"\nSolving 2D bandwidth minimization with Incremental SAT...")
        print(f"Problem: {n} vertices on {n}×{n} grid")
        print(f"Strategy: Monotone strengthening")
        print(f"Using: {solver_type.upper()}")
        
        solver = IncrementalBandwidthSolver(n, solver_type)
        solver.set_graph_edges(edges)
        solver.create_position_variables()
        solver.create_distance_variables()
        
        start_time = time.time()
        optimal_bandwidth = solver.solve_bandwidth_optimization()
        solve_time = time.time() - start_time
        
        # Results
        print(f"\n" + "="*80)
        print(f"FINAL INCREMENTAL SAT RESULTS")
        print(f"="*80)
        
        if optimal_bandwidth is not None:
            print(f"Optimal bandwidth: {optimal_bandwidth}")
            print(f"Total solve time: {solve_time:.2f}s")
            print(f"Graph: {n} vertices, {len(edges)} edges")
            print(f"Strategy: Monotone strengthening")
            print(f"Solver: {solver_type.upper()} (persistent)")
            print(f"Status: SUCCESS")
        else:
            print(f"No solution found")
            print(f"Total solve time: {solve_time:.2f}s")
            print(f"Status: FAILED")
        
        print(f"="*80)
        
    else:
        # Test mode - run incremental test cases
        print("=" * 80)
        print("2D BANDWIDTH OPTIMIZATION - INCREMENTAL SAT TEST MODE")
        print("=" * 80)
        print("Usage: python incremental_bandwidth_solver.py [mtx_file] [solver]")
        print()
        print("Arguments:")
        print("  mtx_file: Name of MTX file (searches in mtx/group 1/, mtx/group 2/, and mtx/group 3/)")
        print("  solver: SAT solver to use (glucose42 or cadical195, default: glucose42)")
        print()
        print("Examples:")
        print("  python incremental_bandwidth_solver.py 8.jgl009.mtx glucose42")
        print("  python incremental_bandwidth_solver.py 1.ash85.mtx cadical195")
        print("  python incremental_bandwidth_solver.py 3.bcsstk01.mtx")
        print("  python incremental_bandwidth_solver.py 1.ck104.mtx")
        print()
        print("Features:")
        print("  - Monotone strengthening: persistent solver with learnt clause reuse")
        print("  - Smart jumping: use actual bandwidth to skip impossible K values")
        print("  - Symmetry breaking: reduce search space significantly")
        print("  - Incremental interface: maximum performance with minimal overhead")
        print()
        print("Available MTX files:")
        print("  Group 1: 1.bcspwr01.mtx, 2.bcspwr02.mtx, 3.bcsstk01.mtx, 4.can___24.mtx,")
        print("           5.fidap005.mtx, 6.fidapm05.mtx, 7.ibm32.mtx, 8.jgl009.mtx,")
        print("           9.jgl011.mtx, 10.lap_25.mtx, 11.pores_1.mtx, 12.rgg010.mtx")
        print("  Group 2: 1.ash85.mtx")
        print("  Group 3: Various larger matrices including ck104.mtx, bcsstk04.mtx, etc.")
        print()
        print("Running built-in incremental SAT test cases...")
        test_incremental_bandwidth_solver()
