# non_incremental_bandwidth_solver.py
# 2D Bandwidth Minimization using Non-Incremental SAT 
# Strategy: Create fresh solver for each K value test

from pysat.formula import IDPool
from pysat.solvers import Glucose42, Cadical195, Solver
import time
import sys
import os
import gc
import ctypes
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

class NonIncrementalBandwidthSolver:
    """
    2D Bandwidth Minimization solver using Non-Incremental SAT
    
    Strategy:
    1. Create fresh solver for each K value test
    2. Add all constraints (position, distance, bandwidth) for specific K
    3. No persistent state - complete independence between K tests
    4. Simpler debugging and isolation of K-specific issues
    
    Problem: Place n vertices on n×n grid to minimize bandwidth
    bandwidth = max{|x_u - x_v| + |y_u - y_v| : (u,v) ∈ E}
    """
    
    def __init__(self, n, solver_type='glucose42'):
        self.n = n
        self.solver_type = solver_type
        
        # Graph structure
        self.edges = []
        self.last_model = None  # Store last successful SAT model
        
        print(f"Created non-incremental solver: n={n}, using {solver_type}")
        print(f"Strategy: Fresh solver instance per K value")
    
    def set_graph_edges(self, edges):
        """Set graph edges"""
        self.edges = edges
        print(f"Graph has {len(edges)} edges")
    
    def _create_solver(self):
        """Create SAT solver instance"""
        if self.solver_type == 'glucose42':
            return Glucose42()
        elif self.solver_type == 'cadical195':
            return Cadical195()
        else:
            print(f"Unknown solver '{self.solver_type}', using Glucose42")
            return Glucose42()
    
    def create_variables_for_k(self, K):
        """
        Create all variables needed for specific K value
        Returns: vpool, X_vars, Y_vars, Tx_vars, Ty_vars
        """
        vpool = IDPool()
        
        # Position variables for X,Y coordinates
        X_vars, Y_vars = create_position_variables(self.n, vpool)
        
        # Distance variables for each edge
        Tx_vars = {}  # T variables for X distances
        Ty_vars = {}  # T variables for Y distances
        
        for i, (u, v) in enumerate(self.edges):
            edge_id = f'edge_{u}_{v}'
            Tx_vars[edge_id] = [vpool.id(f'Tx_{edge_id}_{d}') for d in range(1, self.n)]
            Ty_vars[edge_id] = [vpool.id(f'Ty_{edge_id}_{d}') for d in range(1, self.n)]
        
        return vpool, X_vars, Y_vars, Tx_vars, Ty_vars
    
    def encode_position_constraints(self, X_vars, Y_vars, vpool):
        """
        Position constraints: each vertex gets exactly one position on each axis
        Each position can have at most one vertex
        Uses Sequential Counter encoding for O(n²) complexity
        """
        return encode_all_position_constraints(self.n, X_vars, Y_vars, vpool)
    
    def encode_distance_constraints(self, X_vars, Y_vars, Tx_vars, Ty_vars, vpool):
        """Encode distance constraints for each edge"""
        clauses = []
        
        for edge_id, (u, v) in zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges):
            # X distance encoding
            Tx_vars_edge, Tx_clauses = encode_abs_distance_final(
                X_vars[u], X_vars[v], self.n, vpool, f"Tx_{edge_id}"
            )
            Tx_vars[edge_id] = Tx_vars_edge
            clauses.extend(Tx_clauses)
            
            # Y distance encoding
            Ty_vars_edge, Ty_clauses = encode_abs_distance_final(
                Y_vars[u], Y_vars[v], self.n, vpool, f"Ty_{edge_id}"
            )
            Ty_vars[edge_id] = Ty_vars_edge
            clauses.extend(Ty_clauses)
        
        return clauses
    
    def encode_bandwidth_constraints(self, Tx_vars, Ty_vars, K):
        """
        Encode bandwidth <= K constraints
        
        For each edge: (Tx<=K) ∧ (Ty<=K) ∧ (Tx>=i → Ty<=K-i) ∧ (Ty>=i → Tx<=K-i)
        
        The constraint Tx + Ty <= K is encoded as:
        - Direct bounds: Tx <= K, Ty <= K
        - Symmetric implications: 
          * Tx >= i → Ty <= K-i (if Tx is at least i, Ty must be at most K-i)
          * Ty >= i → Tx <= K-i (if Ty is at least i, Tx must be at most K-i)
        
        IMPORTANT: Both directions are needed for correctness!
        """
        clauses = []
        edges_processed = 0
        
        for edge_id in Tx_vars:
            Tx = Tx_vars[edge_id]  # Tx[i] means Tx >= i+1
            Ty = Ty_vars[edge_id]  # Ty[i] means Ty >= i+1
            edges_processed += 1
            
            # Tx <= K (i.e., not Tx >= K+1)
            if K < len(Tx):
                clause = [-Tx[K]]
                clauses.append(clause)
            
            # Ty <= K (i.e., not Ty >= K+1)  
            if K < len(Ty):
                clause = [-Ty[K]]
                clauses.append(clause)
            
            # Symmetric implications for Tx + Ty <= K
            for i in range(1, K + 1):
                remaining = K - i
                if remaining >= 0:
                    # Direction 1: Tx >= i → Ty <= remaining
                    # Equivalent: ¬(Tx >= i) ∨ ¬(Ty >= remaining+1)
                    if i - 1 < len(Tx) and remaining < len(Ty):
                        tx_geq_i = Tx[i - 1]           # Tx >= i
                        ty_geq_rem_plus1 = Ty[remaining]  # Ty >= remaining+1
                        clause = [-tx_geq_i, -ty_geq_rem_plus1]
                        clauses.append(clause)
                    
                    # Direction 2: Ty >= i → Tx <= remaining (SYMMETRIC - CRITICAL!)
                    # Equivalent: ¬(Ty >= i) ∨ ¬(Tx >= remaining+1)
                    if i - 1 < len(Ty) and remaining < len(Tx):
                        ty_geq_i = Ty[i - 1]           # Ty >= i
                        tx_geq_rem_plus1 = Tx[remaining]  # Tx >= remaining+1
                        clause = [-ty_geq_i, -tx_geq_rem_plus1]
                        clauses.append(clause)
        
        print(f"  Generated {len(clauses)} bandwidth clauses for {edges_processed} edges, K={K}")
        return clauses
    
    def _extract_positions_from_model(self, model, X_vars, Y_vars):
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
                if X_vars[v][pos-1] in posset:
                    positions[f'X_{v}'] = pos
                    break
            
            # Find Y position
            for pos in range(1, self.n + 1):
                if Y_vars[v][pos-1] in posset:
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
    
    def extract_and_verify_solution(self, model, X_vars, Y_vars, K):
        """Extract solution and check if it satisfies K constraint"""
        print(f"--- Verifying solution for K={K} ---")
        
        positions = self._extract_positions_from_model(model, X_vars, Y_vars)
        bandwidth, edge_distances = self._calculate_bandwidth(positions)
        self._print_solution_details(positions, edge_distances, bandwidth, K)
        
        return bandwidth <= K, bandwidth  # Return both validity and actual bandwidth
    
    def extract_actual_bandwidth(self, model, X_vars, Y_vars):
        """Extract actual bandwidth from SAT solution without printing details"""
        positions = self._extract_positions_from_model(model, X_vars, Y_vars)
        bandwidth, _ = self._calculate_bandwidth(positions)
        return bandwidth
    
    def solve_for_k(self, K):
        """
        Solve bandwidth minimization for specific K value using fresh solver
        Uses streaming approach to minimize peak RAM usage.
        
        Returns: (is_sat, actual_bandwidth, solve_time, model)
        """
        print(f"\nTesting K = {K} (fresh solver)")
        print(f"Using streaming approach to minimize peak RAM")
        
        # Create fresh variables for this K
        vpool, X_vars, Y_vars, Tx_vars, Ty_vars = self.create_variables_for_k(K)
        
        # Create fresh solver
        solver = self._create_solver()
        
        # Stream position constraints (add directly from generator)
        print(f"  Adding position constraints...")
        position_clause_count = 0
        for clause in self.encode_position_constraints(X_vars, Y_vars, vpool):
            solver.add_clause(clause)
            position_clause_count += 1
        print(f"    Position: {position_clause_count} clauses")
        
        # Stream distance constraints (add and clear immediately)
        print(f"  Adding distance constraints...")
        distance_clauses = self.encode_distance_constraints(X_vars, Y_vars, Tx_vars, Ty_vars, vpool)
        print(f"    Distance: {len(distance_clauses)} clauses")
        
        for clause in distance_clauses:
            solver.add_clause(clause)
        distance_clauses.clear()
        del distance_clauses
        
        # Stream bandwidth constraints (add and clear immediately)
        print(f"  Adding bandwidth constraints...")
        bandwidth_clauses = self.encode_bandwidth_constraints(Tx_vars, Ty_vars, K)
        print(f"    Bandwidth: {len(bandwidth_clauses)} clauses")
        
        for clause in bandwidth_clauses:
            solver.add_clause(clause)
        bandwidth_clauses.clear()
        del bandwidth_clauses
        
        # Force garbage collection and memory trim
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
        
        # Solve
        print(f"  Solving with fresh {self.solver_type.upper()}...")
        solve_start = time.time()
        result = solver.solve()
        solve_time = time.time() - solve_start
        
        print(f"  Solve time: {solve_time:.3f}s")
        
        if result:
            # SAT - extract model and calculate actual bandwidth
            model = solver.get_model()
            actual_bandwidth = self.extract_actual_bandwidth(model, X_vars, Y_vars)
            print(f"SATISFIABLE: K = {K}, actual bandwidth = {actual_bandwidth}")
            
            # Store last successful model
            self.last_model = model
            
            # Cleanup solver
            solver.delete()
            
            return True, actual_bandwidth, solve_time, model
        else:
            print(f"UNSATISFIABLE: K = {K}")
            
            # Cleanup solver
            solver.delete()
            
            return False, None, solve_time, None
    
    def solve_bandwidth_optimization(self, start_k=None, end_k=None):
        """
        Main solve function using non-incremental SAT
        
        1. Calculate theoretical upper bound
        2. Test K values from high to low using fresh solvers
        3. Use smart jumping based on actual bandwidth
        """
        if start_k is None:
            start_k = 1
        if end_k is None:
            # Use theoretical upper bound
            end_k = calculate_theoretical_upper_bound(self.n)
        
        print(f"\n" + "="*80)
        print(f"2D BANDWIDTH OPTIMIZATION - NON-INCREMENTAL SAT")
        print(f"Graph: {self.n} nodes, {len(self.edges)} edges")
        print(f"Strategy: Fresh solver per K value")
        print(f"Testing range: K = {start_k} to {end_k}")
        print(f"="*80)
        
        # Phase 1: Find feasible upper bound
        print(f"\nPhase 1: Theoretical upper bound analysis")
        theoretical_ub = calculate_theoretical_upper_bound(self.n)
        feasible_ub = min(theoretical_ub, end_k)
        
        print(f"Theoretical UB = {theoretical_ub}")
        print(f"Using UB = {feasible_ub} (capped at {end_k})")
        
        # Phase 2: Non-incremental SAT optimization
        print(f"\nPhase 2: Non-incremental SAT optimization")
        
        current_k = feasible_ub
        optimal_k = None
        solver_stats = {
            'total_solves': 0,
            'sat_results': 0,
            'unsat_results': 0,
            'smart_jumps': 0,
            'total_solve_time': 0.0
        }
        
        while current_k >= start_k:
            solver_stats['total_solves'] += 1
            
            # Solve for this K using fresh solver
            is_sat, actual_bandwidth, solve_time, model = self.solve_for_k(current_k)
            solver_stats['total_solve_time'] += solve_time
            
            if is_sat:
                # SAT - found solution
                solver_stats['sat_results'] += 1
                optimal_k = current_k
                
                # Smart jumping based on actual bandwidth
                if actual_bandwidth < current_k:
                    print(f"SMART JUMP: actual={actual_bandwidth} < K={current_k}")
                    print(f"   Jumping from K={current_k} directly to K={actual_bandwidth}")
                    
                    # Update optimal and jump
                    optimal_k = actual_bandwidth
                    current_k = actual_bandwidth - 1
                    solver_stats['smart_jumps'] += 1
                else:
                    # Normal case: actual bandwidth equals K
                    current_k -= 1
                    
            else:
                # UNSAT - K is too small
                solver_stats['unsat_results'] += 1
                print(f"Optimal bandwidth found: {optimal_k}")
                break
        
        # Final results
        print(f"\nNON-INCREMENTAL SAT OPTIMIZATION COMPLETE")
        print(f"="*60)
        print(f"Final optimal bandwidth: {optimal_k}")
        print(f"Solver statistics:")
        print(f"  Total solve calls: {solver_stats['total_solves']}")
        print(f"  SAT results: {solver_stats['sat_results']}")
        print(f"  UNSAT results: {solver_stats['unsat_results']}")
        print(f"  Smart jumps: {solver_stats['smart_jumps']}")
        print(f"  Total solve time: {solver_stats['total_solve_time']:.3f}s")
        print(f"  Average per solve: {solver_stats['total_solve_time']/max(1,solver_stats['total_solves']):.3f}s")
        print(f"  Solver: {self.solver_type.upper()} (fresh per K)")
        print(f"="*60)
        
        return optimal_k

def test_non_incremental_bandwidth_solver():
    """Test the non-incremental solver on some small graphs"""
    print("=== NON-INCREMENTAL BANDWIDTH SOLVER TESTS ===")
    
    # Triangle
    print(f"\n" + "="*50)
    print(f"Test 1: Triangle (Non-Incremental SAT)")
    print(f"="*50)
    
    n1 = 3
    edges1 = [(1, 2), (2, 3), (1, 3)]
    
    solver1 = NonIncrementalBandwidthSolver(n1, 'glucose42')
    solver1.set_graph_edges(edges1)
    
    optimal1 = solver1.solve_bandwidth_optimization(start_k=1, end_k=4)
    print(f"Triangle result: {optimal1}")
    
    # Path
    print(f"\n" + "="*50)
    print(f"Test 2: Path (Non-Incremental SAT)")
    print(f"="*50)
    
    n2 = 4
    edges2 = [(1, 2), (2, 3), (3, 4)]
    
    solver2 = NonIncrementalBandwidthSolver(n2, 'cadical195')
    solver2.set_graph_edges(edges2)
    
    optimal2 = solver2.solve_bandwidth_optimization(start_k=1, end_k=6)
    print(f"Path result: {optimal2}")
    
    # Cycle
    print(f"\n" + "="*50)
    print(f"Test 3: Cycle (Non-Incremental SAT)")
    print(f"="*50)
    
    n3 = 5
    edges3 = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
    
    solver3 = NonIncrementalBandwidthSolver(n3, 'glucose42')
    solver3.set_graph_edges(edges3)
    
    optimal3 = solver3.solve_bandwidth_optimization(start_k=1, end_k=8)
    print(f"Cycle result: {optimal3}")
    
    # Performance comparison summary
    print(f"\n" + "="*80)
    print(f"NON-INCREMENTAL SAT RESULTS SUMMARY")
    print(f"="*80)
    print(f"Triangle (3 nodes): {optimal1}")
    print(f"Path (4 nodes): {optimal2}")
    print(f"Cycle (5 nodes): {optimal3}")
    print(f"="*80)
    print(f"Strategy: Fresh solver per K value")
    print(f"Benefits: Complete isolation, easier debugging, no persistent state issues")
    print(f"Drawbacks: No learnt clause reuse, more memory allocation overhead")
    print(f"="*80)

if __name__ == '__main__':
    """
    Command line usage: python non_incremental_bandwidth_solver.py [mtx_file] [solver]
    
    Arguments:
        mtx_file: Name of MTX file (searches in mtx/group 1/, mtx/group 2/, and mtx/group 3/)
        solver: SAT solver to use (glucose42 or cadical195, default: glucose42)
    
    Examples:
        python non_incremental_bandwidth_solver.py 8.jgl009.mtx glucose42
        python non_incremental_bandwidth_solver.py 1.ash85.mtx cadical195  
        python non_incremental_bandwidth_solver.py 3.bcsstk01.mtx
        python non_incremental_bandwidth_solver.py 1.ck104.mtx
        python non_incremental_bandwidth_solver.py  # Run test mode
        
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
        print("2D BANDWIDTH OPTIMIZATION - NON-INCREMENTAL SAT SOLVER")
        print("=" * 80)
        print(f"File: {mtx_file}")
        print(f"Solver: {solver_type.upper()}")
        print(f"Strategy: Fresh solver per K value")
        
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

                print(f"\nAvailable files in mtx/regular/:")
                regular_path = "mtx/regular"
                if os.path.exists(regular_path):
                    for file in sorted(os.listdir(regular_path)):
                        if file.endswith('.mtx'):
                            print(f"  - {file}")

                print(f"\nUsage examples:")
                print(f"  python non_incremental_bandwidth_solver.py 8.jgl009.mtx glucose42")
                print(f"  python non_incremental_bandwidth_solver.py 1.ash85.mtx cadical195")
                print(f"  python non_incremental_bandwidth_solver.py 3.bcsstk01.mtx")
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
        
        # Solve bandwidth problem with non-incremental SAT
        print(f"\nSolving 2D bandwidth minimization with Non-Incremental SAT...")
        print(f"Problem: {n} vertices on {n}×{n} grid")
        print(f"Strategy: Fresh solver per K value")
        print(f"Using: {solver_type.upper()}")
        
        solver = NonIncrementalBandwidthSolver(n, solver_type)
        solver.set_graph_edges(edges)
        
        start_time = time.time()
        optimal_bandwidth = solver.solve_bandwidth_optimization()
        solve_time = time.time() - start_time
        
        # Results
        print(f"\n" + "="*80)
        print(f"FINAL NON-INCREMENTAL SAT RESULTS")
        print(f"="*80)
        
        if optimal_bandwidth is not None:
            print(f"Optimal bandwidth: {optimal_bandwidth}")
            print(f"Total solve time: {solve_time:.2f}s")
            print(f"Graph: {n} vertices, {len(edges)} edges")
            print(f"Strategy: Fresh solver per K value")
            print(f"Solver: {solver_type.upper()} (fresh per K)")
            print(f"Status: SUCCESS")
        else:
            print(f"No solution found")
            print(f"Total solve time: {solve_time:.2f}s")
            print(f"Status: FAILED")
        
        print(f"="*80)
        
    else:
        # Test mode - run non-incremental test cases
        print("=" * 80)
        print("2D BANDWIDTH OPTIMIZATION - NON-INCREMENTAL SAT TEST MODE")
        print("=" * 80)
        print("Usage: python non_incremental_bandwidth_solver.py [mtx_file] [solver]")
        print()
        print("Arguments:")
        print("  mtx_file: Name of MTX file (searches in mtx/group 1/, mtx/group 2/, and mtx/group 3/, and mtx/regular/)")
        print("  solver: SAT solver to use (glucose42 or cadical195, default: glucose42)")
        print()
        print("Examples:")
        print("  python non_incremental_bandwidth_solver.py 8.jgl009.mtx glucose42")
        print("  python non_incremental_bandwidth_solver.py 1.ash85.mtx cadical195")
        print("  python non_incremental_bandwidth_solver.py 3.bcsstk01.mtx")
        print("  python non_incremental_bandwidth_solver.py 1.ck104.mtx")
        print()
        print("Features:")
        print("  - Fresh solver per K: complete isolation between K tests")
        print("  - Smart jumping: use actual bandwidth to skip impossible K values")
        print("  - Exact solving: no symmetry breaking for maximum accuracy")
        print("  - Independent solving: easier debugging and state management")
        print()
        print("Available MTX files:")
        print("  Group 1: 1.bcspwr01.mtx, 2.bcspwr02.mtx, 3.bcsstk01.mtx, 4.can___24.mtx,")
        print("           5.fidap005.mtx, 6.fidapm05.mtx, 7.ibm32.mtx, 8.jgl009.mtx,")
        print("           9.jgl011.mtx, 10.lap_25.mtx, 11.pores_1.mtx, 12.rgg010.mtx")
        print("  Group 2: 1.ash85.mtx")
        print("  Group 3: Various larger matrices including ck104.mtx, bcsstk04.mtx, etc.")
        print("  Regular: Various regular matrices")
        print()
        print("Running built-in non-incremental SAT test cases...")
        test_non_incremental_bandwidth_solver()
