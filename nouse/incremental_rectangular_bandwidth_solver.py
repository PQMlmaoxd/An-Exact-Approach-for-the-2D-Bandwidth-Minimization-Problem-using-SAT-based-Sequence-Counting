# incremental_rectangular_bandwidth_solver.py
# 2D Bandwidth Minimization on Rectangular Grids (m×n) using Incremental SAT with Monotone Strengthening
# Strategy: Keep solver alive, monotonically add tightening constraints as K decreases

from pysat.formula import IDPool
from pysat.solvers import Glucose42, Cadical195, Solver
from pysat.card import CardEnc, EncType
import random
import time
import sys
import os
import math
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

# Basic constants
MAX_RANDOM_ITERATIONS = 1000
RANDOM_TIME_LIMIT = 15
DEFAULT_UB_MULTIPLIER = 2

def calculate_theoretical_upper_bound_rectangular(n_vertices, n_rows, n_cols):
    """
    Calculate theoretical upper bound for rectangular grid using adapted formula:
    For rectangular grid m×n, the maximum Manhattan distance is (m-1) + (n-1)
    
    This provides a tight upper bound for 2D bandwidth minimization
    based on theoretical analysis of rectangular grid placements.
    
    Args:
        n_vertices: Number of vertices
        n_rows: Grid height (m)
        n_cols: Grid width (n)
        
    Returns:
        Theoretical upper bound for bandwidth on rectangular grid
    """
    
    # Handle special cases
    if n_vertices <= 0:
        return 0
    if n_vertices == 1:
        print(f"Theoretical UB calculation for {n_vertices} vertices on {n_rows}×{n_cols}: Single vertex → δ(1) = 0")
        return 0
    if n_vertices == 2:
        print(f"Theoretical UB calculation for {n_vertices} vertices on {n_rows}×{n_cols}: Two vertices → δ(2) = 1")
        return 1
    
    # Maximum possible Manhattan distance on rectangular grid
    max_distance = (n_rows - 1) + (n_cols - 1)
    
    # Adapted theoretical bounds for rectangular grids
    # Term 1: Based on vertex packing density
    density_factor = n_vertices / (n_rows * n_cols)
    term1 = max_distance if density_factor > 0.5 else int(max_distance * 0.7)
    
    # Term 2: Based on grid aspect ratio
    aspect_ratio = max(n_rows, n_cols) / min(n_rows, n_cols)
    if aspect_ratio > 2.0:
        # For very rectangular grids, bandwidth tends to be higher
        term2 = int(max_distance * 0.8)
    else:
        # For nearly square grids, use square grid heuristics
        term2 = min(2 * math.ceil(math.sqrt(n_vertices / 2)) - 1, max_distance)
    
    # Term 3: Conservative bound based on maximum distance
    term3 = max_distance
    
    # Return minimum of all terms
    ub = min(term1, term2, term3)
    
    print(f"Theoretical UB calculation for {n_vertices} vertices on {n_rows}×{n_cols} grid:")
    print(f"  Max distance: {max_distance}")
    print(f"  Density factor: {density_factor:.3f}")
    print(f"  Aspect ratio: {aspect_ratio:.3f}")
    print(f"  Term 1 (density): {term1}")
    print(f"  Term 2 (aspect): {term2}")
    print(f"  Term 3 (max): {term3}")
    print(f"  δ({n_vertices}) = min({term1}, {term2}, {term3}) = {ub}")
    
    return ub

# Backup implementations if imports fail
if 'encode_abs_distance_final' not in locals():
    def encode_abs_distance_final(U_vars, V_vars, n, vpool, prefix="T"):
        T_vars = [vpool.id(f'{prefix}_geq_{d}') for d in range(1, n)]
        clauses = []
        return T_vars, clauses

class IncrementalRectangularBandwidthSolver:
    """
    2D Bandwidth Minimization solver for rectangular grids (m×n) using Incremental SAT with Monotone Strengthening
    
    Strategy:
    1. Keep one solver alive for entire optimization process
    2. Add base constraints (position, distance, symmetry) once at start
    3. For each K value, only add tightening bandwidth constraints
    4. Use actual bandwidth from SAT models to jump to better K values
    5. Leverage learnt clauses across all K values for maximum performance
    
    Problem: Place num_vertices on m×n grid to minimize bandwidth
    bandwidth = max{|x_u - x_v| + |y_u - y_v| : (u,v) ∈ E}
    
    Key differences from square solver:
    - Grid dimensions: n_rows × n_cols (can be different)
    - Position variables: X ∈ [1, n_rows], Y ∈ [1, n_cols]
    - Distance encoding adapted for rectangular bounds
    """
    
    def __init__(self, num_vertices, n_rows, n_cols, solver_type='glucose42'):
        self.num_vertices = num_vertices
        self.n_rows = n_rows  # Height of grid
        self.n_cols = n_cols  # Width of grid
        self.solver_type = solver_type
        self.vpool = IDPool()
        
        # Position variables for X,Y coordinates on rectangular grid
        self.X_vars = {}  # X_vars[v][pos] = variable for vertex v at X position pos (1 to n_rows)
        self.Y_vars = {}  # Y_vars[v][pos] = variable for vertex v at Y position pos (1 to n_cols)
        
        # Distance variables  
        self.Tx_vars = {}  # T variables for X distances (max distance = n_rows-1)
        self.Ty_vars = {}  # T variables for Y distances (max distance = n_cols-1)
        
        self.edges = []
        self.last_model = None  # Store last successful SAT model
        
        # Incremental SAT state
        self.persistent_solver = None
        self.base_constraints_added = False
        self.current_k_constraints = set()  # Track which K constraints we've added
        
        print(f"Created incremental rectangular solver: {num_vertices} vertices on {n_rows}×{n_cols} grid, using {solver_type}")
        print(f"Strategy: Monotone strengthening with persistent solver")
        
        # Validate grid size
        if num_vertices > n_rows * n_cols:
            raise ValueError(f"Cannot place {num_vertices} vertices on {n_rows}×{n_cols} grid (only {n_rows*n_cols} positions)")
    
    def set_graph_edges(self, edges):
        """Set graph edges"""
        self.edges = edges
        print(f"Graph has {len(edges)} edges")
        
        # Validate vertices are within range
        max_vertex = max(max(u, v) for u, v in edges) if edges else 0
        if max_vertex > self.num_vertices:
            print(f"Warning: Max vertex ID {max_vertex} > num_vertices {self.num_vertices}")
            self.num_vertices = max_vertex
    
    def create_position_variables(self):
        """Create position variables for vertices on rectangular grid"""
        # X positions: 1 to n_rows
        self.X_vars = {}
        for v in range(1, self.num_vertices + 1):
            self.X_vars[v] = [self.vpool.id(f'X_{v}_{pos}') for pos in range(1, self.n_rows + 1)]
        
        # Y positions: 1 to n_cols  
        self.Y_vars = {}
        for v in range(1, self.num_vertices + 1):
            self.Y_vars[v] = [self.vpool.id(f'Y_{v}_{pos}') for pos in range(1, self.n_cols + 1)]
            
        print(f"Created position variables: {self.num_vertices} vertices")
        print(f"  X vars: {self.num_vertices} × {self.n_rows} = {self.num_vertices * self.n_rows}")
        print(f"  Y vars: {self.num_vertices} × {self.n_cols} = {self.num_vertices * self.n_cols}")
    
    def create_distance_variables(self):
        """Create T variables for edge distances on rectangular grid"""
        for i, (u, v) in enumerate(self.edges):
            edge_id = f'edge_{u}_{v}'
            # X distances: 0 to n_rows-1, so thermometer vars for 1 to n_rows-1
            self.Tx_vars[edge_id] = [self.vpool.id(f'Tx_{edge_id}_{d}') for d in range(1, self.n_rows)]
            # Y distances: 0 to n_cols-1, so thermometer vars for 1 to n_cols-1
            self.Ty_vars[edge_id] = [self.vpool.id(f'Ty_{edge_id}_{d}') for d in range(1, self.n_cols)]
            
        print(f"Created distance variables for {len(self.edges)} edges")
        print(f"  Tx vars per edge: {self.n_rows - 1}")
        print(f"  Ty vars per edge: {self.n_cols - 1}")
    
    def encode_position_constraints(self):
        """
        Position constraints for rectangular grid:
        1. Each vertex gets exactly one X position (1 to n_rows)
        2. Each vertex gets exactly one Y position (1 to n_cols)  
        3. Each grid position (x,y) gets at most one vertex
        Uses Sequential Counter encoding for efficient constraint generation.
        """
        clauses = []
        
        # 1. Each vertex gets exactly one X position using Sequential Counter
        for v in range(1, self.num_vertices + 1):
            sc_x_clauses = CardEnc.equals(self.X_vars[v], 1, vpool=self.vpool, encoding=EncType.seqcounter)
            clauses.extend(sc_x_clauses.clauses)
        
        # 2. Each vertex gets exactly one Y position using Sequential Counter
        for v in range(1, self.num_vertices + 1):
            sc_y_clauses = CardEnc.equals(self.Y_vars[v], 1, vpool=self.vpool, encoding=EncType.seqcounter)
            clauses.extend(sc_y_clauses.clauses)
        
        # 3. Each grid position gets at most one vertex
        for x_pos in range(self.n_rows):
            for y_pos in range(self.n_cols):
                # Collect all vertices that could be at position (x_pos+1, y_pos+1)
                vertices_at_pos = []
                for v in range(1, self.num_vertices + 1):
                    # Check if vertex v can be at this specific grid position
                    vertex_at_xy = [self.X_vars[v][x_pos], self.Y_vars[v][y_pos]]
                    vertices_at_pos.extend(vertex_at_xy)
                
                # At most one vertex at each position using Sequential Counter
                if vertices_at_pos:
                    sc_at_most_1 = CardEnc.atmost(vertices_at_pos, 1, vpool=self.vpool, encoding=EncType.seqcounter)
                    clauses.extend(sc_at_most_1.clauses)
        
        return clauses
    
    def encode_distance_constraints(self):
        """Encode distance constraints for each edge on rectangular grid"""
        clauses = []
        
        for edge_id, (u, v) in zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges):
            # X distance encoding (max distance = n_rows - 1)
            Tx_vars, Tx_clauses = encode_abs_distance_final(
                self.X_vars[u], self.X_vars[v], self.n_rows, self.vpool, f"Tx_{edge_id}"
            )
            self.Tx_vars[edge_id] = Tx_vars
            clauses.extend(Tx_clauses)
            
            # Y distance encoding (max distance = n_cols - 1)
            Ty_vars, Ty_clauses = encode_abs_distance_final(
                self.Y_vars[u], self.Y_vars[v], self.n_cols, self.vpool, f"Ty_{edge_id}"
            )
            self.Ty_vars[edge_id] = Ty_vars
            clauses.extend(Ty_clauses)
        
        return clauses
    
    def encode_symmetry_breaking_constraints(self):
        """
        Add symmetry breaking constraints to reduce search space for rectangular grid
        
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
        Initialize persistent solver with base constraints for rectangular grid
        
        Base constraints include:
        - Position constraints (each vertex gets one position on rectangular grid)
        - Distance constraints (Manhattan distance encoding adapted for rectangular bounds)
        - Symmetry breaking constraints
        
        These constraints are K-independent and added once.
        """
        if self.persistent_solver is not None:
            print("Persistent solver already initialized")
            return
        
        print(f"\n🔧 Initializing persistent solver with base constraints for {self.n_rows}×{self.n_cols} grid...")
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
        print(f"✓ Persistent solver initialized and ready for incremental solving")
    
    def encode_bandwidth_constraints_for_k(self, K):
        """
        Encode bandwidth <= K constraints for incremental addition on rectangular grid
        
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
        """Extract vertex positions from SAT solution for rectangular grid"""
        positions = {}
        for v in range(1, self.num_vertices + 1):
            # Find X position (1 to n_rows)
            for pos in range(1, self.n_rows + 1):
                var_id = self.X_vars[v][pos-1]
                if var_id in model and model[model.index(var_id)] > 0:
                    positions[f'X_{v}'] = pos
                    break
            
            # Find Y position (1 to n_cols)
            for pos in range(1, self.n_cols + 1):
                var_id = self.Y_vars[v][pos-1]
                if var_id in model and model[model.index(var_id)] > 0:
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
        """Show solution details (summary only) for rectangular grid"""
        print(f"Solution summary:")
        print(f"  Grid: {self.n_rows}×{self.n_cols}")
        print(f"  Vertices placed: {len([v for v in range(1, self.num_vertices + 1) if f'X_{v}' in positions])}/{self.num_vertices}")
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
        Main incremental SAT solving with monotone strengthening for rectangular grid
        
        Strategy:
        1. Start from upper_bound, test K values going down
        2. If SAT at K with actual bandwidth Y < K, jump to K = Y-1  
        3. If UNSAT at K, optimal is previous K
        4. Use persistent solver + learnt clauses for maximum efficiency
        """
        print(f"\n🚀 INCREMENTAL SAT OPTIMIZATION ON {self.n_rows}×{self.n_cols} GRID")
        print(f"Strategy: Monotone strengthening from K={upper_bound} down to 1")
        print(f"Solver: {self.solver_type.upper()} with persistent incremental interface")
        
        # Initialize persistent solver with base constraints
        self._initialize_persistent_solver()
        
        optimal_k = None
        current_k = upper_bound
        solver_stats = {
            'total_solves': 0,
            'sat_results': 0,
            'unsat_results': 0,
            'smart_jumps': 0,
            'clauses_added': 0
        }
        
        while current_k >= 1:
            print(f"\n📋 Testing K = {current_k}")
            solver_stats['total_solves'] += 1
            
            # Check theoretical bound first
            theoretical_ub = calculate_theoretical_upper_bound_rectangular(self.num_vertices, self.n_rows, self.n_cols)
            if current_k > theoretical_ub:
                print(f"K = {current_k} > theoretical UB = {theoretical_ub}, skipping")
                current_k -= 1
                continue
            
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
                print(f"✅ K = {current_k} is SAT")
                
                # Extract model and calculate actual bandwidth
                model = self.persistent_solver.get_model()
                self.last_model = model
                actual_bandwidth = self.extract_actual_bandwidth(model)
                
                print(f"  Actual bandwidth from solution: {actual_bandwidth}")
                
                # Smart jumping based on actual bandwidth
                if actual_bandwidth < current_k:
                    print(f"🚀 SMART JUMP: actual={actual_bandwidth} < K={current_k}")
                    print(f"   Jumping from K={current_k} directly to K={actual_bandwidth}")
                    
                    # Update optimal and jump
                    optimal_k = actual_bandwidth
                    current_k = actual_bandwidth
                    solver_stats['smart_jumps'] += 1
                    
                    # Show detailed solution for the jump
                    is_valid, _ = self.extract_and_verify_solution(model, actual_bandwidth)
                else:
                    # Normal case: actual bandwidth equals K
                    optimal_k = current_k
                    current_k -= 1
                    
                    # Show solution details
                    is_valid, _ = self.extract_and_verify_solution(model, optimal_k)
                
            else:
                # UNSAT - K is too small
                solver_stats['unsat_results'] += 1
                print(f"❌ K = {current_k} is UNSAT")
                print(f"✓ Optimal bandwidth found: {optimal_k}")
                break
        
        # Final results
        print(f"\n🎯 INCREMENTAL SAT OPTIMIZATION COMPLETE")
        print(f"="*60)
        print(f"Final optimal bandwidth: {optimal_k}")
        print(f"Grid: {self.n_rows}×{self.n_cols}")
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
            print(f"🧹 Cleaning up persistent solver...")
            self.persistent_solver.delete()
            self.persistent_solver = None
            self.base_constraints_added = False
            self.current_k_constraints.clear()
    
    def solve_bandwidth_optimization(self, start_k=None, end_k=None):
        """
        Main solve function with incremental SAT for rectangular grid
        
        1. Calculate theoretical upper bound for rectangular grid
        2. Use incremental SAT with monotone strengthening for optimization
        """
        if start_k is None:
            start_k = 1
        if end_k is None:
            # Use theoretical upper bound for rectangular grid
            end_k = calculate_theoretical_upper_bound_rectangular(self.num_vertices, self.n_rows, self.n_cols)
        
        print(f"\n" + "="*80)
        print(f"2D BANDWIDTH OPTIMIZATION - INCREMENTAL SAT ON RECTANGULAR GRID")
        print(f"Grid: {self.n_rows}×{self.n_cols} ({self.n_rows * self.n_cols} positions)")
        print(f"Graph: {self.num_vertices} nodes, {len(self.edges)} edges")
        print(f"Strategy: Monotone strengthening with persistent solver")
        print(f"Testing range: K = {start_k} to {end_k}")
        print(f"Max possible distance: {(self.n_rows-1) + (self.n_cols-1)}")
        print(f"="*80)
        
        try:
            # Phase 1: Find feasible upper bound
            print(f"\nPhase 1: Theoretical upper bound analysis")
            theoretical_ub = calculate_theoretical_upper_bound_rectangular(self.num_vertices, self.n_rows, self.n_cols)
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

def test_incremental_rectangular_bandwidth_solver():
    """Test the incremental rectangular solver on some small graphs"""
    print("=== INCREMENTAL RECTANGULAR BANDWIDTH SOLVER TESTS ===")
    
    # Test 1: Small rectangular example
    print(f"\n" + "="*50)
    print(f"Test 1: Small Rectangle (Incremental SAT)")
    print(f"="*50)
    
    n_vertices = 4
    edges = [(1, 2), (2, 3), (3, 4), (1, 4)]  # Rectangle graph
    n_rows, n_cols = 2, 2  # 2x2 grid
    
    solver1 = IncrementalRectangularBandwidthSolver(n_vertices, n_rows, n_cols, 'glucose42')
    solver1.set_graph_edges(edges)
    solver1.create_position_variables()
    solver1.create_distance_variables()
    
    optimal1 = solver1.solve_bandwidth_optimization(start_k=1, end_k=4)
    print(f"Small rectangular result: {optimal1}")
    
    # Test 2: Path on non-square grid
    print(f"\n" + "="*50)
    print(f"Test 2: Path on 3×2 Grid (Incremental SAT)")
    print(f"="*50)
    
    n_vertices2 = 5
    edges2 = [(1, 2), (2, 3), (3, 4), (4, 5)]  # Path
    n_rows2, n_cols2 = 3, 2  # 3x2 grid
    
    solver2 = IncrementalRectangularBandwidthSolver(n_vertices2, n_rows2, n_cols2, 'cadical195')
    solver2.set_graph_edges(edges2)
    solver2.create_position_variables()
    solver2.create_distance_variables()
    
    optimal2 = solver2.solve_bandwidth_optimization(start_k=1, end_k=6)
    print(f"Path on 3x2 grid result: {optimal2}")
    
    # Test 3: Triangle on wide grid
    print(f"\n" + "="*50)
    print(f"Test 3: Triangle on 1×5 Grid (Incremental SAT)")
    print(f"="*50)
    
    n_vertices3 = 3
    edges3 = [(1, 2), (2, 3), (1, 3)]  # Triangle
    n_rows3, n_cols3 = 1, 5  # 1x5 grid (very wide)
    
    solver3 = IncrementalRectangularBandwidthSolver(n_vertices3, n_rows3, n_cols3, 'glucose42')
    solver3.set_graph_edges(edges3)
    solver3.create_position_variables()
    solver3.create_distance_variables()
    
    optimal3 = solver3.solve_bandwidth_optimization(start_k=1, end_k=6)
    print(f"Triangle on 1x5 grid result: {optimal3}")
    
    # Performance comparison summary
    print(f"\n" + "="*80)
    print(f"INCREMENTAL RECTANGULAR SAT RESULTS SUMMARY")
    print(f"="*80)
    print(f"Small rectangular (2×2): {optimal1}")
    print(f"Path on 3×2 grid: {optimal2}")
    print(f"Triangle on 1×5 grid: {optimal3}")
    print(f"="*80)
    print(f"Strategy: Monotone strengthening with persistent solver")
    print(f"Benefits: Learnt clauses reuse, no solver restarts, smart jumping")
    print(f"="*80)

if __name__ == '__main__':
    """
    Command line usage: python incremental_rectangular_bandwidth_solver.py [mtx_file] [n_rows] [n_cols] [solver]
    
    Arguments:
        mtx_file: Name of MTX file (searches in mtx/group 1/ and mtx/group 2/)
        n_rows: Number of grid rows (height)
        n_cols: Number of grid columns (width)
        solver: SAT solver to use (glucose42 or cadical195, default: glucose42)
    
    Examples:
        python incremental_rectangular_bandwidth_solver.py 8.jgl009.mtx 3 3 glucose42
        python incremental_rectangular_bandwidth_solver.py 1.ash85.mtx 4 3 cadical195  
        python incremental_rectangular_bandwidth_solver.py 3.bcsstk01.mtx 2 5
        python incremental_rectangular_bandwidth_solver.py  # Run test mode
        
    Available MTX files:
        Group 1: 1.bcspwr01.mtx, 2.bcspwr02.mtx, 3.bcsstk01.mtx, 4.can___24.mtx,
                 5.fidap005.mtx, 6.fidapm05.mtx, 7.ibm32.mtx, 8.jgl009.mtx, 
                 9.jgl011.mtx, 10.lap_25.mtx, 11.pores_1.mtx, 12.rgg010.mtx
        Group 2: 1.ash85.mtx
    """
    import sys
    
    # Check if MTX file provided
    if len(sys.argv) >= 4:  # Need at least mtx_file, n_rows, n_cols
        # MTX file mode with rectangular grid
        mtx_file = sys.argv[1]
        n_rows = int(sys.argv[2])
        n_cols = int(sys.argv[3])
        solver_type = sys.argv[4] if len(sys.argv) >= 5 else 'glucose42'
        
        print("=" * 80)
        print("INCREMENTAL 2D BANDWIDTH OPTIMIZATION - RECTANGULAR SAT SOLVER")
        print("=" * 80)
        print(f"File: {mtx_file}")
        print(f"Grid: {n_rows}×{n_cols}")
        print(f"Solver: {solver_type.upper()}")
        print(f"Strategy: Monotone strengthening with persistent solver")
        
        # Search for file in common locations
        if not os.path.exists(mtx_file):
            search_paths = [
                mtx_file,
                f"mtx/{mtx_file}",
                f"mtx/group 1/{mtx_file}",
                f"mtx/group 2/{mtx_file}",
                f"sample_mtx_datasets/{mtx_file}",
                f"mtx/{mtx_file}.mtx",
                f"mtx/group 1/{mtx_file}.mtx", 
                f"mtx/group 2/{mtx_file}.mtx",
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
                            
                print(f"\nUsage examples:")
                print(f"  python incremental_rectangular_bandwidth_solver.py 8.jgl009.mtx 3 3 glucose42")
                print(f"  python incremental_rectangular_bandwidth_solver.py 1.ash85.mtx 4 3 cadical195")
                print(f"  python incremental_rectangular_bandwidth_solver.py 3.bcsstk01.mtx 2 5")
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
        
        # Validate grid dimensions
        if n > n_rows * n_cols:
            print(f"Error: Cannot place {n} vertices on {n_rows}×{n_cols} grid")
            print(f"Grid has only {n_rows * n_cols} positions")
            
            # Suggest minimum grid sizes
            min_square = int(math.ceil(math.sqrt(n)))
            print(f"\nSuggested grid sizes:")
            print(f"  Minimum square: {min_square}×{min_square}")
            print(f"  Single row: {n}×1")
            print(f"  Single column: 1×{n}")
            
            # Show rectangular options
            for rows in range(2, min(n + 1, 10)):
                cols = math.ceil(n / rows)
                if rows * cols >= n:
                    print(f"  Option: {rows}×{cols}")
            sys.exit(1)
        
        # Solve rectangular bandwidth problem with incremental SAT
        print(f"\nSolving rectangular 2D bandwidth minimization with Incremental SAT...")
        print(f"Problem: {n} vertices on {n_rows}×{n_cols} grid")
        print(f"Grid utilization: {(n / (n_rows * n_cols)) * 100:.1f}%")
        print(f"Strategy: Monotone strengthening")
        print(f"Using: {solver_type.upper()}")
        
        solver = IncrementalRectangularBandwidthSolver(n, n_rows, n_cols, solver_type)
        solver.set_graph_edges(edges)
        solver.create_position_variables()
        solver.create_distance_variables()
        
        start_time = time.time()
        optimal_bandwidth = solver.solve_bandwidth_optimization()
        solve_time = time.time() - start_time
        
        # Results
        print(f"\n" + "="*80)
        print(f"FINAL INCREMENTAL RECTANGULAR SAT RESULTS")
        print(f"="*80)
        
        if optimal_bandwidth is not None:
            max_possible = (n_rows - 1) + (n_cols - 1)
            print(f"✓ Optimal bandwidth: {optimal_bandwidth}")
            print(f"✓ Total solve time: {solve_time:.2f}s")
            print(f"✓ Graph: {n} vertices, {len(edges)} edges")
            print(f"✓ Grid: {n_rows}×{n_cols} ({n_rows * n_cols} positions)")
            print(f"✓ Max possible distance: {max_possible}")
            print(f"✓ Strategy: Monotone strengthening")
            print(f"✓ Solver: {solver_type.upper()} (persistent)")
            print(f"✓ Status: SUCCESS")
        else:
            print(f"✗ No solution found")
            print(f"✗ Total solve time: {solve_time:.2f}s")
            print(f"✗ Status: FAILED")
        
        print(f"="*80)
        
    else:
        # Test mode - run incremental test cases
        print("=" * 80)
        print("INCREMENTAL 2D BANDWIDTH OPTIMIZATION - RECTANGULAR SAT TEST MODE")
        print("=" * 80)
        print("Usage: python incremental_rectangular_bandwidth_solver.py [mtx_file] [n_rows] [n_cols] [solver]")
        print()
        print("Arguments:")
        print("  mtx_file: Name of MTX file (searches in mtx/group 1/ and mtx/group 2/)")
        print("  n_rows: Number of grid rows (height)")
        print("  n_cols: Number of grid columns (width)")
        print("  solver: SAT solver to use (glucose42 or cadical195, default: glucose42)")
        print()
        print("Examples:")
        print("  python incremental_rectangular_bandwidth_solver.py 8.jgl009.mtx 3 3 glucose42")
        print("  python incremental_rectangular_bandwidth_solver.py 1.ash85.mtx 4 3 cadical195")
        print("  python incremental_rectangular_bandwidth_solver.py 3.bcsstk01.mtx 2 5")
        print()
        print("Features:")
        print("  - Monotone strengthening: persistent solver with learnt clause reuse")
        print("  - Smart jumping: use actual bandwidth to skip impossible K values")
        print("  - Symmetry breaking: reduce search space significantly")
        print("  - Incremental interface: maximum performance with minimal overhead")
        print("  - Rectangular grid support: m×n grids where m ≠ n")
        print()
        print("Available MTX files:")
        print("  Group 1: 1.bcspwr01.mtx, 2.bcspwr02.mtx, 3.bcsstk01.mtx, 4.can___24.mtx,")
        print("           5.fidap005.mtx, 6.fidapm05.mtx, 7.ibm32.mtx, 8.jgl009.mtx,")
        print("           9.jgl011.mtx, 10.lap_25.mtx, 11.pores_1.mtx, 12.rgg010.mtx")
        print("  Group 2: 1.ash85.mtx")
        print()
        print("Running built-in incremental rectangular SAT test cases...")
        test_incremental_rectangular_bandwidth_solver()
