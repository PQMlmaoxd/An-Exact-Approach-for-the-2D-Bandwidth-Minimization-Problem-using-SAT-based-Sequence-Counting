# rectangular_bandwidth_solver.py
# 2D Bandwidth Minimization on Rectangular Grids (n×m) using SAT solvers
# Extended from square grid solver to support rectangular grids

from pysat.formula import IDPool
from pysat.solvers import Glucose42, Cadical195, Solver
import random
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from distance_encoder import encode_abs_distance_final
    from random_assignment_ub_finder import RandomAssignmentUBFinder
    from nsc_encoder import encode_nsc_exactly_k, encode_nsc_at_most_k
    from position_constraints import encode_all_position_constraints, create_position_variables
    print("All modules loaded OK")
except ImportError as e:
    print(f"Import error: {e}")
    print("Need NSC encoder module")
    raise ImportError("Missing required modules")

# Basic constants
MAX_RANDOM_ITERATIONS = 1000
RANDOM_TIME_LIMIT = 15
DEFAULT_UB_MULTIPLIER = 2

def parse_mtx_file(file_path):
    """
    Parse MTX file format for adjacency matrix graphs
    Returns: (grid_rows, grid_cols, num_vertices, edges_list)
    
    Format interpretation:
    - Line: rows cols nnz  => Matrix dimensions: rows×cols 
    - Entries: (i,j) or (i,j,val) => Adjacency matrix entries
    - Self-loops are ignored, weights are ignored
    - Returns undirected graph edges only
    """
    edges = []
    edges_set = set()
    
    with open(file_path, 'r') as f:
        # Skip header comments
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()
        
        # Read dimensions: rows cols nnz
        parts = line.strip().split()
        matrix_rows, matrix_cols, nnz = int(parts[0]), int(parts[1]), int(parts[2])
        
        print(f"MTX file: {matrix_rows}×{matrix_cols} matrix, {nnz} entries")
        
        # Parse matrix entries as adjacency matrix
        for _ in range(nnz):
            line = f.readline().strip()
            if line:
                parts = line.split()
                i, j = int(parts[0]), int(parts[1])  # Vertex indices (1-indexed)
                # Ignore weights (parts[2]) - dataset is unweighted
                
                # Skip self-loops
                if i == j:
                    continue
                    
                # Convert to undirected edge (sorted tuple)
                edge = tuple(sorted([i, j]))
                
                if edge not in edges_set:
                    edges_set.add(edge)
                    edges.append(edge)
    
    # Number of vertices is the maximum vertex index
    num_vertices = max(matrix_rows, matrix_cols) if edges else matrix_rows
    
    print(f"Graph: {num_vertices} vertices, {len(edges)} edges (undirected, unweighted)")
    return matrix_rows, matrix_cols, num_vertices, edges

class RectangularBandwidthOptimizationSolver:
    """
    2D Bandwidth Minimization solver for rectangular grids (n×m)
    
    Problem: Place num_vertices on n×m grid to minimize bandwidth
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
        
        print(f"Created rectangular solver: {num_vertices} vertices on {n_rows}×{n_cols} grid, using {solver_type}")
        
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
        """
        clauses = []
        
        # 1. Each vertex gets exactly one X position
        for v in range(1, self.num_vertices + 1):
            # At least one X position
            clauses.append(self.X_vars[v])
            
            # At most one X position (pairwise exclusion)
            for i in range(self.n_rows):
                for j in range(i + 1, self.n_rows):
                    clauses.append([-self.X_vars[v][i], -self.X_vars[v][j]])
        
        # 2. Each vertex gets exactly one Y position
        for v in range(1, self.num_vertices + 1):
            # At least one Y position
            clauses.append(self.Y_vars[v])
            
            # At most one Y position (pairwise exclusion)
            for i in range(self.n_cols):
                for j in range(i + 1, self.n_cols):
                    clauses.append([-self.Y_vars[v][i], -self.Y_vars[v][j]])
        
        # 3. Each grid position gets at most one vertex
        for x_pos in range(self.n_rows):
            for y_pos in range(self.n_cols):
                # Collect all vertices that could be at position (x_pos+1, y_pos+1)
                vertices_at_pos = []
                for v in range(1, self.num_vertices + 1):
                    # Vertex v is at (x_pos+1, y_pos+1) if X_vars[v][x_pos] AND Y_vars[v][y_pos]
                    pos_var = self.vpool.id(f'pos_{x_pos+1}_{y_pos+1}_{v}')
                    vertices_at_pos.append(pos_var)
                    
                    # pos_var ↔ (X_vars[v][x_pos] ∧ Y_vars[v][y_pos])
                    # pos_var → X_vars[v][x_pos]
                    clauses.append([-pos_var, self.X_vars[v][x_pos]])
                    # pos_var → Y_vars[v][y_pos]  
                    clauses.append([-pos_var, self.Y_vars[v][y_pos]])
                    # (X_vars[v][x_pos] ∧ Y_vars[v][y_pos]) → pos_var
                    clauses.append([-self.X_vars[v][x_pos], -self.Y_vars[v][y_pos], pos_var])
                
                # At most one vertex at each position
                for i in range(len(vertices_at_pos)):
                    for j in range(i + 1, len(vertices_at_pos)):
                        clauses.append([-vertices_at_pos[i], -vertices_at_pos[j]])
        
        print(f"Position constraints: {len(clauses)} clauses")
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
        
        print(f"Distance constraints: {len(clauses)} clauses")
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
    
    def step1_test_ub_pure_random(self, K):
        """
        Step 1: Test if K is achievable using random placement on rectangular grid
        """
        print(f"\n--- Step 1: Testing K={K} with random placement on {self.n_rows}×{self.n_cols} grid ---")
        print(f"Looking for assignment with bandwidth <= {K}")
        
        best_bandwidth = float('inf')
        best_assignment = None
        
        for iteration in range(MAX_RANDOM_ITERATIONS):
            # Generate random assignment
            positions = []
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    positions.append((i + 1, j + 1))  # 1-indexed positions
            
            if len(positions) < self.num_vertices:
                print(f"Error: Not enough positions ({len(positions)}) for {self.num_vertices} vertices")
                return False
                
            random.shuffle(positions)
            assignment = {}
            for v in range(1, self.num_vertices + 1):
                x, y = positions[v - 1]
                assignment[f'X_{v}'] = x
                assignment[f'Y_{v}'] = y
            
            # Calculate bandwidth
            max_distance = 0
            for u, v in self.edges:
                x_u = assignment[f'X_{u}']
                y_u = assignment[f'Y_{u}']
                x_v = assignment[f'X_{v}']
                y_v = assignment[f'Y_{v}']
                
                distance = abs(x_u - x_v) + abs(y_u - y_v)
                max_distance = max(max_distance, distance)
            
            if max_distance < best_bandwidth:
                best_bandwidth = max_distance
                best_assignment = assignment.copy()
            
            if max_distance <= K:
                print(f"SUCCESS: Found placement with bandwidth {max_distance} <= {K} in iteration {iteration + 1}")
                return True
        
        print(f"FAILED: Best placement has bandwidth {best_bandwidth} > {K} after {MAX_RANDOM_ITERATIONS} iterations")
        return False
    
    def encode_thermometer_bandwidth_constraints(self, K):
        """
        Encode bandwidth <= K using thermometer encoding for rectangular grid
        
        For each edge: (Tx<=K) ∧ (Ty<=K) ∧ (Tx>=i → Ty<=K-i)
        """
        clauses = []
        
        print(f"Encoding thermometer for K={K} on {self.n_rows}×{self.n_cols} grid:")
        
        for edge_id in self.Tx_vars:
            Tx = self.Tx_vars[edge_id]  # Tx[i] means Tx >= i+1
            Ty = self.Ty_vars[edge_id]  # Ty[i] means Ty >= i+1
            
            print(f"  {edge_id}: {len(Tx)} Tx vars, {len(Ty)} Ty vars")
            
            # Tx <= K (i.e., not Tx >= K+1)
            if K < len(Tx):
                clauses.append([-Tx[K]])
                print(f"    Tx <= {K}")
            
            # Ty <= K (i.e., not Ty >= K+1)
            if K < len(Ty):
                clauses.append([-Ty[K]])
                print(f"    Ty <= {K}")
            
            # Implication: Tx >= i → Ty <= K-i
            for i in range(1, K + 1):
                if K - i >= 0:
                    tx_geq_i = None
                    ty_leq_ki = None
                    
                    if i-1 < len(Tx):
                        tx_geq_i = Tx[i-1]  # Tx >= i
                    
                    if K-i < len(Ty):
                        ty_leq_ki = -Ty[K-i]  # Ty <= K-i (not Ty >= K-i+1)
                    
                    if tx_geq_i is not None and ty_leq_ki is not None:
                        clauses.append([-tx_geq_i, ty_leq_ki])
                        print(f"    Tx>={i} → Ty<={K-i}")
        
        print(f"Generated {len(clauses)} thermometer clauses")
        return clauses
    
    def step2_encode_advanced_constraints_legacy(self, K):
        """
        Legacy method: Test K using complete SAT encoding for rectangular grid
        This creates a new solver instance for each K - not efficient for optimization
        Kept for reference/debugging purposes only
        """
        print(f"\n--- LEGACY: Testing K={K} with SAT encoding on {self.n_rows}×{self.n_cols} grid ---")
        print(f"Using {self.solver_type.upper()} solver")
        print(f"Encoding thermometer constraints for bandwidth <= {K}")
        
        solver = self._create_solver()
        
        try:
            print(f"Building constraints...")
            
            # Position constraints
            position_clauses = self.encode_position_constraints()
            print(f"  Position: {len(position_clauses)} clauses")
            
            # Distance constraints  
            distance_clauses = self.encode_distance_constraints()
            print(f"  Distance: {len(distance_clauses)} clauses")
            
            # Bandwidth constraints
            bandwidth_clauses = self.encode_thermometer_bandwidth_constraints(K)
            print(f"  Bandwidth: {len(bandwidth_clauses)} clauses")
            
            # Add all constraints
            all_clauses = position_clauses + distance_clauses + bandwidth_clauses
            print(f"Total: {len(all_clauses)} clauses")
            
            for clause in all_clauses:
                solver.add_clause(clause)
            
            # Solve
            result = solver.solve()
            
            if result:
                model = solver.get_model()
                self.last_model = model  # Store for extraction later
                print(f"K={K} is SAT")
                print(f"Extracting solution...")
                self.extract_and_verify_solution(model, K)
                return True
            else:
                print(f"K={K} is UNSAT")
                return False
                
        finally:
            solver.delete()
    
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
        """Show solution details for rectangular grid"""
        print(f"Vertex positions on {self.n_rows}×{self.n_cols} grid:")
        for v in range(1, self.num_vertices + 1):
            x = positions.get(f'X_{v}', '?')
            y = positions.get(f'Y_{v}', '?')
            print(f"  v{v}: ({x}, {y})")
        
        print(f"Edge distances:")
        for u, v, distance in edge_distances:
            print(f"  ({u},{v}): {distance}")
        
        print(f"Bandwidth: {bandwidth} (constraint: {K})")
        print(f"Valid: {'Yes' if bandwidth <= K else 'No'}")
    
    def extract_and_verify_solution(self, model, K):
        """Extract solution and check if it satisfies K constraint"""
        print(f"--- Verifying solution on {self.n_rows}×{self.n_cols} grid ---")
        
        positions = self._extract_positions_from_model(model)
        bandwidth, edge_distances = self._calculate_bandwidth(positions)
        self._print_solution_details(positions, edge_distances, bandwidth, K)
        
        return bandwidth <= K
    
    def _find_feasible_upper_bound_phase1(self, start_k, end_k):
        """Phase 1: Find feasible upper bound using random search"""
        print(f"\nPhase 1: Finding feasible UB with random search on {self.n_rows}×{self.n_cols} grid")
        
        for K in range(start_k, end_k + 1):
            print(f"\nTrying K = {K}")
            
            if self.step1_test_ub_pure_random(K):
                print(f"Found feasible UB = {K}")
                return K
            else:
                print(f"K = {K} not achievable")
        
        print(f"\nError: No feasible UB in range [{start_k}, {end_k}]")
        return None
    
    def _optimize_with_sat_phase2(self, feasible_ub):
        """Phase 2: Incremental SAT optimization to find optimal bandwidth"""
        print(f"\nPhase 2: Incremental SAT optimization from K={feasible_ub-1} down to 1")
        
        # Create solver once and reuse
        solver = self._create_solver()
        
        try:
            # Add base constraints once (position + distance)
            print(f"Adding base constraints (position + distance)...")
            
            position_clauses = self.encode_position_constraints()
            distance_clauses = self.encode_distance_constraints()
            base_clauses = position_clauses + distance_clauses
            
            print(f"  Position: {len(position_clauses)} clauses")
            print(f"  Distance: {len(distance_clauses)} clauses")
            print(f"  Total base: {len(base_clauses)} clauses")
            
            for clause in base_clauses:
                solver.add_clause(clause)
            
            optimal_k = feasible_ub
            
            # Incremental SAT: try smaller K values
            for K in range(feasible_ub - 1, 0, -1):
                print(f"\nTrying K = {K} with incremental SAT")
                
                # Add bandwidth constraints for this K directly to solver
                bandwidth_clauses = self.encode_thermometer_bandwidth_constraints(K)
                print(f"  Adding {len(bandwidth_clauses)} bandwidth clauses for K={K}")
                
                for clause in bandwidth_clauses:
                    solver.add_clause(clause)
                
                # Solve with current constraints
                result = solver.solve()
                
                if result:
                    optimal_k = K
                    print(f"K = {K} is SAT")
                    
                    # Extract solution for verification
                    model = solver.get_model()
                    self.last_model = model  # Store for extraction later
                    self.extract_and_verify_solution(model, K)
                else:
                    print(f"K = {K} is UNSAT")
                    print(f"Optimal bandwidth = {optimal_k}")
                    break
            
            print(f"Final optimal bandwidth = {optimal_k}")
            return optimal_k
            
        finally:
            solver.delete()

    def solve_bandwidth_optimization(self, start_k=None, end_k=None):
        """
        Main solve function for rectangular grid
        
        1. Random search to find upper bound
        2. SAT optimization to find minimum
        """
        if start_k is None:
            start_k = 1
        if end_k is None:
            # Maximum possible distance on rectangular grid
            max_distance = (self.n_rows - 1) + (self.n_cols - 1)
            end_k = min(max_distance, DEFAULT_UB_MULTIPLIER * max(self.n_rows, self.n_cols))
        
        print(f"\n" + "="*70)
        print(f"2D BANDWIDTH OPTIMIZATION ON RECTANGULAR GRID")
        print(f"Grid: {self.n_rows}×{self.n_cols} ({self.n_rows * self.n_cols} positions)")
        print(f"Graph: {self.num_vertices} vertices, {len(self.edges)} edges")
        print(f"Testing range: K = {start_k} to {end_k}")
        print(f"Max possible distance: {(self.n_rows-1) + (self.n_cols-1)}")
        print(f"="*70)
        
        # Phase 1: Find upper bound
        feasible_ub = self._find_feasible_upper_bound_phase1(start_k, end_k)
        if feasible_ub is None:
            return None
        
        # Phase 2: Optimize with SAT
        optimal_k = self._optimize_with_sat_phase2(feasible_ub)
        
        return optimal_k

def test_trec5():
    """Test the rectangular solver on Trec5.mtx"""
    print("=== TESTING TREC5.MTX ===")
    
    # Parse Trec5.mtx
    mtx_file = r"e:\2DBMP\An-Exact-Approach-for-the-2D-Bandwidth-Minimization-Problem-using-SAT-based-Sequence-Counting\mtx\Trec5.mtx"
    
    try:
        n_rows, n_cols, num_vertices, edges = parse_mtx_file(mtx_file)
        print(f"Loaded Trec5: {n_rows}×{n_cols} grid with {num_vertices} vertices, {len(edges)} edges")
        print(f"Edges: {edges}")
        
        print(f"\n" + "="*50)
        print(f"Testing Trec5 on {n_rows}×{n_cols} grid")
        print(f"="*50)
        
        try:
            solver = RectangularBandwidthOptimizationSolver(
                num_vertices, n_rows, n_cols, 'glucose42'
            )
            solver.set_graph_edges(edges)
            solver.create_position_variables()
            solver.create_distance_variables()
            
            # Limit search range for efficiency
            max_k = min(20, (n_rows - 1) + (n_cols - 1))
            optimal = solver.solve_bandwidth_optimization(start_k=1, end_k=max_k)
            
            print(f"Result for Trec5 on {n_rows}×{n_cols}: {optimal}")
            return optimal
            
        except Exception as e:
            print(f"Error with {n_rows}×{n_cols} grid: {e}")
            return None
        
    except Exception as e:
        print(f"Error loading Trec5.mtx: {e}")
        return None

if __name__ == '__main__':
    test_trec5()
