# bandwidth_optimization_solver.py
# 2D Bandwidth Minimization using SAT solvers
# Works with Glucose42 and Cadical195

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

# Backup implementations if imports fail
if 'encode_abs_distance_final' not in locals():
    def encode_abs_distance_final(U_vars, V_vars, n, vpool, prefix="T"):
        T_vars = [vpool.id(f'{prefix}_geq_{d}') for d in range(1, n)]
        clauses = []
        return T_vars, clauses

if 'RandomAssignmentUBFinder' not in locals():
    class RandomAssignmentUBFinder:
        def __init__(self, n, edges, seed=None):
            self.n = n
            self.edges = edges
        
        def find_ub_random_search(self, max_iterations=1000, time_limit=30):
            return {'ub': 2 * (self.n - 1), 'assignment': None, 'iterations': 0, 'time': 0}

class BandwidthOptimizationSolver:
    """
    2D Bandwidth Minimization solver using SAT
    
    Problem: Place n vertices on n×n grid to minimize bandwidth
    bandwidth = max{|x_u - x_v| + |y_u - y_v| : (u,v) ∈ E}
    
    Two-phase approach:
    1. Random assignment to find upper bound
    2. SAT encoding with NSC and Thermometer constraints
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
        
        print(f"Created solver: n={n}, using {solver_type}")
    
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
        Uses NSC encoding for O(n²) complexity
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
        Step 1: Test if K is achievable using random placement
        
        Try random vertex placements to see if we can get bandwidth <= K
        This is much faster than SAT for finding upper bounds
        """
        print(f"\n--- Step 1: Testing K={K} with random placement ---")
        print(f"Looking for assignment with bandwidth <= {K}")
        
        ub_finder = RandomAssignmentUBFinder(self.n, self.edges, seed=42)
        
        result = ub_finder.find_ub_random_search(
            max_iterations=MAX_RANDOM_ITERATIONS, 
            time_limit=RANDOM_TIME_LIMIT
        )
        
        achieved_ub = result['ub']
        
        print(f"Random search results:")
        print(f"  Target: {K}")
        print(f"  Best found: {achieved_ub}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Time: {result['time']:.2f}s")
        
        if achieved_ub <= K:
            print(f"SUCCESS: Found placement with bandwidth {achieved_ub} <= {K}")
            return True
        else:
            print(f"FAILED: Best placement has bandwidth {achieved_ub} > {K}")
            return False
    
    def step2_encode_advanced_constraints(self, K):
        """
        Step 2: Test K using complete SAT encoding
        
        Encode the full problem as SAT:
        - Position constraints (each vertex gets one position)
        - Distance constraints (Manhattan distance encoding)  
        - Bandwidth constraints (all distances <= K)
        
        Formula: (Tx<=K) ∧ (Ty<=K) ∧ (Tx>=1 → Ty<=K-1) ∧ ... ∧ (Tx>=K → Ty<=0)
        """
        print(f"\n--- Step 2: Testing K={K} with SAT encoding ---")
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
    
    def encode_thermometer_bandwidth_constraints(self, K):
        """
        Encode bandwidth <= K using thermometer encoding
        
        For each edge: (Tx<=K) ∧ (Ty<=K) ∧ (Tx>=i → Ty<=K-i)
        """
        clauses = []
        
        print(f"Encoding thermometer for K={K}:")
        
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
                        ty_leq_ki = -Ty[K-i]  # Ty <= K-i
                    
                    if tx_geq_i is not None and ty_leq_ki is not None:
                        clauses.append([-tx_geq_i, ty_leq_ki])
                        print(f"    Tx>={i} → Ty<={K-i}")
        
        print(f"Generated {len(clauses)} thermometer clauses")
        return clauses
    
    def _extract_positions_from_model(self, model):
        """Extract vertex positions from SAT solution"""
        positions = {}
        for v in range(1, self.n + 1):
            # Find X position
            for pos in range(1, self.n + 1):
                var_id = self.X_vars[v][pos-1]
                if var_id in model and model[model.index(var_id)] > 0:
                    positions[f'X_{v}'] = pos
                    break
            
            # Find Y position
            for pos in range(1, self.n + 1):
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
        """Show solution details"""
        print(f"Vertex positions:")
        for v in range(1, self.n + 1):
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
        print(f"--- Verifying solution ---")
        
        positions = self._extract_positions_from_model(model)
        bandwidth, edge_distances = self._calculate_bandwidth(positions)
        self._print_solution_details(positions, edge_distances, bandwidth, K)
        
        return bandwidth <= K
    
    def _find_feasible_upper_bound_phase1(self, start_k, end_k):
        """Phase 1: Find feasible upper bound using random search"""
        print(f"\nPhase 1: Finding feasible UB with random search")
        
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
        Main solve function
        
        1. Random search to find upper bound
        2. SAT optimization to find minimum
        """
        if start_k is None:
            start_k = 1
        if end_k is None:
            end_k = DEFAULT_UB_MULTIPLIER * (self.n - 1)
        
        print(f"\n" + "="*60)
        print(f"2D BANDWIDTH OPTIMIZATION")
        print(f"Graph: {self.n} nodes, {len(self.edges)} edges")
        print(f"Testing range: K = {start_k} to {end_k}")
        print(f"="*60)
        
        # Phase 1: Find upper bound
        feasible_ub = self._find_feasible_upper_bound_phase1(start_k, end_k)
        if feasible_ub is None:
            return None
        
        # Phase 2: Optimize with SAT
        optimal_k = self._optimize_with_sat_phase2(feasible_ub)
        
        return optimal_k

def test_bandwidth_solver():
    """Test the solver on some small graphs"""
    print("=== BANDWIDTH SOLVER TESTS ===")
    
    # Triangle
    print(f"\n" + "="*40)
    print(f"Test 1: Triangle")
    print(f"="*40)
    
    n1 = 3
    edges1 = [(1, 2), (2, 3), (1, 3)]
    
    solver1 = BandwidthOptimizationSolver(n1, 'glucose42')
    solver1.set_graph_edges(edges1)
    solver1.create_position_variables()
    solver1.create_distance_variables()
    
    optimal1 = solver1.solve_bandwidth_optimization(start_k=1, end_k=4)
    print(f"Triangle result: {optimal1}")
    
    # Path
    print(f"\n" + "="*40)
    print(f"Test 2: Path")
    print(f"="*40)
    
    n2 = 4
    edges2 = [(1, 2), (2, 3), (3, 4)]
    
    solver2 = BandwidthOptimizationSolver(n2, 'cadical195')
    solver2.set_graph_edges(edges2)
    solver2.create_position_variables()
    solver2.create_distance_variables()
    
    optimal2 = solver2.solve_bandwidth_optimization(start_k=1, end_k=6)
    print(f"Path result: {optimal2}")
    
    # Cycle
    print(f"\n" + "="*40)
    print(f"Test 3: Cycle")
    print(f"="*40)
    
    n3 = 5
    edges3 = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
    
    solver3 = BandwidthOptimizationSolver(n3, 'glucose42')
    solver3.set_graph_edges(edges3)
    solver3.create_position_variables()
    solver3.create_distance_variables()
    
    optimal3 = solver3.solve_bandwidth_optimization(start_k=1, end_k=8)
    print(f"Cycle result: {optimal3}")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"RESULTS SUMMARY")
    print(f"="*60)
    print(f"Triangle (3 nodes): {optimal1}")
    print(f"Path (4 nodes): {optimal2}")
    print(f"Cycle (5 nodes): {optimal3}")
    print(f"="*60)

if __name__ == '__main__':
    test_bandwidth_solver()
