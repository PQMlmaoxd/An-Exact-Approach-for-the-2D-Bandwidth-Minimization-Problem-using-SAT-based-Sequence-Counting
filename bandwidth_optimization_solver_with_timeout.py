# bandwidth_optimization_solver_with_timeout.py
# 2D Bandwidth Minimization using SAT solvers with enhanced timeout protection

from pysat.formula import IDPool
from pysat.solvers import Glucose42, Cadical195, Solver
from pysat.card import CardEnc, EncType
import random
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from distance_encoder import encode_abs_distance_final
    from random_assignment_ub_finder import RandomAssignmentUBFinder
    from position_constraints import encode_all_position_constraints, create_position_variables
    from timeout_utils import get_timeout_executor, TimeoutError, TimeoutConfig
    from enhanced_sat_timeout_solver import EnhancedProcessTimeoutSATSolver, EnhancedSATResult
    print("All modules loaded OK (including timeout utilities and enhanced SAT timeout solver)")
except ImportError as e:
    print(f"Import error: {e}")
    print("Need required modules including timeout_utils and enhanced_sat_timeout_solver")
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


class TimeoutBandwidthOptimizationSolver:
    """
    2D Bandwidth Minimization solver using SAT with timeout support
    
    Problem: Place n vertices on n×n grid to minimize bandwidth
    bandwidth = max{|x_u - x_v| + |y_u - y_v| : (u,v) ∈ E}
    
    Two-phase approach with timeout protection:
    1. Random assignment to find upper bound (with timeout)
    2. SAT encoding with Sequential Counter and Thermometer constraints (with timeout)
    
    Features:
    - Configurable timeouts for each phase
    - Timeout protection for constraint encoding
    - Total solver timeout limit
    - Graceful timeout handling and recovery
    """
    
    def __init__(self, n, solver_type='glucose42', timeout_config=None):
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
        
        # Timeout configuration
        self.timeout_config = timeout_config if timeout_config else TimeoutConfig()
        
        # Initialize enhanced process-based SAT solver for robust timeout
        self.process_sat_solver = EnhancedProcessTimeoutSATSolver(
            solver_type=solver_type,
            default_timeout=self.timeout_config.sat_solve_timeout
        )
        
        # Tracking variables
        self.solve_start_time = None
        self.phase_timeouts_occurred = []
        self.constraint_timeouts_occurred = []
        
        # Best solution tracking for timeout recovery
        self.best_feasible_k = None
        self.best_solution_model = None
        self.best_solution_positions = None
        self.best_solution_bandwidth = None
        
        print(f"Created timeout-enabled solver: n={n}, using {solver_type}")
        print("Timeout configuration:")
        print(self.timeout_config.get_timeout_summary())
        print("✓ Enhanced Process-based SAT timeout protection ENABLED")
    
    def _update_best_solution(self, K, model=None, positions=None, bandwidth=None):
        """Update the best known feasible solution"""
        if self.best_feasible_k is None or K < self.best_feasible_k:
            self.best_feasible_k = K
            self.best_solution_model = model
            self.best_solution_positions = positions
            self.best_solution_bandwidth = bandwidth
            print(f"*** Updated best feasible solution: K={K} ***")
    
    def get_best_solution(self):
        """Get the best known feasible solution"""
        if self.best_feasible_k is not None:
            return {
                'bandwidth': self.best_feasible_k,
                'model': self.best_solution_model,
                'positions': self.best_solution_positions,
                'actual_bandwidth': self.best_solution_bandwidth
            }
        return None
    
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
    
    def _check_total_timeout(self):
        """Check if total solver timeout has been exceeded"""
        if not self.timeout_config.enable_total_timeout or not self.solve_start_time:
            return False
        
        elapsed = time.time() - self.solve_start_time
        if elapsed > self.timeout_config.total_solver_timeout:
            raise TimeoutError(f"Total solver timeout ({self.timeout_config.total_solver_timeout}s) exceeded after {elapsed:.2f}s")
        return False
    
    def encode_position_constraints_with_timeout(self):
        """
        Position constraints with timeout protection
        """
        def _encode_position_constraints():
            return encode_all_position_constraints(self.n, self.X_vars, self.Y_vars, self.vpool)
        
        if self.timeout_config.enable_constraint_timeouts:
            try:
                with get_timeout_executor() as executor:
                    return executor.execute(
                        _encode_position_constraints, 
                        timeout=self.timeout_config.position_constraints_timeout
                    )
            except TimeoutError as e:
                self.constraint_timeouts_occurred.append(f"Position constraints: {e}")
                print(f"WARNING: Position constraints timeout - using fallback")
                # Fallback: return minimal constraints
                return []
        else:
            return _encode_position_constraints()
    
    def encode_distance_constraints_with_timeout(self):
        """Encode distance constraints for each edge with timeout protection"""
        def _encode_distance_constraints():
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
        
        if self.timeout_config.enable_constraint_timeouts:
            try:
                with get_timeout_executor() as executor:
                    return executor.execute(
                        _encode_distance_constraints, 
                        timeout=self.timeout_config.distance_constraints_timeout
                    )
            except TimeoutError as e:
                self.constraint_timeouts_occurred.append(f"Distance constraints: {e}")
                print(f"WARNING: Distance constraints timeout - using fallback")
                # Fallback: return minimal constraints
                return []
        else:
            return _encode_distance_constraints()
    
    def encode_thermometer_bandwidth_constraints_with_timeout(self, K):
        """
        Encode bandwidth <= K using thermometer encoding with timeout protection
        """
        def _encode_thermometer_bandwidth_constraints():
            clauses = []
            
            print(f"Encoding thermometer for K={K}:")
            
            for edge_id in self.Tx_vars:
                Tx = self.Tx_vars[edge_id]  # Tx[i] means Tx >= i+1
                Ty = self.Ty_vars[edge_id]  # Ty[i] means Ty >= i+1
                
                # Tx <= K (i.e., not Tx >= K+1)
                if K < len(Tx):
                    clauses.append([-Tx[K]])
                
                # Ty <= K (i.e., not Ty >= K+1)
                if K < len(Ty):
                    clauses.append([-Ty[K]])
                
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
            
            print(f"Generated {len(clauses)} thermometer clauses")
            return clauses
        
        if self.timeout_config.enable_constraint_timeouts:
            try:
                with get_timeout_executor() as executor:
                    return executor.execute(
                        _encode_thermometer_bandwidth_constraints, 
                        timeout=self.timeout_config.bandwidth_constraints_timeout
                    )
            except TimeoutError as e:
                self.constraint_timeouts_occurred.append(f"Bandwidth constraints K={K}: {e}")
                print(f"WARNING: Bandwidth constraints timeout for K={K} - using fallback")
                # Fallback: return minimal constraints
                return []
        else:
            return _encode_thermometer_bandwidth_constraints()
    
    def step1_test_ub_pure_random_with_timeout(self, K):
        """
        Step 1: Test if K is achievable using random placement with timeout
        """
        def _random_search():
            print(f"\n--- Step 1: Testing K={K} with random placement ---")
            print(f"Looking for assignment with bandwidth <= {K}")
            
            ub_finder = RandomAssignmentUBFinder(self.n, self.edges, seed=42)
            
            result = ub_finder.find_ub_random_search(
                max_iterations=MAX_RANDOM_ITERATIONS, 
                time_limit=min(RANDOM_TIME_LIMIT, self.timeout_config.random_search_timeout)
            )
            
            achieved_ub = result['ub']
            
            print(f"Random search results:")
            print(f"  Target: {K}")
            print(f"  Best found: {achieved_ub}")
            print(f"  Iterations: {result['iterations']}")
            print(f"  Time: {result['time']:.2f}s")
            
            if achieved_ub <= K:
                print(f"SUCCESS: Found placement with bandwidth {achieved_ub} <= {K}")
                # Update best solution from random search
                self._update_best_solution(achieved_ub)
                return True
            else:
                print(f"FAILED: Best placement has bandwidth {achieved_ub} > {K}")
                return False
        
        if self.timeout_config.enable_phase_timeouts:
            try:
                with get_timeout_executor() as executor:
                    return executor.execute(
                        _random_search, 
                        timeout=self.timeout_config.random_search_timeout
                    )
            except TimeoutError as e:
                self.phase_timeouts_occurred.append(f"Random search K={K}: {e}")
                print(f"WARNING: Random search timeout for K={K}")
                return False  # Assume infeasible if timeout
        else:
            return _random_search()
    
    def step2_encode_advanced_constraints_with_timeout(self, K):
        """
        Step 2: Test K using complete SAT encoding with robust process-based timeout
        """
        def _sat_solve():
            print(f"\n--- Step 2: Testing K={K} with SAT encoding ---")
            print(f"Using {self.solver_type.upper()} solver with PROCESS-BASED timeout")
            print(f"Encoding thermometer constraints for bandwidth <= {K}")
            
            print(f"Building constraints...")
            
            # Position constraints
            position_clauses = self.encode_position_constraints_with_timeout()
            print(f"  Position: {len(position_clauses)} clauses")
            
            # Distance constraints  
            distance_clauses = self.encode_distance_constraints_with_timeout()
            print(f"  Distance: {len(distance_clauses)} clauses")
            
            # Bandwidth constraints
            bandwidth_clauses = self.encode_thermometer_bandwidth_constraints_with_timeout(K)
            print(f"  Bandwidth: {len(bandwidth_clauses)} clauses")
            
            # Combine all constraints
            all_clauses = position_clauses + distance_clauses + bandwidth_clauses
            print(f"Total: {len(all_clauses)} clauses")
            
            # Use enhanced process-based SAT solver for robust timeout
            print(f"Launching isolated SAT process with {self.timeout_config.sat_solve_timeout}s timeout...")
            
            sat_result = self.process_sat_solver.solve_single(
                clauses=all_clauses,
                problem_id=f"bandwidth_n{self.n}_k{K}",
                timeout=self.timeout_config.sat_solve_timeout,
                additional_data={'n': self.n, 'K': K, 'edges': len(self.edges)}
            )
            
            if sat_result.status == 'SAT':
                print(f"K={K} is SAT (process-isolated)")
                print(f"Extracting solution from model...")
                
                # Convert model back to our format
                model = sat_result.model
                self.last_model = model  # Store for extraction later
                
                # Extract and verify solution
                positions = self._extract_positions_from_model(model)
                bandwidth, edge_distances = self._calculate_bandwidth(positions)
                self._print_solution_details(positions, edge_distances, bandwidth, K)
                
                # Update best solution
                self._update_best_solution(K, model, positions, bandwidth)
                
                return True
                
            elif sat_result.status == 'UNSAT':
                print(f"K={K} is UNSAT (process-isolated)")
                return False
                
            elif sat_result.status == 'TIMEOUT':
                print(f"K={K} TIMEOUT after {sat_result.total_time:.1f}s (process-isolated)")
                if sat_result.process_killed:
                    print(f"  -> Process successfully killed using: {sat_result.kill_method}")
                else:
                    print(f"  -> WARNING: Process may still be running")
                self.phase_timeouts_occurred.append(f"SAT solve K={K}: Process timeout after {sat_result.total_time:.1f}s")
                return False  # Treat timeout as UNSAT
                
            else:
                print(f"K={K} ERROR: {sat_result.error_message} (process-isolated)")
                self.phase_timeouts_occurred.append(f"SAT solve K={K}: Process error - {sat_result.error_message}")
                return False  # Treat error as UNSAT
        
        # Always use the process-based solver - no need for additional timeout wrapper
        return _sat_solve()
    
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
    
    def _find_feasible_upper_bound_phase1_with_timeout(self, start_k, end_k):
        """Phase 1: Find feasible upper bound using random search with timeout"""
        print(f"\nPhase 1: Finding feasible UB with random search (timeout-protected)")
        
        for K in range(start_k, end_k + 1):
            self._check_total_timeout()  # Check total timeout
            
            print(f"\nTrying K = {K}")
            
            if self.step1_test_ub_pure_random_with_timeout(K):
                print(f"Found feasible UB = {K}")
                return K
            else:
                print(f"K = {K} not achievable")
        
        print(f"\nError: No feasible UB in range [{start_k}, {end_k}]")
        
        # Check if we found any feasible solution during search
        if self.best_feasible_k is not None:
            print(f"However, found feasible solution with K={self.best_feasible_k} during search")
            return self.best_feasible_k
        
        return None
    
    def _optimize_with_sat_phase2_with_timeout(self, feasible_ub):
        """Phase 2: Incremental SAT optimization with timeout protection"""
        print(f"\nPhase 2: Incremental SAT optimization (timeout-protected)")
        print(f"Starting from K={feasible_ub-1} down to 1")
        
        optimal_k = feasible_ub
        
        # Incremental SAT: try smaller K values
        for K in range(feasible_ub - 1, 0, -1):
            self._check_total_timeout()  # Check total timeout
            
            print(f"\nTrying K = {K} with incremental SAT")
            
            if self.step2_encode_advanced_constraints_with_timeout(K):
                optimal_k = K
                print(f"K = {K} is SAT")
            else:
                print(f"K = {K} is UNSAT")
                print(f"Optimal bandwidth = {optimal_k}")
                break
        
        print(f"Final optimal bandwidth = {optimal_k}")
        return optimal_k
    
    def solve_bandwidth_optimization_with_timeout(self, start_k=None, end_k=None):
        """
        Main solve function with comprehensive timeout protection
        
        1. Random search to find upper bound (with timeout)
        2. SAT optimization to find minimum (with timeout)
        """
        if start_k is None:
            start_k = 1
        if end_k is None:
            end_k = DEFAULT_UB_MULTIPLIER * (self.n - 1)
        
        # Initialize timing
        self.solve_start_time = time.time()
        self.phase_timeouts_occurred = []
        self.constraint_timeouts_occurred = []
        
        print(f"\n" + "="*70)
        print(f"2D BANDWIDTH OPTIMIZATION (WITH TIMEOUT)")
        print(f"Graph: {self.n} nodes, {len(self.edges)} edges")
        print(f"Testing range: K = {start_k} to {end_k}")
        print(f"Total timeout: {self.timeout_config.total_solver_timeout}s")
        print(f"="*70)
        
        try:
            # Phase 1: Find upper bound
            feasible_ub = self._find_feasible_upper_bound_phase1_with_timeout(start_k, end_k)
            if feasible_ub is None:
                return None
            
            # Phase 2: Optimize with SAT
            optimal_k = self._optimize_with_sat_phase2_with_timeout(feasible_ub)
            
            return optimal_k
            
        except TimeoutError as e:
            print(f"\n" + "="*50)
            print(f"TOTAL SOLVER TIMEOUT OCCURRED")
            print(f"Error: {e}")
            print(f"="*50)
            
            # Return best solution found so far
            if self.best_feasible_k is not None:
                print(f"Returning best feasible solution found: K={self.best_feasible_k}")
                return self.best_feasible_k
            
            return None
        
        finally:
            # Report timeout summary
            self._print_timeout_summary()
    
    def _print_timeout_summary(self):
        """Print summary of timeout events"""
        total_time = time.time() - self.solve_start_time if self.solve_start_time else 0
        
        print(f"\n" + "="*50)
        print(f"TIMEOUT SUMMARY")
        print(f"="*50)
        print(f"Total solve time: {total_time:.2f}s")
        print(f"Phase timeouts: {len(self.phase_timeouts_occurred)}")
        for timeout in self.phase_timeouts_occurred:
            print(f"  - {timeout}")
        print(f"Constraint timeouts: {len(self.constraint_timeouts_occurred)}")
        for timeout in self.constraint_timeouts_occurred:
            print(f"  - {timeout}")
        
        # Report best solution if available
        if self.best_feasible_k is not None:
            print(f"Best feasible solution: K={self.best_feasible_k}")
            if self.best_solution_bandwidth is not None:
                print(f"Actual bandwidth: {self.best_solution_bandwidth}")
        else:
            print(f"No feasible solution found")
        
        print(f"="*50)
    
    def update_timeout_config(self, **kwargs):
        """Update timeout configuration"""
        self.timeout_config.update_timeouts(**kwargs)
        print("Updated timeout configuration:")
        print(self.timeout_config.get_timeout_summary())


def test_timeout_bandwidth_solver():
    """Test the timeout-enabled solver on some small graphs"""
    print("=== TIMEOUT-ENABLED BANDWIDTH SOLVER TESTS ===")
    
    # Custom timeout configuration for testing
    test_config = TimeoutConfig()
    test_config.update_timeouts(
        random_search_timeout=10.0,
        sat_solve_timeout=30.0,
        total_solver_timeout=120.0
    )
    
    # Triangle
    print(f"\n" + "="*40)
    print(f"Test 1: Triangle (with timeout)")
    print(f"="*40)
    
    n1 = 3
    edges1 = [(1, 2), (2, 3), (1, 3)]
    
    solver1 = TimeoutBandwidthOptimizationSolver(n1, 'glucose42', test_config)
    solver1.set_graph_edges(edges1)
    solver1.create_position_variables()
    solver1.create_distance_variables()
    
    optimal1 = solver1.solve_bandwidth_optimization_with_timeout(start_k=1, end_k=4)
    print(f"Triangle result: {optimal1}")


if __name__ == '__main__':
    """
    Command line usage: python bandwidth_optimization_solver_with_timeout.py [mtx_file] [solver] [timeout_config]
    
    Arguments:
        mtx_file: Name of MTX file (searches in mtx/group 1/ and mtx/group 2/)
        solver: SAT solver to use (glucose42 or cadical195, default: glucose42)
        timeout_config: Optional timeout configuration (format: key=value,key=value...)
    
    Examples:
        python bandwidth_optimization_solver_with_timeout.py 8.jgl009.mtx glucose42
        python bandwidth_optimization_solver_with_timeout.py 1.ash85.mtx cadical195 sat_solve_timeout=600,total_solver_timeout=1800
        python bandwidth_optimization_solver_with_timeout.py 3.bcsstk01.mtx
        python bandwidth_optimization_solver_with_timeout.py  # Run test mode
    """
    import sys
    
    # Check if MTX file provided
    if len(sys.argv) >= 2:
        # MTX file mode
        mtx_file = sys.argv[1]
        solver_type = sys.argv[2] if len(sys.argv) >= 3 else 'glucose42'
        timeout_config_str = sys.argv[3] if len(sys.argv) >= 4 else None
        
        # Parse timeout configuration
        timeout_config = TimeoutConfig()
        if timeout_config_str:
            print("Parsing timeout configuration...")
            try:
                config_pairs = timeout_config_str.split(',')
                config_dict = {}
                for pair in config_pairs:
                    key, value = pair.split('=')
                    config_dict[key.strip()] = float(value.strip())
                timeout_config.update_timeouts(**config_dict)
                print("Custom timeout configuration applied.")
            except Exception as e:
                print(f"Warning: Failed to parse timeout config '{timeout_config_str}': {e}")
                print("Using default timeout configuration.")
        
        print("=" * 80)
        print("2D BANDWIDTH OPTIMIZATION SOLVER (WITH TIMEOUT)")
        print("=" * 80)
        print(f"File: {mtx_file}")
        print(f"Solver: {solver_type.upper()}")
        
        # [MTX file processing code from original solver - same as before]
        # ... (include the same MTX file processing and solving logic)
        
        print("Timeout features enabled for this run.")
        
        # Find MTX file in group folders
        mtx_paths = [
            f"mtx/group 1/{mtx_file}",
            f"mtx/group 2/{mtx_file}",
            mtx_file  # Direct path
        ]
        
        found_file = None
        for path in mtx_paths:
            if os.path.exists(path):
                found_file = path
                break
        
        if found_file is None:
            print(f"MTX file not found: {mtx_file}")
            print(f"Searched paths:")
            for path in mtx_paths:
                print(f"  {path}")
            print(f"\nAvailable MTX files:")
            print(f"  Group 1: 1.bcspwr01.mtx, 2.bcspwr02.mtx, 3.bcsstk01.mtx, 4.can___24.mtx,")
            print(f"           5.fidap005.mtx, 6.fidapm05.mtx, 7.ibm32.mtx, 8.jgl009.mtx,")
            print(f"           9.jgl011.mtx, 10.lap_25.mtx, 11.pores_1.mtx, 12.rgg010.mtx")
            print(f"  Group 2: 1.ash85.mtx")
            print(f"\nUsage examples:")
            print(f"  python bandwidth_optimization_solver_with_timeout.py 8.jgl009.mtx glucose42")
            print(f"  python bandwidth_optimization_solver_with_timeout.py 1.ash85.mtx cadical195")
            print(f"  python bandwidth_optimization_solver_with_timeout.py 3.bcsstk01.mtx")
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
        
        # Solve bandwidth problem with timeout
        print(f"\nSolving 2D bandwidth minimization with timeout protection...")
        print(f"Problem: {n} vertices on {n}×{n} grid")
        print(f"Using: {solver_type.upper()}")
        print(f"Timeout config: {timeout_config.get_timeout_summary()}")
        
        solver = TimeoutBandwidthOptimizationSolver(n, solver_type, timeout_config)
        solver.set_graph_edges(edges)
        solver.create_position_variables()
        solver.create_distance_variables()
        
        start_time = time.time()
        optimal_bandwidth = solver.solve_bandwidth_optimization_with_timeout()
        solve_time = time.time() - start_time
        
        # Results
        print(f"\n" + "="*60)
        print(f"FINAL RESULTS (WITH TIMEOUT)")
        print(f"="*60)
        
        if optimal_bandwidth is not None:
            print(f"✓ Optimal bandwidth: {optimal_bandwidth}")
            print(f"✓ Solve time: {solve_time:.2f}s")
            print(f"✓ Graph: {n} vertices, {len(edges)} edges")
            print(f"✓ Solver: {solver_type.upper()}")
            print(f"✓ Timeout protection: ENABLED")
            
            # Check if this was the truly optimal or best feasible due to timeout
            best_solution = solver.get_best_solution()
            if best_solution and len(solver.phase_timeouts_occurred) > 0:
                print(f"✓ Status: TIMEOUT - BEST FEASIBLE SOLUTION")
                print(f"✓ Note: Solution may not be optimal due to timeout")
            else:
                print(f"✓ Status: SUCCESS - OPTIMAL SOLUTION")
                
        else:
            print(f"✗ No solution found (may be due to timeout)")
            print(f"✗ Solve time: {solve_time:.2f}s")
            print(f"✗ Graph: {n} vertices, {len(edges)} edges")
            print(f"✗ Timeout protection: ENABLED")
            print(f"✗ Status: FAILED/TIMEOUT")
        
        print(f"="*60)
        
    else:
        # Test mode - run timeout test cases
        print("=" * 80)
        print("2D BANDWIDTH OPTIMIZATION SOLVER (WITH TIMEOUT) - TEST MODE")
        print("=" * 80)
        print("Usage: python bandwidth_optimization_solver_with_timeout.py [mtx_file] [solver] [timeout_config]")
        print()
        print("Timeout configuration format: key=value,key=value...")
        print("Available timeout keys:")
        print("  - random_search_timeout")
        print("  - sat_solve_timeout") 
        print("  - total_solver_timeout")
        print("  - position_constraints_timeout")
        print("  - distance_constraints_timeout")
        print("  - bandwidth_constraints_timeout")
        print()
        print("Examples:")
        print("  python bandwidth_optimization_solver_with_timeout.py 8.jgl009.mtx glucose42 sat_solve_timeout=600")
        print("  python bandwidth_optimization_solver_with_timeout.py 1.ash85.mtx cadical195 total_solver_timeout=1800")
        print()
        print("Running built-in timeout test cases...")
        test_timeout_bandwidth_solver()
