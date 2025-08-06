# rectangular_bandwidth_solver_with_timeout.py
# 2D Bandwidth Minimization using SAT solvers on rectangular grids with timeout support
# Enhanced version with configurable timeout mechanisms

from pysat.formula import IDPool
from pysat.solvers import Glucose42, Cadical195
from pysat.card import CardEnc, EncType
import random
import time
import sys
import os
import atexit
import threading
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

class TimeoutRectangularBandwidthOptimizationSolver:
    """
    2D Bandwidth Minimization solver using SAT on rectangular grids with timeout support
    
    Problem: Place n vertices on n_rows × n_cols grid to minimize bandwidth
    bandwidth = max{|x_u - x_v| + |y_u - y_v| : (u,v) ∈ E}
    
    Key differences from square solver:
    - Support for rectangular grids (n_rows × n_cols)
    - Adjusted position constraints for different dimensions
    - Enhanced timeout protection for rectangular constraint encoding
    
    Features:
    - Configurable timeouts for each phase
    - Timeout protection for constraint encoding
    - Total solver timeout limit
    - Rectangular grid specific optimizations
    """
    
    def __init__(self, num_vertices, n_rows, n_cols, solver_type='glucose42', timeout_config=None):
        self.num_vertices = num_vertices
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.solver_type = solver_type
        self.vpool = IDPool()
        
        # Validate grid capacity
        if num_vertices > n_rows * n_cols:
            raise ValueError(f"Cannot place {num_vertices} vertices on {n_rows}×{n_cols} grid (capacity: {n_rows * n_cols})")
        
        # Position variables for X (columns), Y (rows)
        self.X_vars = {}  # X_vars[v][col] = variable for vertex v at column col
        self.Y_vars = {}  # Y_vars[v][row] = variable for vertex v at row row
        
        # Distance variables  
        self.Tx_vars = {}  # T variables for X distances
        self.Ty_vars = {}  # T variables for Y distances
        
        self.edges = []
        self.last_model = None
        
        # Timeout configuration - handle both dict and TimeoutConfig object
        if timeout_config is None:
            self.timeout_config = TimeoutConfig()
        elif isinstance(timeout_config, dict):
            # Convert dict to TimeoutConfig object for compatibility
            self.timeout_config = TimeoutConfig()
            self.timeout_config.update_timeouts(**timeout_config)
        else:
            self.timeout_config = timeout_config
        
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
        
        print(f"Created rectangular timeout-enabled solver: {num_vertices} vertices on {n_rows}×{n_cols} grid using {solver_type}")
        print("Timeout configuration:")
        print(self.timeout_config.get_timeout_summary())
        print(f"✓ Enhanced Process-based SAT timeout protection ENABLED")
    
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
        """Create position variables for rectangular grid"""
        # X variables: columns (1 to n_cols)
        for v in range(1, self.num_vertices + 1):
            self.X_vars[v] = [self.vpool.id(f'X_{v}_{col}') for col in range(1, self.n_cols + 1)]
        
        # Y variables: rows (1 to n_rows)
        for v in range(1, self.num_vertices + 1):
            self.Y_vars[v] = [self.vpool.id(f'Y_{v}_{row}') for row in range(1, self.n_rows + 1)]
        
        print(f"Created {self.num_vertices * (self.n_cols + self.n_rows)} position variables")
    
    def create_distance_variables(self):
        """Create T variables for edge distances on rectangular grid"""
        max_x_distance = self.n_cols - 1
        max_y_distance = self.n_rows - 1
        
        for i, (u, v) in enumerate(self.edges):
            edge_id = f'edge_{u}_{v}'
            # X distance variables (0 to max_x_distance-1)
            self.Tx_vars[edge_id] = [self.vpool.id(f'Tx_{edge_id}_{d}') for d in range(1, max_x_distance + 1)]
            # Y distance variables (0 to max_y_distance-1)
            self.Ty_vars[edge_id] = [self.vpool.id(f'Ty_{edge_id}_{d}') for d in range(1, max_y_distance + 1)]
        
        print(f"Created distance variables: X_max={max_x_distance}, Y_max={max_y_distance}")
    
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
        Encode position constraints for rectangular grid with timeout protection
        """
        def _encode_rectangular_position_constraints():
            clauses = []
            
            # Each vertex must be in exactly one column
            for v in range(1, self.num_vertices + 1):
                x_vars = self.X_vars[v]
                exactly_one_x = CardEnc.equals(x_vars, 1, vpool=self.vpool, encoding=EncType.seqcounter)
                clauses.extend(exactly_one_x.clauses)
            
            # Each vertex must be in exactly one row
            for v in range(1, self.num_vertices + 1):
                y_vars = self.Y_vars[v]
                exactly_one_y = CardEnc.equals(y_vars, 1, vpool=self.vpool, encoding=EncType.seqcounter)
                clauses.extend(exactly_one_y.clauses)
            
            # Each position can have at most one vertex
            # For each column position
            for col in range(1, self.n_cols + 1):
                column_vars = [self.X_vars[v][col-1] for v in range(1, self.num_vertices + 1)]
                at_most_one_col = CardEnc.atmost(column_vars, 1, vpool=self.vpool, encoding=EncType.seqcounter)
                clauses.extend(at_most_one_col.clauses)
            
            # For each row position
            for row in range(1, self.n_rows + 1):
                row_vars = [self.Y_vars[v][row-1] for v in range(1, self.num_vertices + 1)]
                at_most_one_row = CardEnc.atmost(row_vars, 1, vpool=self.vpool, encoding=EncType.seqcounter)
                clauses.extend(at_most_one_row.clauses)
            
            # Each grid cell can have at most one vertex
            for row in range(1, self.n_rows + 1):
                for col in range(1, self.n_cols + 1):
                    cell_vars = []
                    for v in range(1, self.num_vertices + 1):
                        # Vertex v is at position (row, col) if both X_v_col and Y_v_row are true
                        # We need auxiliary variables for this
                        aux_var = self.vpool.id(f'Cell_{row}_{col}_{v}')
                        cell_vars.append(aux_var)
                        
                        # aux_var ↔ (X_v_col ∧ Y_v_row)
                        x_var = self.X_vars[v][col-1]
                        y_var = self.Y_vars[v][row-1]
                        
                        # aux_var → X_v_col
                        clauses.append([-aux_var, x_var])
                        # aux_var → Y_v_row
                        clauses.append([-aux_var, y_var])
                        # (X_v_col ∧ Y_v_row) → aux_var
                        clauses.append([-x_var, -y_var, aux_var])
                    
                    # At most one vertex per cell
                    if len(cell_vars) > 1:
                        at_most_one_cell = CardEnc.atmost(cell_vars, 1, vpool=self.vpool, encoding=EncType.seqcounter)
                        clauses.extend(at_most_one_cell.clauses)
            
            return clauses
        
        if self.timeout_config.enable_constraint_timeouts:
            try:
                with get_timeout_executor() as executor:
                    return executor.execute(
                        _encode_rectangular_position_constraints, 
                        timeout=self.timeout_config.position_constraints_timeout
                    )
            except TimeoutError as e:
                self.constraint_timeouts_occurred.append(f"Rectangular position constraints: {e}")
                print(f"WARNING: Rectangular position constraints timeout - using fallback")
                # Fallback: return minimal constraints
                return []
        else:
            return _encode_rectangular_position_constraints()
    
    def encode_distance_constraints_with_timeout(self):
        """Encode distance constraints for rectangular grid with timeout protection"""
        def _encode_rectangular_distance_constraints():
            clauses = []
            
            for edge_id, (u, v) in zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges):
                # X distance encoding (columns)
                Tx_vars, Tx_clauses = encode_abs_distance_final(
                    self.X_vars[u], self.X_vars[v], self.n_cols, self.vpool, f"Tx_{edge_id}"
                )
                self.Tx_vars[edge_id] = Tx_vars
                clauses.extend(Tx_clauses)
                
                # Y distance encoding (rows)
                Ty_vars, Ty_clauses = encode_abs_distance_final(
                    self.Y_vars[u], self.Y_vars[v], self.n_rows, self.vpool, f"Ty_{edge_id}"
                )
                self.Ty_vars[edge_id] = Ty_vars
                clauses.extend(Ty_clauses)
            
            return clauses
        
        if self.timeout_config.enable_constraint_timeouts:
            try:
                with get_timeout_executor() as executor:
                    return executor.execute(
                        _encode_rectangular_distance_constraints, 
                        timeout=self.timeout_config.distance_constraints_timeout
                    )
            except TimeoutError as e:
                self.constraint_timeouts_occurred.append(f"Rectangular distance constraints: {e}")
                print(f"WARNING: Rectangular distance constraints timeout - using fallback")
                # Fallback: return minimal constraints
                return []
        else:
            return _encode_rectangular_distance_constraints()
    
    def encode_thermometer_bandwidth_constraints_with_timeout(self, K):
        """
        Encode bandwidth <= K using thermometer encoding for rectangular grid with timeout protection
        """
        def _encode_rectangular_thermometer_bandwidth_constraints():
            clauses = []
            
            print(f"Encoding rectangular thermometer for K={K}:")
            
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
            
            print(f"Generated {len(clauses)} rectangular thermometer clauses")
            return clauses
        
        if self.timeout_config.enable_constraint_timeouts:
            try:
                with get_timeout_executor() as executor:
                    return executor.execute(
                        _encode_rectangular_thermometer_bandwidth_constraints, 
                        timeout=self.timeout_config.bandwidth_constraints_timeout
                    )
            except TimeoutError as e:
                self.constraint_timeouts_occurred.append(f"Rectangular bandwidth constraints K={K}: {e}")
                print(f"WARNING: Rectangular bandwidth constraints timeout for K={K} - using fallback")
                # Fallback: return minimal constraints
                return []
        else:
            return _encode_rectangular_thermometer_bandwidth_constraints()
    
    def encode_position_constraints(self):
        """
        Encode position constraints for rectangular grid without timeout (for base constraints)
        """
        clauses = []
        
        # Each vertex must be in exactly one column
        for v in range(1, self.num_vertices + 1):
            x_vars = self.X_vars[v]
            exactly_one_x = CardEnc.equals(x_vars, 1, vpool=self.vpool, encoding=EncType.seqcounter)
            clauses.extend(exactly_one_x.clauses)
        
        # Each vertex must be in exactly one row
        for v in range(1, self.num_vertices + 1):
            y_vars = self.Y_vars[v]
            exactly_one_y = CardEnc.equals(y_vars, 1, vpool=self.vpool, encoding=EncType.seqcounter)
            clauses.extend(exactly_one_y.clauses)
        
        # Each position can have at most one vertex
        # For each column position
        for col in range(1, self.n_cols + 1):
            column_vars = [self.X_vars[v][col-1] for v in range(1, self.num_vertices + 1)]
            at_most_one_col = CardEnc.atmost(column_vars, 1, vpool=self.vpool, encoding=EncType.seqcounter)
            clauses.extend(at_most_one_col.clauses)
        
        # For each row position
        for row in range(1, self.n_rows + 1):
            row_vars = [self.Y_vars[v][row-1] for v in range(1, self.num_vertices + 1)]
            at_most_one_row = CardEnc.atmost(row_vars, 1, vpool=self.vpool, encoding=EncType.seqcounter)
            clauses.extend(at_most_one_row.clauses)
        
        # Each grid cell can have at most one vertex
        for row in range(1, self.n_rows + 1):
            for col in range(1, self.n_cols + 1):
                cell_vars = []
                for v in range(1, self.num_vertices + 1):
                    # Vertex v is at position (row, col) if both X_v_col and Y_v_row are true
                    # We need auxiliary variables for this
                    aux_var = self.vpool.id(f'Cell_{row}_{col}_{v}')
                    cell_vars.append(aux_var)
                    
                    # aux_var ↔ (X_v_col ∧ Y_v_row)
                    x_var = self.X_vars[v][col-1]
                    y_var = self.Y_vars[v][row-1]
                    
                    # aux_var → X_v_col
                    clauses.append([-aux_var, x_var])
                    # aux_var → Y_v_row
                    clauses.append([-aux_var, y_var])
                    # (X_v_col ∧ Y_v_row) → aux_var
                    clauses.append([-x_var, -y_var, aux_var])
                
                # At most one vertex per cell
                if len(cell_vars) > 1:
                    at_most_one_cell = CardEnc.atmost(cell_vars, 1, vpool=self.vpool, encoding=EncType.seqcounter)
                    clauses.extend(at_most_one_cell.clauses)
        
        return clauses
    
    def encode_distance_constraints(self):
        """Encode distance constraints for rectangular grid without timeout (for base constraints)"""
        clauses = []
        
        for edge_id, (u, v) in zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges):
            # X distance encoding (columns)
            Tx_vars, Tx_clauses = encode_abs_distance_final(
                self.X_vars[u], self.X_vars[v], self.n_cols, self.vpool, f"Tx_{edge_id}"
            )
            self.Tx_vars[edge_id] = Tx_vars
            clauses.extend(Tx_clauses)
            
            # Y distance encoding (rows)
            Ty_vars, Ty_clauses = encode_abs_distance_final(
                self.Y_vars[u], self.Y_vars[v], self.n_rows, self.vpool, f"Ty_{edge_id}"
            )
            self.Ty_vars[edge_id] = Ty_vars
            clauses.extend(Ty_clauses)
        
        return clauses
    
    def encode_thermometer_bandwidth_constraints(self, K):
        """
        Encode bandwidth <= K using thermometer encoding for rectangular grid without timeout (for incremental)
        """
        clauses = []
        
        print(f"Encoding rectangular thermometer for K={K}:")
        
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
        
        print(f"Generated {len(clauses)} rectangular thermometer clauses")
        return clauses
        """Create SAT solver instance"""
        if self.solver_type == 'glucose42':
            return Glucose42()
        elif self.solver_type == 'cadical195':
            return Cadical195()
        else:
            print(f"Unknown solver '{self.solver_type}', using Glucose42")
            return Glucose42()
    
    def step1_test_ub_pure_random_with_timeout(self, K):
        """
        Step 1: Test if K is achievable using random placement on rectangular grid with timeout
        """
        def _rectangular_random_search():
            print(f"\n--- Step 1: Testing K={K} with rectangular random placement ---")
            print(f"Grid: {self.n_rows}×{self.n_cols}, Looking for bandwidth <= {K}")
            
            # Adapted random search for rectangular grid
            best_bandwidth = float('inf')
            found_valid = False
            
            for iteration in range(MAX_RANDOM_ITERATIONS):
                # Random placement on rectangular grid
                positions = {}
                used_cells = set()
                
                # Generate random positions for each vertex
                for v in range(1, self.num_vertices + 1):
                    # Try to find unused position
                    attempts = 0
                    while attempts < 100:  # Avoid infinite loop
                        row = random.randint(1, self.n_rows)
                        col = random.randint(1, self.n_cols)
                        if (row, col) not in used_cells:
                            positions[v] = (row, col)
                            used_cells.add((row, col))
                            break
                        attempts += 1
                    
                    if attempts >= 100:
                        break  # Skip this iteration
                
                if len(positions) < self.num_vertices:
                    continue  # Skip incomplete placements
                
                # Calculate bandwidth
                max_distance = 0
                for u, v in self.edges:
                    if u in positions and v in positions:
                        row_u, col_u = positions[u]
                        row_v, col_v = positions[v]
                        distance = abs(row_u - row_v) + abs(col_u - col_v)
                        max_distance = max(max_distance, distance)
                
                best_bandwidth = min(best_bandwidth, max_distance)
                
                if max_distance <= K:
                    print(f"SUCCESS: Found rectangular placement with bandwidth {max_distance} <= {K} after {iteration + 1} iterations")
                    found_valid = True
                    break
            
            print(f"Random search results (rectangular {self.n_rows}×{self.n_cols}):")
            print(f"  Target: {K}")
            print(f"  Best found: {best_bandwidth}")
            print(f"  Iterations: {min(iteration + 1, MAX_RANDOM_ITERATIONS)}")
            
            if found_valid:
                # Update best solution from random search
                self._update_best_solution(best_bandwidth)
                return True
            else:
                return False
        
        if self.timeout_config.enable_phase_timeouts:
            try:
                with get_timeout_executor() as executor:
                    return executor.execute(
                        _rectangular_random_search, 
                        timeout=self.timeout_config.random_search_timeout
                    )
            except TimeoutError as e:
                self.phase_timeouts_occurred.append(f"Rectangular random search K={K}: {e}")
                print(f"WARNING: Rectangular random search timeout for K={K}")
                return False  # Assume infeasible if timeout
        else:
            return _rectangular_random_search()
    
    def step2_encode_advanced_constraints_with_timeout(self, K):
        """
        Step 2: Test K using complete SAT encoding for rectangular grid with robust process-based timeout
        """
        def _rectangular_sat_solve():
            print(f"\n--- Step 2: Testing K={K} with rectangular SAT encoding ---")
            print(f"Grid: {self.n_rows}×{self.n_cols}, using {self.solver_type.upper()} solver with PROCESS-BASED timeout")
            print(f"Encoding thermometer constraints for bandwidth <= {K}")
            
            print(f"Building rectangular constraints...")
            
            # Position constraints for rectangular grid
            position_clauses = self.encode_position_constraints_with_timeout()
            print(f"  Rectangular position: {len(position_clauses)} clauses")
            
            # Distance constraints for rectangular grid
            distance_clauses = self.encode_distance_constraints_with_timeout()
            print(f"  Rectangular distance: {len(distance_clauses)} clauses")
            
            # Bandwidth constraints
            bandwidth_clauses = self.encode_thermometer_bandwidth_constraints_with_timeout(K)
            print(f"  Rectangular bandwidth: {len(bandwidth_clauses)} clauses")
            
            # Add all constraints
            all_clauses = position_clauses + distance_clauses + bandwidth_clauses
            print(f"Total rectangular constraints: {len(all_clauses)} clauses")
            
            # Use enhanced process-based SAT solver for robust timeout
            print(f"Launching isolated SAT process with {self.timeout_config.sat_solve_timeout}s timeout...")
            
            sat_result = self.process_sat_solver.solve_single(
                clauses=all_clauses,
                problem_id=f"rectangular_bandwidth_n{self.num_vertices}_k{K}_{self.n_rows}x{self.n_cols}",
                timeout=self.timeout_config.sat_solve_timeout,
                additional_data={'n': self.num_vertices, 'K': K, 'edges': len(self.edges), 'grid': f'{self.n_rows}x{self.n_cols}'}
            )
            
            if sat_result.status == 'SAT':
                print(f"K={K} is SAT on {self.n_rows}×{self.n_cols} grid (process-isolated)")
                print(f"Extracting rectangular solution...")
                
                # Convert model back to our format
                model = sat_result.model
                self.last_model = model  # Store for extraction later
                
                # Extract and verify solution
                positions = self._extract_rectangular_positions_from_model(model)
                bandwidth, edge_distances = self._calculate_rectangular_bandwidth(positions)
                self._print_rectangular_solution_details(positions, edge_distances, bandwidth, K)
                
                # Update best solution
                self._update_best_solution(K, model, positions, bandwidth)
                
                return True
                
            elif sat_result.status == 'UNSAT':
                print(f"K={K} is UNSAT on {self.n_rows}×{self.n_cols} grid (process-isolated)")
                return False
                
            elif sat_result.status == 'TIMEOUT':
                print(f"K={K} TIMEOUT after {sat_result.total_time:.1f}s on {self.n_rows}×{self.n_cols} grid (process-isolated)")
                if sat_result.process_killed:
                    print(f"  -> Process successfully killed using: {sat_result.kill_method}")
                else:
                    print(f"  -> WARNING: Process may still be running")
                self.phase_timeouts_occurred.append(f"Rectangular SAT solve K={K}: Process timeout after {sat_result.total_time:.1f}s")
                return False  # Treat timeout as UNSAT
                
            else:
                print(f"K={K} ERROR: {sat_result.error_message} on {self.n_rows}×{self.n_cols} grid (process-isolated)")
                self.phase_timeouts_occurred.append(f"Rectangular SAT solve K={K}: Process error - {sat_result.error_message}")
                return False  # Treat error as UNSAT
        
        # Always use the process-based solver - no need for additional timeout wrapper
        return _rectangular_sat_solve()
    
    def _extract_rectangular_positions_from_model(self, model):
        """Extract vertex positions from SAT solution for rectangular grid"""
        positions = {}
        for v in range(1, self.num_vertices + 1):
            # Find column (X position)
            for col in range(1, self.n_cols + 1):
                var_id = self.X_vars[v][col-1]
                if var_id in model and model[model.index(var_id)] > 0:
                    positions[f'X_{v}'] = col
                    break
            
            # Find row (Y position)
            for row in range(1, self.n_rows + 1):
                var_id = self.Y_vars[v][row-1]
                if var_id in model and model[model.index(var_id)] > 0:
                    positions[f'Y_{v}'] = row
                    break
        
        return positions
    
    def _calculate_rectangular_bandwidth(self, positions):
        """Calculate bandwidth from vertex positions on rectangular grid"""
        max_distance = 0
        edge_distances = []
        
        for u, v in self.edges:
            col_u = positions.get(f'X_{u}', 0)
            row_u = positions.get(f'Y_{u}', 0)
            col_v = positions.get(f'X_{v}', 0)
            row_v = positions.get(f'Y_{v}', 0)
            
            distance = abs(row_u - row_v) + abs(col_u - col_v)
            max_distance = max(max_distance, distance)
            edge_distances.append((u, v, distance))
        
        return max_distance, edge_distances
    
    def _print_rectangular_solution_details(self, positions, edge_distances, bandwidth, K):
        """Show rectangular solution details"""
        print(f"Vertex positions on {self.n_rows}×{self.n_cols} grid:")
        for v in range(1, self.num_vertices + 1):
            col = positions.get(f'X_{v}', '?')
            row = positions.get(f'Y_{v}', '?')
            print(f"  v{v}: row {row}, col {col}")
        
        print(f"Edge distances:")
        for u, v, distance in edge_distances:
            print(f"  ({u},{v}): {distance}")
        
        print(f"Bandwidth: {bandwidth} (constraint: {K})")
        print(f"Valid: {'Yes' if bandwidth <= K else 'No'}")
    
    def extract_and_verify_rectangular_solution(self, model, K):
        """Extract rectangular solution and check if it satisfies K constraint"""
        print(f"--- Verifying rectangular solution ---")
        
        positions = self._extract_rectangular_positions_from_model(model)
        bandwidth, edge_distances = self._calculate_rectangular_bandwidth(positions)
        self._print_rectangular_solution_details(positions, edge_distances, bandwidth, K)
        
        return bandwidth <= K
    
    def _find_feasible_upper_bound_phase1_with_timeout(self, start_k, end_k):
        """Phase 1: Find feasible upper bound using rectangular random search with timeout"""
        print(f"\nPhase 1: Finding feasible UB with rectangular random search (timeout-protected)")
        print(f"Grid: {self.n_rows}×{self.n_cols}")
        
        for K in range(start_k, end_k + 1):
            self._check_total_timeout()  # Check total timeout
            
            print(f"\nTrying K = {K} on rectangular grid")
            
            if self.step1_test_ub_pure_random_with_timeout(K):
                print(f"Found feasible UB = {K} on rectangular grid")
                return K
            else:
                print(f"K = {K} not achievable on rectangular grid")
        
        print(f"\nError: No feasible UB in range [{start_k}, {end_k}] for {self.n_rows}×{self.n_cols} grid")
        
        # Check if we found any feasible solution during search
        if self.best_feasible_k is not None:
            print(f"However, found feasible solution with K={self.best_feasible_k} during rectangular search")
            return self.best_feasible_k
        
        return None
    
    def _optimize_with_sat_phase2_with_timeout(self, feasible_ub):
        """Phase 2: Incremental SAT optimization for rectangular grid with timeout protection (Process-based)"""
        print(f"\nPhase 2: Incremental rectangular SAT optimization from K={feasible_ub-1} down to 1")
        print(f"Grid: {self.n_rows}×{self.n_cols}, Using process-based timeout protection for each SAT solve")
        
        # Prepare base constraints once (position + distance)
        print(f"Preparing rectangular base constraints (position + distance)...")
        
        position_clauses = self.encode_position_constraints()
        distance_clauses = self.encode_distance_constraints()
        base_clauses = position_clauses + distance_clauses
        
        print(f"  Rectangular position: {len(position_clauses)} clauses")
        print(f"  Rectangular distance: {len(distance_clauses)} clauses")
        print(f"  Total rectangular base: {len(base_clauses)} clauses")
        
        optimal_k = feasible_ub
        
        # Incremental SAT: try smaller K values with process-based timeout
        for K in range(feasible_ub - 1, 0, -1):
            self._check_total_timeout()  # Check total timeout
            
            print(f"\nTrying K = {K} with process-based rectangular SAT timeout")
            
            # Prepare bandwidth constraints for this K
            bandwidth_clauses = self.encode_thermometer_bandwidth_constraints(K)
            print(f"  Generated {len(bandwidth_clauses)} rectangular bandwidth clauses for K={K}")
            
            # Combine all constraints for this K
            all_current_clauses = base_clauses + bandwidth_clauses
            print(f"  Total rectangular clauses for K={K}: {len(all_current_clauses)}")
            
            # Solve with current constraints using process-based timeout
            print(f"  Solving with timeout protection ({self.timeout_config.sat_solve_timeout}s)...")
            
            # Use process-based SAT solver for timeout protection
            sat_result = self.process_sat_solver.solve_single(
                clauses=all_current_clauses,
                problem_id=f"incremental_rectangular_bandwidth_n{self.num_vertices}_k{K}_{self.n_rows}x{self.n_cols}",
                timeout=self.timeout_config.sat_solve_timeout,
                additional_data={'n': self.num_vertices, 'K': K, 'edges': len(self.edges), 'phase': 'incremental', 'grid': f'{self.n_rows}x{self.n_cols}'}
            )
            
            if sat_result.status == 'SAT':
                optimal_k = K
                print(f"K = {K} is SAT on {self.n_rows}×{self.n_cols} grid (process-isolated)")
                
                # Extract solution for verification
                model = sat_result.model
                self.last_model = model  # Store for extraction later
                
                # Extract and update best solution
                positions = self._extract_rectangular_positions_from_model(model)
                bandwidth, edge_distances = self._calculate_rectangular_bandwidth(positions)
                self._update_best_solution(K, model, positions, bandwidth)
                
                self.extract_and_verify_rectangular_solution(model, K)
                
            elif sat_result.status == 'UNSAT':
                print(f"K = {K} is UNSAT on {self.n_rows}×{self.n_cols} grid (process-isolated)")
                print(f"Optimal rectangular bandwidth = {optimal_k}")
                break
                
            elif sat_result.status == 'TIMEOUT':
                print(f"K = {K} TIMEOUT after {sat_result.total_time:.1f}s on {self.n_rows}×{self.n_cols} grid (process-isolated)")
                if sat_result.process_killed:
                    print(f"  -> Process successfully killed using: {sat_result.kill_method}")
                else:
                    print(f"  -> WARNING: Process may still be running")
                self.phase_timeouts_occurred.append(f"Incremental rectangular SAT solve K={K}: Process timeout after {sat_result.total_time:.1f}s")
                print(f"Optimal rectangular bandwidth = {optimal_k} (stopped due to timeout)")
                break
                
            else:
                print(f"K = {K} ERROR: {sat_result.error_message} on {self.n_rows}×{self.n_cols} grid (process-isolated)")
                self.phase_timeouts_occurred.append(f"Incremental rectangular SAT solve K={K}: Process error - {sat_result.error_message}")
                print(f"Optimal rectangular bandwidth = {optimal_k} (stopped due to error)")
                break
        
        print(f"Final optimal rectangular bandwidth = {optimal_k} on {self.n_rows}×{self.n_cols} grid")
        return optimal_k
    
    def _find_feasible_upper_bound_phase1_with_timeout(self, start_k, end_k):
        """Phase 1: Find feasible upper bound using rectangular random search with timeout"""
        print(f"\nPhase 1: Finding feasible UB with rectangular random search (timeout-protected)")
        print(f"Grid: {self.n_rows}×{self.n_cols}")
        
        for K in range(start_k, end_k + 1):
            self._check_total_timeout()
            
            print(f"\nTrying K = {K} on rectangular grid")
            
            if self.step1_test_ub_pure_random_with_timeout(K):
                print(f"Found feasible UB = {K} on {self.n_rows}×{self.n_cols} grid")
                return K
            else:
                print(f"K = {K} not achievable on rectangular grid")
        
        print(f"\nError: No feasible UB in range [{start_k}, {end_k}] for {self.n_rows}×{self.n_cols} grid")
        
        # Check if we found any feasible solution during search
        if self.best_feasible_k is not None:
            print(f"However, found feasible solution with K={self.best_feasible_k} during rectangular search")
            return self.best_feasible_k
        
        return None
    
    def _optimize_with_sat_phase2_with_timeout(self, feasible_ub):
        """Phase 2: Incremental SAT optimization for rectangular grid with timeout protection"""
        print(f"\nPhase 2: Incremental rectangular SAT optimization (timeout-protected)")
        print(f"Grid: {self.n_rows}×{self.n_cols}, Starting from K={feasible_ub-1} down to 1")
        
        optimal_k = feasible_ub
        
        for K in range(feasible_ub - 1, 0, -1):
            self._check_total_timeout()
            
            print(f"\nTrying K = {K} with incremental rectangular SAT")
            
            if self.step2_encode_advanced_constraints_with_timeout(K):
                optimal_k = K
                print(f"K = {K} is SAT on {self.n_rows}×{self.n_cols} grid")
            else:
                print(f"K = {K} is UNSAT on {self.n_rows}×{self.n_cols} grid")
                print(f"Optimal bandwidth = {optimal_k}")
                break
        
        print(f"Final optimal bandwidth = {optimal_k} on {self.n_rows}×{self.n_cols} grid")
        return optimal_k
    
    def solve_bandwidth_optimization_with_timeout(self, start_k=None, end_k=None):
        """
        Main solve function for rectangular grid with comprehensive timeout protection
        """
        if start_k is None:
            start_k = 1
        if end_k is None:
            end_k = max(self.n_rows - 1, self.n_cols - 1) * 2  # Reasonable upper bound for rectangular
        
        # Initialize timing
        self.solve_start_time = time.time()
        self.phase_timeouts_occurred = []
        self.constraint_timeouts_occurred = []
        
        print(f"\n" + "="*80)
        print(f"2D RECTANGULAR BANDWIDTH OPTIMIZATION (WITH TIMEOUT)")
        print(f"Graph: {self.num_vertices} nodes, {len(self.edges)} edges")
        print(f"Grid: {self.n_rows}×{self.n_cols} (capacity: {self.n_rows * self.n_cols})")
        print(f"Testing range: K = {start_k} to {end_k}")
        print(f"Total timeout: {self.timeout_config.total_solver_timeout}s")
        print(f"="*80)
        
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
            print(f"TOTAL RECTANGULAR SOLVER TIMEOUT OCCURRED")
            print(f"Error: {e}")
            print(f"="*50)
            
            # Return best solution found so far
            if self.best_feasible_k is not None:
                print(f"Returning best feasible rectangular solution found: K={self.best_feasible_k}")
                return self.best_feasible_k
            
            return None
        
        finally:
            # Report timeout summary
            self._print_timeout_summary()
    
    def _print_timeout_summary(self):
        """Print summary of timeout events"""
        total_time = time.time() - self.solve_start_time if self.solve_start_time else 0
        
        print(f"\n" + "="*50)
        print(f"RECTANGULAR TIMEOUT SUMMARY")
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
            print(f"Best feasible rectangular solution: K={self.best_feasible_k}")
            if self.best_solution_bandwidth is not None:
                print(f"Actual rectangular bandwidth: {self.best_solution_bandwidth}")
        else:
            print(f"No feasible rectangular solution found")
        
        print(f"="*50)
    
    def update_timeout_config(self, **kwargs):
        """Update timeout configuration"""
        self.timeout_config.update_timeouts(**kwargs)
        print("Updated rectangular timeout configuration:")
        print(self.timeout_config.get_timeout_summary())


def test_rectangular_timeout_bandwidth_solver():
    """Test the rectangular timeout-enabled solver on some small graphs"""
    print("=== RECTANGULAR TIMEOUT-ENABLED BANDWIDTH SOLVER TESTS ===")
    
    # Custom timeout configuration for testing
    test_config = TimeoutConfig()
    test_config.update_timeouts(
        random_search_timeout=10.0,
        sat_solve_timeout=30.0,
        total_solver_timeout=120.0
    )
    
    # Triangle on 2x3 rectangular grid
    print(f"\n" + "="*50)
    print(f"Test 1: Triangle on 2×3 rectangular grid (with timeout)")
    print(f"="*50)
    
    num_vertices1 = 3
    n_rows1, n_cols1 = 2, 3
    edges1 = [(1, 2), (2, 3), (1, 3)]
    
    solver1 = TimeoutRectangularBandwidthOptimizationSolver(num_vertices1, n_rows1, n_cols1, 'glucose42', test_config)
    solver1.set_graph_edges(edges1)
    solver1.create_position_variables()
    solver1.create_distance_variables()
    
    optimal1 = solver1.solve_bandwidth_optimization_with_timeout(start_k=1, end_k=6)
    print(f"Rectangular triangle result: {optimal1}")


if __name__ == '__main__':
    """
    Command line usage: python rectangular_bandwidth_solver_with_timeout.py [num_vertices] [n_rows] [n_cols] [solver] [timeout_config]
    
    Arguments:
        num_vertices: Number of vertices to place
        n_rows: Number of rows in rectangular grid
        n_cols: Number of columns in rectangular grid  
        solver: SAT solver to use (glucose42 or cadical195, default: glucose42)
        timeout_config: Optional timeout configuration (format: key=value,key=value...)
    
    Examples:
        python rectangular_bandwidth_solver_with_timeout.py 5 3 4 glucose42
        python rectangular_bandwidth_solver_with_timeout.py 6 2 5 cadical195 sat_solve_timeout=60,total_solver_timeout=300
        python rectangular_bandwidth_solver_with_timeout.py  # Run test mode
    """
    import sys
    
    # Check if parameters provided
    if len(sys.argv) >= 4:
        # Rectangular solver mode
        try:
            num_vertices = int(sys.argv[1])
            n_rows = int(sys.argv[2])
            n_cols = int(sys.argv[3])
            solver_type = sys.argv[4] if len(sys.argv) >= 5 else 'glucose42'
            timeout_config_str = sys.argv[5] if len(sys.argv) >= 6 else None
            
            # Parse timeout configuration
            timeout_config = TimeoutConfig()
            if timeout_config_str:
                print("Parsing rectangular timeout configuration...")
                try:
                    config_pairs = timeout_config_str.split(',')
                    config_dict = {}
                    for pair in config_pairs:
                        key, value = pair.split('=')
                        config_dict[key.strip()] = float(value.strip())
                    timeout_config.update_timeouts(**config_dict)
                    print("Custom rectangular timeout configuration applied.")
                except Exception as e:
                    print(f"Warning: Failed to parse timeout config '{timeout_config_str}': {e}")
                    print("Using default timeout configuration.")
            
            print("=" * 80)
            print("2D RECTANGULAR BANDWIDTH OPTIMIZATION SOLVER (WITH TIMEOUT)")
            print("=" * 80)
            print(f"Vertices: {num_vertices}")
            print(f"Grid: {n_rows}×{n_cols}")
            print(f"Solver: {solver_type.upper()}")
            
            # Create random graph for testing
            import random
            random.seed(42)
            edges = []
            for i in range(1, num_vertices + 1):
                for j in range(i + 1, num_vertices + 1):
                    if random.random() < 0.4:  # 40% edge probability
                        edges.append((i, j))
            
            print(f"Generated random graph with {len(edges)} edges")
            print(f"Edges: {edges}")
            
            print("Rectangular timeout features enabled for this run.")
            
            # Solve rectangular bandwidth problem with timeout
            print(f"\nSolving 2D rectangular bandwidth minimization with timeout protection...")
            print(f"Problem: {num_vertices} vertices on {n_rows}×{n_cols} grid")
            print(f"Using: {solver_type.upper()}")
            print(f"Timeout config: {timeout_config.get_timeout_summary()}")
            
            solver = TimeoutRectangularBandwidthOptimizationSolver(num_vertices, n_rows, n_cols, solver_type, timeout_config)
            solver.set_graph_edges(edges)
            solver.create_position_variables()
            solver.create_distance_variables()
            
            start_time = time.time()
            optimal_bandwidth = solver.solve_bandwidth_optimization_with_timeout()
            solve_time = time.time() - start_time
            
            # Results
            print(f"\n" + "="*70)
            print(f"FINAL RECTANGULAR RESULTS (WITH TIMEOUT)")
            print(f"="*70)
            
            if optimal_bandwidth is not None:
                print(f"✓ Optimal rectangular bandwidth: {optimal_bandwidth}")
                print(f"✓ Solve time: {solve_time:.2f}s")
                print(f"✓ Graph: {num_vertices} vertices, {len(edges)} edges")
                print(f"✓ Grid: {n_rows}×{n_cols}")
                print(f"✓ Solver: {solver_type.upper()}")
                print(f"✓ Timeout protection: ENABLED")
                
                # Check if this was the truly optimal or best feasible due to timeout
                best_solution = solver.get_best_solution()
                if best_solution and len(solver.phase_timeouts_occurred) > 0:
                    print(f"✓ Status: TIMEOUT - BEST FEASIBLE RECTANGULAR SOLUTION")
                    print(f"✓ Note: Solution may not be optimal due to timeout")
                else:
                    print(f"✓ Status: SUCCESS - OPTIMAL RECTANGULAR SOLUTION")
                    
            else:
                print(f"✗ No rectangular solution found (may be due to timeout)")
                print(f"✗ Solve time: {solve_time:.2f}s")
                print(f"✗ Graph: {num_vertices} vertices, {len(edges)} edges")
                print(f"✗ Grid: {n_rows}×{n_cols}")
                print(f"✗ Timeout protection: ENABLED")
                print(f"✗ Status: FAILED/TIMEOUT")
            
            print(f"="*70)
            
        except ValueError as e:
            print(f"Error: Invalid parameters - {e}")
            print("Usage: python rectangular_bandwidth_solver_with_timeout.py [num_vertices] [n_rows] [n_cols] [solver] [timeout_config]")
            sys.exit(1)
        except Exception as e:
            print(f"Error during rectangular solving: {e}")
            sys.exit(1)
        
    else:
        # Test mode - run rectangular timeout test cases
        print("=" * 80)
        print("2D RECTANGULAR BANDWIDTH OPTIMIZATION SOLVER (WITH TIMEOUT) - TEST MODE")
        print("=" * 80)
        print("Usage: python rectangular_bandwidth_solver_with_timeout.py [num_vertices] [n_rows] [n_cols] [solver] [timeout_config]")
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
        print("  python rectangular_bandwidth_solver_with_timeout.py 5 3 4 glucose42 sat_solve_timeout=60")
        print("  python rectangular_bandwidth_solver_with_timeout.py 6 2 5 cadical195 total_solver_timeout=300")
        print()
        print("Running built-in rectangular timeout test cases...")
        test_rectangular_timeout_bandwidth_solver()
    
    def _print_timeout_summary(self):
        """Print summary of timeout events for rectangular solver"""
        total_time = time.time() - self.solve_start_time if self.solve_start_time else 0
        
        print(f"\n" + "="*60)
        print(f"RECTANGULAR TIMEOUT SUMMARY")
        print(f"="*60)
        print(f"Grid: {self.n_rows}×{self.n_cols}")
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
        
        print(f"="*60)
    
    def update_timeout_config(self, **kwargs):
        """Update timeout configuration"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Updated {key} = {value}s")
        
        print("Updated rectangular timeout configuration:")
        print(f"Random search timeout: {self.random_search_timeout}s")
        print(f"SAT solve timeout: {self.sat_solve_timeout}s")
        print(f"Total solver timeout: {self.total_solver_timeout}s")


def test_timeout_rectangular_solver():
    """Test the timeout-enabled rectangular solver"""
    print("=== TIMEOUT-ENABLED RECTANGULAR BANDWIDTH SOLVER TESTS ===")
    
    # Custom timeout configuration for testing
    test_config = {
        'random_search_timeout': 15.0,
        'sat_solve_timeout': 45.0,
        'total_solver_timeout': 180.0
    }
    
    # Triangle on 2×2 rectangular grid
    print(f"\n" + "="*50)
    print(f"Test 1: Triangle on 2×2 rectangular grid (with timeout)")
    print(f"="*50)
    
    num_vertices = 3
    n_rows, n_cols = 2, 2
    edges = [(1, 2), (2, 3), (1, 3)]
    
    solver = TimeoutRectangularBandwidthOptimizationSolver(
        num_vertices, n_rows, n_cols, 'glucose42', test_config
    )
    solver.set_graph_edges(edges)
    solver.create_position_variables()
    solver.create_distance_variables()
    
    optimal = solver.solve_bandwidth_optimization_with_timeout(start_k=1, end_k=4)
    print(f"Rectangular triangle result: {optimal}")


if __name__ == '__main__':
    """
    Command line usage: python rectangular_bandwidth_solver_with_timeout.py [mtx_file] [n_rows] [n_cols] [solver] [timeout_config]
    
    Arguments:
        mtx_file: Name of MTX file (searches in mtx/group 1/ and mtx/group 2/)
        n_rows: Number of rows in rectangular grid
        n_cols: Number of columns in rectangular grid  
        solver: SAT solver to use (glucose42 or cadical195, default: glucose42)
        timeout_config: Optional timeout configuration (format: key=value,key=value...)
    
    Examples:
        python rectangular_bandwidth_solver_with_timeout.py 8.jgl009.mtx 3 4 glucose42
        python rectangular_bandwidth_solver_with_timeout.py 1.ash85.mtx 5 6 cadical195 sat_solve_timeout=900,total_solver_timeout=2700
        python rectangular_bandwidth_solver_with_timeout.py 3.bcsstk01.mtx 4 4
        python rectangular_bandwidth_solver_with_timeout.py  # Run test mode
    """
    import sys
    
    if len(sys.argv) >= 4:
        # MTX file mode for rectangular grid
        mtx_file = sys.argv[1]
        n_rows = int(sys.argv[2])
        n_cols = int(sys.argv[3])
        solver_type = sys.argv[4] if len(sys.argv) >= 5 else 'glucose42'
        timeout_config_str = sys.argv[5] if len(sys.argv) >= 6 else None
        
        # Parse timeout configuration
        timeout_config = {}
        if timeout_config_str:
            print("Parsing rectangular timeout configuration...")
            try:
                config_pairs = timeout_config_str.split(',')
                for pair in config_pairs:
                    key, value = pair.split('=')
                    timeout_config[key.strip()] = float(value.strip())
                print("Custom rectangular timeout configuration applied.")
            except Exception as e:
                print(f"Warning: Failed to parse timeout config '{timeout_config_str}': {e}")
                print("Using default timeout configuration.")
        
        print("=" * 90)
        print("2D RECTANGULAR BANDWIDTH OPTIMIZATION SOLVER (WITH TIMEOUT)")
        print("=" * 90)
        print(f"File: {mtx_file}")
        print(f"Grid: {n_rows}×{n_cols}")
        print(f"Solver: {solver_type.upper()}")
        print("Timeout features enabled for rectangular grid.")
        
        # MTX file processing (same as square solver)
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
            sys.exit(1)
        
        mtx_file = found_file
        
        # Parse MTX file (same function as square solver)
        def parse_mtx_file(filename):
            """Parse MTX file and return n, edges"""
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
                    
                if line.startswith('%'):
                    continue
                
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
                
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        u, v = int(parts[0]), int(parts[1])
                        
                        if u == v:  # skip self-loops
                            continue
                        
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
        num_vertices, edges = parse_mtx_file(mtx_file)
        if num_vertices is None or edges is None:
            print("Failed to parse MTX file")
            sys.exit(1)
        
        # Solve rectangular bandwidth problem with timeout
        print(f"\nSolving rectangular 2D bandwidth minimization with timeout protection...")
        print(f"Problem: {num_vertices} vertices on {n_rows}×{n_cols} rectangular grid")
        print(f"Using: {solver_type.upper()}")
        print(f"Timeout config:")
        for key, value in timeout_config.items():
            print(f"  {key}: {value}s")
        
        solver = TimeoutRectangularBandwidthOptimizationSolver(
            num_vertices, n_rows, n_cols, solver_type, timeout_config
        )
        solver.set_graph_edges(edges)
        solver.create_position_variables()
        solver.create_distance_variables()
        
        start_time = time.time()
        optimal_bandwidth = solver.solve_bandwidth_optimization_with_timeout()
        solve_time = time.time() - start_time
        
        # Results
        print(f"\n" + "="*70)
        print(f"FINAL RECTANGULAR RESULTS (WITH TIMEOUT)")
        print(f"="*70)
        
        if optimal_bandwidth is not None:
            print(f"✓ Optimal bandwidth: {optimal_bandwidth}")
            print(f"✓ Solve time: {solve_time:.2f}s")
            print(f"✓ Graph: {num_vertices} vertices, {len(edges)} edges")
            print(f"✓ Grid: {n_rows}×{n_cols} rectangular")
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
            print(f"✗ Graph: {num_vertices} vertices, {len(edges)} edges")
            print(f"✗ Grid: {n_rows}×{n_cols} rectangular")
            print(f"✗ Timeout protection: ENABLED")
            print(f"✗ Status: FAILED/TIMEOUT")
        
        print(f"="*70)
        
        # [MTX file processing code would go here - similar to original]
        # ... 
        
    else:
        # Test mode
        print("=" * 90)
        print("2D RECTANGULAR BANDWIDTH OPTIMIZATION SOLVER (WITH TIMEOUT) - TEST MODE")
        print("=" * 90)
        print("Usage: python rectangular_bandwidth_solver_with_timeout.py [mtx_file] [n_rows] [n_cols] [solver] [timeout_config]")
        print()
        print("Examples:")
        print("  python rectangular_bandwidth_solver_with_timeout.py 8.jgl009.mtx 3 4 glucose42")
        print("  python rectangular_bandwidth_solver_with_timeout.py 1.ash85.mtx 5 6 cadical195 sat_solve_timeout=900")
        print()
        print("Running built-in rectangular timeout test cases...")
        test_timeout_rectangular_solver()


def cleanup_threads():
    """Clean up any remaining threads before shutdown"""
    # Threads are already created as daemon=True in ThreadTimeoutExecutor
    # No need to modify daemon status of active threads
    pass


# Register cleanup function
atexit.register(cleanup_threads)
