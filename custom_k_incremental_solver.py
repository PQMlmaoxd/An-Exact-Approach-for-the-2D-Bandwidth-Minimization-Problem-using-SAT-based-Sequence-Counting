#!/usr/bin/env python3
# custom_k_incremental_solver.py
# 2D Bandwidth Solver for Custom K Value using Incremental SAT
# Usage: python custom_k_incremental_solver.py <mtx_file> <solver> <K>

import os
import sys
import time
import math
from typing import Dict, List, Tuple, Optional

# Import required modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from pysat.formula import IDPool
    from pysat.solvers import Glucose42, Cadical195
    from distance_encoder import encode_abs_distance_final
    from position_constraints import encode_all_position_constraints, create_position_variables
    print("All modules loaded successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)


class CustomKIncrementalSolver:
    """
    Custom K Bandwidth Solver using Incremental SAT for 2D bandwidth minimization
    
    Tests if a specific K value is achievable for a given graph.
    Uses incremental SAT approach with persistent solver state.
    
    Features:
    - Single K value testing (no optimization loop)
    - Incremental SAT with persistent solver
    - Base constraints added once, K constraints added incrementally
    - SAT/UNSAT result with solution verification
    - Leverages learnt clauses for efficiency
    """
    
    def __init__(self, filename: str):
        """Initialize with MTX file"""
        self.filename = filename
        self.n = 0
        self.edges = []
        self.vpool = IDPool()
        
        # Position variables for X,Y coordinates
        self.X_vars = {}
        self.Y_vars = {}
        
        # Distance variables  
        self.Tx_vars = {}
        self.Ty_vars = {}
        
        # Incremental SAT state
        self.persistent_solver = None
        self.base_constraints_added = False
        
        self._parse_mtx_file()
    
    def _parse_mtx_file(self) -> None:
        """
        Parse MTX file and extract graph data
        
        Handles MatrixMarket format:
        - Comments and metadata parsing
        - Self-loop removal  
        - Undirected graph processing only
        - Error handling for malformed files
        """
        print(f"Reading MTX file: {os.path.basename(self.filename)}")

        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"File not found: {self.filename}")
            sys.exit(1)

        header_found = False
        edges_set = set()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            if not line:
                continue

            # Handle comments
            if line.startswith('%'):
                continue

            # Parse dimensions (first non-comment line)
            if not header_found:
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        rows, cols, nnz = map(int, parts[:3])
                        self.n = max(rows, cols)
                        print(f"Matrix: {rows}×{cols}, {nnz} entries")
                        print(f"Graph: undirected, unweighted")
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
                    # Ignore weights - dataset is unweighted

                    if u == v:  # skip self-loops
                        continue

                    # Convert to undirected edge (sorted tuple)
                    edge = tuple(sorted([u, v]))

                    if edge not in edges_set:
                        edges_set.add(edge)
                        self.edges.append(edge)

            except (ValueError, IndexError):
                print(f"Warning: bad edge at line {line_num}: {line}")
                continue

        print(f"Loaded {self.n} vertices and {len(self.edges)} edges")
    
    def _create_solver(self, solver_type: str):
        """Create SAT solver instance"""
        if solver_type == 'glucose42':
            return Glucose42()
        elif solver_type == 'cadical195':
            return Cadical195()
        else:
            print(f"Unknown solver '{solver_type}', using Glucose42")
            return Glucose42()
    
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
    
    def _initialize_persistent_solver(self, solver_type: str):
        """
        Initialize persistent solver with base constraints
        
        Base constraints include:
        - Position constraints (each vertex gets one position)
        - Distance constraints (Manhattan distance encoding)
        
        These constraints are K-independent and added once.
        """
        if self.persistent_solver is not None:
            print("Persistent solver already initialized")
            return
        
        print(f"\nInitializing persistent solver with base constraints...")
        print(f"Using {solver_type.upper()} with incremental interface")
        
        self.persistent_solver = self._create_solver(solver_type)
        
        # Create variables
        self.create_position_variables()
        self.create_distance_variables()
        
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
        
        total_base_clauses = len(position_clauses) + len(distance_clauses)
        print(f"  Total base constraints: {total_base_clauses} clauses")
        
        self.base_constraints_added = True
        print(f"Persistent solver initialized and ready for K-specific constraints")
    
    def encode_bandwidth_constraints_for_k(self, K: int):
        """
        Encode bandwidth <= K constraints for incremental addition
        
        For each edge: (Tx<=K) ∧ (Ty<=K) ∧ (Tx>=i → Ty<=K-i)
        """
        clauses = []
        edges_processed = 0
        
        print(f"  Encoding bandwidth constraints for K={K}...")
        
        for edge_id in self.Tx_vars:
            Tx = self.Tx_vars[edge_id]  # Tx[i] means Tx >= i+1
            Ty = self.Ty_vars[edge_id]  # Ty[i] means Ty >= i+1
            edges_processed += 1
            
            # Tx <= K (i.e., not Tx >= K+1)
            if K < len(Tx):
                clause = [-Tx[K]]
                clauses.append(clause)
            
            # Ty <= K (i.e., not Ty >= K+1)  
            if K < len(Ty):
                clause = [-Ty[K]]
                clauses.append(clause)
            
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
                        clauses.append(clause)
        
        print(f"  Generated {len(clauses)} bandwidth clauses for {edges_processed} edges, K={K}")
        return clauses
    
    def test_bandwidth_k(self, K: int, solver_type: str = 'glucose42') -> Tuple[bool, Dict]:
        """
        Test if bandwidth K is achievable using incremental SAT solver
        
        Args:
            K: Target bandwidth to test
            solver_type: SAT solver to use ('glucose42' or 'cadical195')
            
        Returns:
            (is_sat, info_dict) where:
            - is_sat: True if K is achievable (SAT), False if not (UNSAT)
            - info_dict: Detailed results including solve time, model, etc.
        """
        print("\n" + "="*60)
        print(f"Testing bandwidth K = {K}")
        print("="*60)
        print(f"Graph: {self.n} vertices, {len(self.edges)} edges")
        print(f"Solver: {solver_type.upper()} (Incremental)")
        print(f"Checking feasibility of bandwidth <= {K}")

        # Initialize persistent solver with base constraints
        self._initialize_persistent_solver(solver_type)

        # Add bandwidth constraints for this specific K
        print(f"Adding bandwidth constraints for K = {K}...")
        bandwidth_clauses = self.encode_bandwidth_constraints_for_k(K)
        print(f"  Bandwidth constraints: {len(bandwidth_clauses)} clauses")

        # Add K-specific constraints to persistent solver
        for clause in bandwidth_clauses:
            self.persistent_solver.add_clause(clause)

        total_variables = self.vpool.top
        print(f"Total variables: {total_variables}")

        # Solve with incremental solver
        print(f"Starting {solver_type.upper()} incremental solver...")
        start_time = time.time()
        is_sat = self.persistent_solver.solve()
        solve_time = time.time() - start_time

        # Get model if SAT
        model = None
        solution_info = None
        if is_sat:
            model = self.persistent_solver.get_model()
            # Extract and verify solution
            if model:
                solution_info = self.extract_and_verify_solution(model, K)

        # Build result info
        result_info = {
            'K': K,
            'is_sat': is_sat,
            'solve_time': solve_time,
            'solver_type': solver_type,
            'graph_size': self.n,
            'num_edges': len(self.edges),
            'total_variables': total_variables,
            'bandwidth_clauses': len(bandwidth_clauses),
            'model': model,
            'solution_info': solution_info,
            'approach': 'incremental'
        }

        # Print results
        print(f"Solve time: {solve_time:.3f} seconds")

        if is_sat:
            print("RESULT: SAT")
            print(f"K = {K} is ACHIEVABLE")
            print(f"The graph CAN be placed on a {self.n}×{self.n} grid with bandwidth <= {K}")
            if solution_info and solution_info.get('is_valid'):
                actual_bw = solution_info.get('actual_bandwidth', -1)
                print(f"Actual bandwidth in solution: {actual_bw}")
        else:
            print("RESULT: UNSAT")
            print(f"K = {K} is NOT ACHIEVABLE")
            print(f"The graph CANNOT be placed on a {self.n}×{self.n} grid with bandwidth <= {K}")

        return is_sat, result_info
    
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
    
    def extract_and_verify_solution(self, model, K: int) -> Optional[Dict]:
        """
        Extract vertex positions from SAT model and verify the solution
        """
        if not model:
            return None
        
        positions = self._extract_positions_from_model(model)
        violations = []
        
        # Check if all vertices have exactly one position
        for v in range(1, self.n + 1):
            if f'X_{v}' not in positions or f'Y_{v}' not in positions:
                violations.append(v)
        
        # Handle constraint violations
        if violations:
            print("\nModel decode error: some vertices do not have positions.")
            for v in violations[:10]:  # Show first 10 violations
                print(f"  v{v}: missing position")
            return {
                'positions': positions,
                'actual_bandwidth': None,
                'constraint_K': K,
                'is_valid': False,
                'edge_distances': [],
                'reason': 'missing_positions'
            }
        
        # Calculate actual bandwidth
        max_distance = 0
        edge_distances = []
        
        for u, v in self.edges:
            x_u = positions.get(f'X_{u}', 0)
            y_u = positions.get(f'Y_{u}', 0)
            x_v = positions.get(f'X_{v}', 0)
            y_v = positions.get(f'Y_{v}', 0)
            
            distance = abs(x_u - x_v) + abs(y_u - y_v)
            edge_distances.append((u, v, distance))
            if distance > max_distance:
                max_distance = distance
        
        is_valid = (max_distance <= K)
        
        # Show solution summary
        print(f"\nSolution verification:")
        print(f"  Extracted positions: {len([v for v in range(1, self.n + 1) if f'X_{v}' in positions])}/{self.n} vertices")
        print(f"  Actual bandwidth: {max_distance}")
        print(f"  Constraint limit: {K}")
        print(f"  Valid solution: {'Yes' if is_valid else 'No'}")
        print(f"  Edges within limit: {sum(1 for _, _, d in edge_distances if d <= K)}/{len(edge_distances)}")
        
        return {
            'positions': positions,
            'actual_bandwidth': max_distance,
            'constraint_K': K,
            'is_valid': is_valid,
            'edge_distances': edge_distances
        }
    
    def cleanup_solver(self):
        """Clean up persistent solver"""
        if self.persistent_solver is not None:
            print(f"Cleaning up incremental solver...")
            self.persistent_solver.delete()
            self.persistent_solver = None
            self.base_constraints_added = False


def solve_custom_k_incremental(mtx_file: str, solver_type: str, K: int) -> Dict:
    """
    Test if bandwidth K is achievable for given MTX file using incremental SAT
    
    Args:
        mtx_file: Path to MTX file
        solver_type: SAT solver ('glucose42' or 'cadical195')
        K: Target bandwidth to test
        
    Returns:
        Dictionary with test results
    """
    print(f"CUSTOM K BANDWIDTH SOLVER (INCREMENTAL)")
    print(f"File: {mtx_file}")
    print(f"Solver: {solver_type}")
    print(f"Target K: {K}")
    
    # Search for file in common locations
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
                print(f"Found file at: {path}")
                break
        
        if found_file is None:
            print(f"Error: File '{mtx_file}' not found")
            print("Searched in:")
            for path in search_paths:
                print(f"  - {path}")
            return {'status': 'file_not_found', 'filename': mtx_file}
        
        mtx_file = found_file
    
    # Run incremental solver
    solver = CustomKIncrementalSolver(mtx_file)
    
    try:
        is_sat, result_info = solver.test_bandwidth_k(K, solver_type)
        
        return {
            'status': 'success',
            'filename': mtx_file,
            'K': K,
            'is_sat': is_sat,
            'result_info': result_info
        }
    
    finally:
        # Always cleanup solver
        solver.cleanup_solver()


if __name__ == "__main__":
    """
    Command line usage: python custom_k_incremental_solver.py <mtx_file> <solver> <K>
    
    Arguments:
        mtx_file: Name of MTX file (searches in mtx/group 1/, mtx/group 2/, etc.)
        solver: SAT solver to use (glucose42 or cadical195)
        K: Target bandwidth to test (required)
    
    Examples:
        python custom_k_incremental_solver.py bcsstk01.mtx cadical195 4
        python custom_k_incremental_solver.py jgl009.mtx glucose42 10
        python custom_k_incremental_solver.py ash85.mtx cadical195 25
        python custom_k_incremental_solver.py ck104.mtx glucose42 15
    
    Output:
        SAT: K is achievable (bandwidth <= K is possible)
        UNSAT: K is not achievable (bandwidth <= K is impossible)
    """
    
    # Check arguments
    if len(sys.argv) < 4:
        print("=" * 80)
        print("CUSTOM K BANDWIDTH SOLVER (INCREMENTAL)")
        print("=" * 80)
        print("Usage: python custom_k_incremental_solver.py <mtx_file> <solver> <K>")
        print()
        print("Arguments:")
        print("  mtx_file: Name of MTX file")
        print("  solver:   SAT solver (glucose42 or cadical195)")
        print("  K:        Target bandwidth to test")
        print()
        print("Examples:")
        print("  python custom_k_incremental_solver.py bcsstk01.mtx cadical195 4")
        print("  python custom_k_incremental_solver.py jgl009.mtx glucose42 10")
        print("  python custom_k_incremental_solver.py ash85.mtx cadical195 25")
        print("  python custom_k_incremental_solver.py ck104.mtx glucose42 15")
        print()
        print("Features:")
        print("  - Single K testing: tests only the specified K value")
        print("  - Incremental SAT: persistent solver with base constraints")
        print("  - Learnt clause reuse: leverages SAT solver learning")
        print("  - SAT/UNSAT result: clear achievability determination")
        print("  - Solution verification: validates extracted solutions")
        print()
        print("Available MTX files:")
        print("  Group 1: bcspwr01.mtx, bcspwr02.mtx, bcsstk01.mtx, can___24.mtx,")
        print("           fidap005.mtx, fidapm05.mtx, ibm32.mtx, jgl009.mtx,")
        print("           jgl011.mtx, lap_25.mtx, pores_1.mtx, rgg010.mtx")
        print("  Group 2: ash85.mtx")
        print("  Group 3: ck104.mtx, bcsstk04.mtx, bcsstk05.mtx, etc.")
        print()
        print("Output:")
        print("  SAT:   K is achievable (solution exists)")
        print("  UNSAT: K is not achievable (no solution with bandwidth <= K)")
        print()
        print("Advantages over non-incremental:")
        print("  - Base constraints added once (position, distance)")
        print("  - Only K-specific constraints added per test")
        print("  - Learnt clauses retained for efficiency")
        print("  - Potentially faster for multiple K tests")
        sys.exit(1)
    
    # Parse arguments
    mtx_file = sys.argv[1]
    solver_type = sys.argv[2]
    
    try:
        K = int(sys.argv[3])
    except ValueError:
        print(f"Error: K must be an integer, got '{sys.argv[3]}'")
        sys.exit(1)
    
    if K <= 0:
        print(f"Error: K must be positive, got {K}")
        sys.exit(1)
    
    if solver_type not in ['glucose42', 'cadical195']:
        print(f"Error: Solver must be 'glucose42' or 'cadical195', got '{solver_type}'")
        sys.exit(1)
    
    # Run incremental solver
    print("=" * 80)
    print("CUSTOM K BANDWIDTH SOLVER (INCREMENTAL)")
    print("=" * 80)
    print(f"File: {mtx_file}")
    print(f"Solver: {solver_type.upper()}")
    print(f"Target K: {K}")
    print(f"Approach: Incremental SAT with persistent solver")
    
    try:
        results = solve_custom_k_incremental(mtx_file, solver_type, K)
        
        # Print final result
        print(f"\n" + "="*60)
        print(f"FINAL RESULT")
        print(f"="*60)
        
        status = results.get('status', 'unknown')
        if status == 'success':
            is_sat = results.get('is_sat', False)
            result_info = results.get('result_info', {})
            solve_time = result_info.get('solve_time', 0)
            approach = result_info.get('approach', 'incremental')
            
            if is_sat:
                print(f"✓ RESULT: SAT")
                print(f"✓ K = {K} is ACHIEVABLE")
                print(f"✓ The graph CAN be placed with bandwidth <= {K}")
                print(f"✓ Solve time: {solve_time:.3f}s")
                print(f"✓ Approach: {approach.upper()} SAT")
                
                # Show solution if extracted
                solution_info = result_info.get('solution_info')
                if solution_info and solution_info.get('is_valid'):
                    actual_bw = solution_info.get('actual_bandwidth', -1)
                    print(f"✓ Actual bandwidth in solution: {actual_bw}")
                    print(f"✓ Solution verified: VALID")
            else:
                print(f"✗ RESULT: UNSAT")
                print(f"✗ K = {K} is NOT ACHIEVABLE") 
                print(f"✗ The graph CANNOT be placed with bandwidth <= {K}")
                print(f"✗ Solve time: {solve_time:.3f}s")
                print(f"✗ Approach: {approach.upper()} SAT")
                
        elif status == 'file_not_found':
            print(f"✗ ERROR: File not found")
            print(f"✗ Searched for: {mtx_file}")
        else:
            print(f"✗ ERROR: {status}")
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
