#!/usr/bin/env python3
# custom_k_bandwidth_solver.py
# 2D Bandwidth Solver for Custom K Value
# Usage: python custom_k_bandwidth_solver.py <mtx_file> <solver> <K>

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


class CustomKBandwidthSolver:
    """
    Custom K Bandwidth Solver for 2D bandwidth minimization
    
    Tests if a specific K value is achievable for a given graph.
    Uses fresh SAT solver instance to test bandwidth <= K constraint.
    
    Features:
    - Single K value testing (no optimization loop)
    - Fresh solver per test (complete isolation)
    - SAT/UNSAT result with solution verification
    - Detailed constraint statistics
    """
    
    def __init__(self, filename: str):
        """Initialize with MTX file"""
        self.filename = filename
        self.n = 0
        self.edges = []
        
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
    
    def create_variables_for_k(self, K: int):
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
    
    def encode_bandwidth_constraints(self, Tx_vars, Ty_vars, K: int):
        """
        Encode bandwidth <= K constraints
        
        For each edge: (Tx<=K) ∧ (Ty<=K) ∧ (Tx>=i → Ty<=K-i)
        """
        clauses = []
        
        for edge_id in Tx_vars:
            Tx = Tx_vars[edge_id]  # Tx[i] means Tx >= i+1
            Ty = Ty_vars[edge_id]  # Ty[i] means Ty >= i+1
            
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
        
        return clauses
    
    def test_bandwidth_k(self, K: int, solver_type: str = 'glucose42') -> Tuple[bool, Dict]:
        """
        Test if bandwidth K is achievable using SAT solver
        
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
        print(f"Solver: {solver_type.upper()}")
        print(f"Checking feasibility of bandwidth <= {K}")

        # Create variables for this K
        vpool, X_vars, Y_vars, Tx_vars, Ty_vars = self.create_variables_for_k(K)

        # Build constraints for bandwidth <= K
        print(f"Preparing SAT constraints for K = {K}...")

        # Position constraints: each vertex gets exactly one position
        position_clauses = self.encode_position_constraints(X_vars, Y_vars, vpool)
        print(f"  Position constraints: {len(position_clauses)} clauses")

        # Distance constraints: encode edge distances  
        distance_clauses = self.encode_distance_constraints(X_vars, Y_vars, Tx_vars, Ty_vars, vpool)
        print(f"  Distance constraints: {len(distance_clauses)} clauses")

        # Bandwidth constraints: ensure all edges have distance <= K
        bandwidth_clauses = self.encode_bandwidth_constraints(Tx_vars, Ty_vars, K)
        print(f"  Bandwidth constraints: {len(bandwidth_clauses)} clauses")

        # Combine all constraints
        all_clauses = position_clauses + distance_clauses + bandwidth_clauses
        total_clauses = len(all_clauses)
        total_variables = vpool.top

        print(f"Total: {total_clauses} clauses, {total_variables} variables")

        # Create SAT solver and add clauses
        print(f"Starting {solver_type.upper()} solver...")
        sat_solver = self._create_solver(solver_type)

        for clause in all_clauses:
            sat_solver.add_clause(clause)

        # Solve
        start_time = time.time()
        is_sat = sat_solver.solve()
        solve_time = time.time() - start_time

        # Get model if SAT
        model = None
        solution_info = None
        if is_sat:
            model = sat_solver.get_model()
            # Extract and verify solution
            if model:
                solution_info = self.extract_and_verify_solution(model, K, X_vars, Y_vars)

        # Clean up
        sat_solver.delete()

        # Build result info
        result_info = {
            'K': K,
            'is_sat': is_sat,
            'solve_time': solve_time,
            'solver_type': solver_type,
            'graph_size': self.n,
            'num_edges': len(self.edges),
            'total_clauses': total_clauses,
            'total_variables': total_variables,
            'model': model,
            'solution_info': solution_info
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
    
    def extract_and_verify_solution(self, model, K: int, X_vars, Y_vars) -> Optional[Dict]:
        """
        Extract vertex positions from SAT model and verify the solution
        """
        if not model:
            return None
        
        # Create set of positive literals for O(1) lookup
        posset = {lit for lit in model if lit > 0}
        
        positions = {}
        violations = []
        
        # Extract positions for each vertex
        for v in range(1, self.n + 1):
            Xrow = X_vars[v]  # list of var-ids for X_v=1..n
            Yrow = Y_vars[v]  # list of var-ids for Y_v=1..n
            
            # Find all X positions set to True
            xs = [i for i, var in enumerate(Xrow, start=1) if var in posset]
            # Find all Y positions set to True  
            ys = [i for i, var in enumerate(Yrow, start=1) if var in posset]
            
            # Check exactly-one constraint
            if len(xs) != 1 or len(ys) != 1:
                violations.append((v, xs, ys))
            else:
                positions[v] = (xs[0], ys[0])
        
        # Handle constraint violations
        if violations:
            print("\nModel decode error: some vertices do not have exactly one position.")
            for v, xs, ys in violations[:10]:  # Show first 10 violations
                print(f"  v{v}: X_true={xs}, Y_true={ys}")
            return {
                'positions': positions,
                'actual_bandwidth': None,
                'constraint_K': K,
                'is_valid': False,
                'edge_distances': [],
                'reason': 'positions_not_exactly_one'
            }
        
        # Calculate actual bandwidth
        max_distance = 0
        edge_distances = []
        
        for u, v in self.edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v] 
            distance = abs(x1 - x2) + abs(y1 - y2)
            edge_distances.append((u, v, distance))
            if distance > max_distance:
                max_distance = distance
        
        is_valid = (max_distance <= K)
        
        # Show solution summary
        print(f"\nSolution verification:")
        print(f"  Extracted positions: {len(positions)} vertices")
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


def solve_custom_k(mtx_file: str, solver_type: str, K: int) -> Dict:
    """
    Test if bandwidth K is achievable for given MTX file
    
    Args:
        mtx_file: Path to MTX file
        solver_type: SAT solver ('glucose42' or 'cadical195')
        K: Target bandwidth to test
        
    Returns:
        Dictionary with test results
    """
    print(f"CUSTOM K BANDWIDTH SOLVER")
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
    
    # Run solver
    solver = CustomKBandwidthSolver(mtx_file)
    is_sat, result_info = solver.test_bandwidth_k(K, solver_type)
    
    return {
        'status': 'success',
        'filename': mtx_file,
        'K': K,
        'is_sat': is_sat,
        'result_info': result_info
    }


if __name__ == "__main__":
    """
    Command line usage: python custom_k_bandwidth_solver.py <mtx_file> <solver> <K>
    
    Arguments:
        mtx_file: Name of MTX file (searches in mtx/group 1/, mtx/group 2/, etc.)
        solver: SAT solver to use (glucose42 or cadical195)
        K: Target bandwidth to test (required)
    
    Examples:
        python custom_k_bandwidth_solver.py bcsstk01.mtx cadical195 4
        python custom_k_bandwidth_solver.py jgl009.mtx glucose42 10
        python custom_k_bandwidth_solver.py ash85.mtx cadical195 25
        python custom_k_bandwidth_solver.py ck104.mtx glucose42 15
    
    Output:
        SAT: K is achievable (bandwidth <= K is possible)
        UNSAT: K is not achievable (bandwidth <= K is impossible)
    """
    
    # Check arguments
    if len(sys.argv) < 4:
        print("=" * 80)
        print("CUSTOM K BANDWIDTH SOLVER")
        print("=" * 80)
        print("Usage: python custom_k_bandwidth_solver.py <mtx_file> <solver> <K>")
        print()
        print("Arguments:")
        print("  mtx_file: Name of MTX file")
        print("  solver:   SAT solver (glucose42 or cadical195)")
        print("  K:        Target bandwidth to test")
        print()
        print("Examples:")
        print("  python custom_k_bandwidth_solver.py bcsstk01.mtx cadical195 4")
        print("  python custom_k_bandwidth_solver.py jgl009.mtx glucose42 10")
        print("  python custom_k_bandwidth_solver.py ash85.mtx cadical195 25")
        print("  python custom_k_bandwidth_solver.py ck104.mtx glucose42 15")
        print()
        print("Features:")
        print("  - Single K testing: tests only the specified K value")
        print("  - Fresh solver: complete independence, no persistent state")
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
    
    # Run solver
    print("=" * 80)
    print("CUSTOM K BANDWIDTH SOLVER")
    print("=" * 80)
    print(f"File: {mtx_file}")
    print(f"Solver: {solver_type.upper()}")
    print(f"Target K: {K}")
    
    try:
        results = solve_custom_k(mtx_file, solver_type, K)
        
        # Print final result
        print(f"\n" + "="*60)
        print(f"FINAL RESULT")
        print(f"="*60)
        
        status = results.get('status', 'unknown')
        if status == 'success':
            is_sat = results.get('is_sat', False)
            result_info = results.get('result_info', {})
            solve_time = result_info.get('solve_time', 0)
            
            if is_sat:
                print(f"✓ RESULT: SAT")
                print(f"✓ K = {K} is ACHIEVABLE")
                print(f"✓ The graph CAN be placed with bandwidth <= {K}")
                print(f"✓ Solve time: {solve_time:.3f}s")
                
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
