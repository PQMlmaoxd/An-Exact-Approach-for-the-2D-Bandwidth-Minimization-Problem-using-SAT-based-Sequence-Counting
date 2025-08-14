# mtx_bandwidth_validator_simple.py
# Simple validator for a target bandwidth K
# Usage: python mtx_bandwidth_validator_simple.py <mtx_file> <solver> <K>

import os
import sys
import time
from typing import Dict, List, Tuple, Optional

# Import our solver
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from incremental_bandwidth_solver import IncrementalBandwidthSolver

class SingleKBandwidthSolver:
    """
    Single K bandwidth solver for validation (non-incremental)
    
    Builds all constraints for a specific K value and solves once.
    Used for validation, not optimization.
    """
    
    def __init__(self, n, solver_type='glucose42'):
        self.n = n
        self.solver_type = solver_type
        # Use IncrementalBandwidthSolver but don't use incremental features
        self.base_solver = IncrementalBandwidthSolver(n, solver_type)
        
        # Expose the same interface
        self.vpool = self.base_solver.vpool
        self.X_vars = self.base_solver.X_vars
        self.Y_vars = self.base_solver.Y_vars
        self.edges = []
    
    def set_graph_edges(self, edges):
        """Set graph edges"""
        self.edges = edges
        self.base_solver.set_graph_edges(edges)
    
    def create_position_variables(self):
        """Create position variables"""
        self.base_solver.create_position_variables()
        self.X_vars = self.base_solver.X_vars
        self.Y_vars = self.base_solver.Y_vars
    
    def create_distance_variables(self):
        """Create distance variables"""
        self.base_solver.create_distance_variables()
    
    def encode_position_constraints(self):
        """Encode position constraints"""
        return self.base_solver.encode_position_constraints()
    
    def encode_distance_constraints(self):
        """Encode distance constraints"""
        return self.base_solver.encode_distance_constraints()
    
    def encode_thermometer_bandwidth_constraints(self, K):
        """
        Encode bandwidth <= K constraints (single K, non-incremental)
        
        For each edge: ensure Manhattan distance <= K
        """
        clauses = []
        
        for edge_id in self.base_solver.Tx_vars:
            Tx = self.base_solver.Tx_vars[edge_id]  # Tx[i] means Tx >= i+1
            Ty = self.base_solver.Ty_vars[edge_id]  # Ty[i] means Ty >= i+1
            
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
    
    def _create_solver(self):
        """Create SAT solver instance"""
        return self.base_solver._create_solver()


class CustomKBandwidthValidator:
    """
    Custom K bandwidth validator for 2D bandwidth minimization
    
    Tests if a specific K value is achievable for a given graph.
    
    Usage:
        python mtx_bandwidth_validator_simple.py <mtx_file> <solver> <K>
        
    Example:
        python mtx_bandwidth_validator_simple.py 3.bcsstk01.mtx cadical195 4
        
    This will test if bandwidth K=4 is achievable for the graph in 3.bcsstk01.mtx
    using the Cadical195 SAT solver.
    
    Output:
        - SAT: K is achievable (solution exists)
        - UNSAT: K is not achievable (no solution with bandwidth <= K)
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
            return

        header_found = False
        edges_set = set()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            if not line:
                continue

            # Handle comments
            if line.startswith('%'):
                # Skip metadata
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
                    # Ignore weights (parts[2]) - dataset is unweighted

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

        # Create solver
        solver = SingleKBandwidthSolver(self.n, solver_type)
        solver.set_graph_edges(self.edges)
        solver.create_position_variables()
        solver.create_distance_variables()

        # Build constraints for bandwidth <= K
        print(f"Preparing SAT constraints for K = {K}...")

        # Position constraints: each vertex gets exactly one position
        position_clauses = solver.encode_position_constraints()
        print(f"  Position constraints: {len(position_clauses)} clauses")

        # Distance constraints: encode edge distances  
        distance_clauses = solver.encode_distance_constraints()
        print(f"  Distance constraints: {len(distance_clauses)} clauses")

        # Bandwidth constraints: ensure all edges have distance <= K
        bandwidth_clauses = solver.encode_thermometer_bandwidth_constraints(K)
        print(f"  Bandwidth constraints: {len(bandwidth_clauses)} clauses")

        # Combine all constraints
        all_clauses = position_clauses + distance_clauses + bandwidth_clauses
        total_clauses = len(all_clauses)
        total_variables = solver.vpool.top

        print(f"Total: {total_clauses} clauses, {total_variables} variables")

        # Create SAT solver and add clauses
        print(f"Starting {solver_type.upper()} solver")
        sat_solver = solver._create_solver()

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
            # Extract solution using SAME solver (same variable IDs)
            if model:
                solution_info = self.extract_and_verify_solution(model, K, solver)

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
            'solution_info': solution_info  # Add solution info here
        }

        # Print results
        print(f"Solve time: {solve_time:.3f} seconds")

        if is_sat:
            print("RESULT: SAT")
            print(f"K = {K} is achievable")
            print(f"The graph can be placed on a {self.n}×{self.n} grid with bandwidth <= {K}")
            if model:
                print(f"Solution model extracted ({len(model)} literals)")
        else:
            print("RESULT: UNSAT")
            print(f"K = {K} is not achievable")
            print(f"The graph cannot be placed on a {self.n}×{self.n} grid with bandwidth <= {K}")

        return is_sat, result_info
    
    def extract_and_verify_solution(self, model, K: int, solver) -> Optional[Dict]:
        """
        Extract vertex positions from SAT model and verify the solution
        
        Optimized version:
        - Uses set for O(1) lookup instead of O(m) list operations
        - Validates exactly-one constraint for each vertex position
        - Detects invalid mappings early to prevent false positive results
        """
        if not model:
            return None
        
        # Create set of positive literals for O(1) lookup
        posset = {lit for lit in model if lit > 0}
        
        positions = {}
        violations = []  # Track vertices without exactly-one position
        
        # Extract positions for each vertex
        for v in range(1, self.n + 1):
            Xrow = solver.X_vars[v]  # list of var-ids for X_v=1..n
            Yrow = solver.Y_vars[v]  # list of var-ids for Y_v=1..n
            
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
            print("\nModel decode error: some vertices do not have exactly one X and one Y position.")
            for v, xs, ys in violations[:10]:  # Show first 10 violations
                print(f"  v{v}: X_true={xs}, Y_true={ys}")
            # Do not compute bandwidth for an invalid assignment
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
        
        # Detailed solution report
        print("\nSolution verification")
        print("-"*50)
        print(f"Extracted positions: {len(positions)} vertices")
        
        # Show vertex positions (first 20)
        max_show = min(20, len(positions))
        for v in sorted(list(positions.keys())[:max_show]):
            x, y = positions[v]
            print(f"  v{v}: ({x}, {y})")
        if len(positions) > max_show:
            print(f"  ... and {len(positions) - max_show} more vertices")
        
        # Show edge distances (all edges)
        print(f"\nEdge distances ({len(edge_distances)} edges):")
        for u, v, distance in edge_distances:
            marker = "ok" if distance <= K else "exceeds"
            print(f"  ({u},{v}): {distance} [{marker}]")
        
        print(f"\nBandwidth summary:")
        print(f"  Actual: {max_distance}")
        print(f"  Limit:  {K}")
        print(f"  Valid:  {'Yes' if is_valid else 'No'}")
        print(f"  Edges within limit: {sum(1 for _, _, d in edge_distances if d <= K)}/{len(edge_distances)}")
        
        return {
            'positions': positions,
            'actual_bandwidth': max_distance,
            'constraint_K': K,
            'is_valid': is_valid,
            'edge_distances': edge_distances
        }


def validate_custom_k(mtx_file: str, solver_type: str, K: int) -> Dict:
    """
    Test if bandwidth K is achievable for given MTX file
    
    Args:
        mtx_file: Path to MTX file
        solver_type: SAT solver ('glucose42' or 'cadical195')
        K: Target bandwidth to test
        
    Returns:
        Dictionary with test results
    """
    print(f"CUSTOM K BANDWIDTH VALIDATOR")
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
            f"mtx/{mtx_file}.mtx",
            f"mtx/group 1/{mtx_file}.mtx",
            f"mtx/group 2/{mtx_file}.mtx"
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
    
    # Run validation
    validator = CustomKBandwidthValidator(mtx_file)
    is_sat, result_info = validator.test_bandwidth_k(K, solver_type)
    
    # Get solution info directly from result_info (already extracted with correct solver)
    solution_info = result_info.get('solution_info')
    
    return {
        'status': 'success',
        'filename': mtx_file,
        'K': K,
        'is_sat': is_sat,
        'result_info': result_info,
        'solution_info': solution_info
    }


if __name__ == "__main__":
    """
    Command line usage: python mtx_bandwidth_validator_simple.py <mtx_file> <solver> <K>
    
    Arguments:
        mtx_file: Name of MTX file (searches in mtx/group 1/ and mtx/group 2/)
        solver: SAT solver to use (glucose42 or cadical195, default: glucose42)
        K: Target bandwidth to test (required)
    
    Examples:
        python mtx_bandwidth_validator_simple.py 3.bcsstk01.mtx cadical195 4
        python mtx_bandwidth_validator_simple.py 8.jgl009.mtx glucose42 10
        python mtx_bandwidth_validator_simple.py 1.ash85.mtx cadical195 25
    
    Output:
        SAT: K is achievable (bandwidth <= K is possible)
        UNSAT: K is not achievable (bandwidth <= K is impossible)
    """
    
    # Check arguments
    if len(sys.argv) < 4:
        print("=" * 80)
        print("CUSTOM K BANDWIDTH VALIDATOR")
        print("=" * 80)
        print("Usage: python mtx_bandwidth_validator_simple.py <mtx_file> <solver> <K>")
        print()
        print("Arguments:")
        print("  mtx_file: Name of MTX file")
        print("  solver:   SAT solver (glucose42 or cadical195)")
        print("  K:        Target bandwidth to test")
        print()
        print("Examples:")
        print("  python mtx_bandwidth_validator_simple.py 3.bcsstk01.mtx cadical195 4")
        print("  python mtx_bandwidth_validator_simple.py 8.jgl009.mtx glucose42 10")
        print("  python mtx_bandwidth_validator_simple.py 1.ash85.mtx cadical195 25")
        print()
        print("Available MTX files:")
        print("  Group 1: 1.bcspwr01.mtx, 2.bcspwr02.mtx, 3.bcsstk01.mtx, 4.can___24.mtx,")
        print("           5.fidap005.mtx, 6.fidapm05.mtx, 7.ibm32.mtx, 8.jgl009.mtx,")
        print("           9.jgl011.mtx, 10.lap_25.mtx, 11.pores_1.mtx, 12.rgg010.mtx")
        print("  Group 2: 1.ash85.mtx")
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
    
    # Run validation
    print("=" * 80)
    print("CUSTOM K BANDWIDTH VALIDATOR")
    print("=" * 80)
    print(f"File: {mtx_file}")
    print(f"Solver: {solver_type.upper()}")
    print(f"Target K: {K}")
    
    try:
        results = validate_custom_k(mtx_file, solver_type, K)
        
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
                solution_info = results.get('solution_info')
                if solution_info and solution_info['is_valid']:
                    actual_bw = solution_info['actual_bandwidth']
                    print(f"✓ Actual bandwidth in solution: {actual_bw}")
                    print(f"✓ Solution verified: VALID")
            else:
                print(f"✗ RESULT: UNSAT")
                print(f"✗ K = {K} is NOT ACHIEVABLE") 
                print(f"✗ The graph CANNOT be placed with bandwidth <= {K}")
                print(f"✗ Solve time: {solve_time:.3f}s")
                print(f"✗ K = {K} is not the bandwidth of this graph")
                
        elif status == 'file_not_found':
            print(f"✗ ERROR: File not found")
            print(f"✗ Searched for: {mtx_file}")
        else:
            print(f"✗ ERROR: {status}")
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        sys.exit(1)
