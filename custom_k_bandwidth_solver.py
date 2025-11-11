#!/usr/bin/env python3
# custom_k_bandwidth_solver.py
# 2D Bandwidth Solver for Custom K Value
# Usage: python custom_k_bandwidth_solver.py <mtx_file> <solver> <K> [--method=standard|cutoff]

import os
import sys
import time
import math
import gc
import ctypes
from typing import Dict, List, Tuple, Optional

# Import required modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from pysat.formula import IDPool
    from pysat.solvers import Glucose42, Cadical195
    # Import all encoding methods
    from distance_encoder import encode_abs_distance_final
    from distance_encoder_cutoff import encode_abs_distance_cutoff, calculate_theoretical_upper_bound
    from distance_encoder_hybrid import encode_abs_distance_hybrid
    from position_constraints import encode_all_position_constraints, create_position_variables
    print("All modules loaded successfully (standard + cutoff + hybrid encoders)")
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
    - Choice between standard, cutoff, and hybrid encoding methods
    
    Encoding Methods:
    - standard: Full distance encoding (1 to n-1), higher variable count
    - cutoff: UB-optimized encoding (1 to theoretical_ub), reduced variables
    - hybrid: Incremental replacement of T variables with mutual exclusions
    """
    
    def __init__(self, filename: str, encoding_method: str = 'standard', num_replacements: int = 0):
        """
        Initialize with MTX file and encoding method
        
        Args:
            filename: Path to MTX file
            encoding_method: 'standard', 'cutoff', or 'hybrid' (default: 'standard')
            num_replacements: For hybrid method, number of T variables to replace from UB
                            - 0: No replacement (equivalent to standard)
                            - 1: Replace only T_UB
                            - 2: Replace T_UB and T_{UB+1}
                            - n-1-UB: Full replacement (equivalent to cutoff)
        """
        self.filename = filename
        self.encoding_method = encoding_method
        self.num_replacements = num_replacements
        self.n = 0
        self.edges = []
        self.theoretical_ub = None  # Will be calculated after parsing
        
        self._parse_mtx_file()
        
        # Calculate theoretical UB
        self.theoretical_ub = calculate_theoretical_upper_bound(self.n)
        
        # Validate and adjust num_replacements for hybrid method
        if self.encoding_method == 'hybrid':
            max_replacements = self.n - 1 - self.theoretical_ub
            
            # Ensure num_replacements does not exceed maximum
            if self.num_replacements > max_replacements:
                print(f"⚠ WARNING: num_replacements={self.num_replacements} exceeds maximum={max_replacements}")
                print(f"           Automatically capping to max_replacements={max_replacements}")
                self.num_replacements = max_replacements
            
            # Ensure num_replacements is non-negative
            if self.num_replacements < 0:
                print(f"⚠ WARNING: num_replacements={self.num_replacements} is negative")
                print(f"           Setting to 0 (no replacement)")
                self.num_replacements = 0
        
        # Display encoding method info
        if self.encoding_method == 'cutoff':
            print(f"Encoding method: CUTOFF (UB={self.theoretical_ub})")
            print(f"Benefits: Reduced variables, faster encoding, early infeasible elimination")
        elif self.encoding_method == 'hybrid':
            max_replacements = self.n - 1 - self.theoretical_ub
            actual_replacements = self.num_replacements  # Already validated above
            
            print(f"Encoding method: HYBRID (UB={self.theoretical_ub})")
            print(f"Replacements: {actual_replacements} levels (max: {max_replacements})")
            
            if actual_replacements > 0:
                replacement_start = self.theoretical_ub + 1
                replacement_end = min(replacement_start + actual_replacements - 1, self.n - 1)
                print(f"Keeping: T_1 to T_{self.theoretical_ub} with activation clauses")
                print(f"Replacing: T_{replacement_start} to T_{replacement_end} with mutual exclusions")
                
                # Show Stage 4 info if partial replacement
                if replacement_end < self.n - 1:
                    print(f"Adding: T_{replacement_end + 1} to T_{self.n - 1} with activation clauses (Stage 4)")
            else:
                print(f"No replacement: Equivalent to standard encoding")
            
            if actual_replacements == max_replacements:
                print(f"Full replacement: Equivalent to cutoff encoding")
        else:
            print(f"Encoding method: STANDARD")
            print(f"Using full distance range: 1 to {self.n-1}")
    
    def _parse_mtx_file(self) -> None:
        """
        Parse MTX file and extract graph data
        
        Handles MatrixMarket format:
        - Comments and metadata parsing
        - Self-loop removal  
        - Undirected graph processing only
        - Error handling for malformed files
        
        Sets:
            self.n: Number of vertices (max of rows/cols)
            self.edges: List of edges as tuples (u, v) where u < v
        """
        print(f"Reading MTX file: {os.path.basename(self.filename)}")

        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"✗ ERROR: File not found: {self.filename}")
            sys.exit(1)
        except IOError as e:
            print(f"✗ ERROR: Cannot read file: {e}")
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
                        
                        # Validation
                        if rows <= 0 or cols <= 0:
                            print(f"✗ ERROR: Invalid dimensions: {rows}×{cols}")
                            sys.exit(1)
                        
                        self.n = max(rows, cols)
                        print(f"Matrix: {rows}×{cols}, {nnz} entries")
                        print(f"Graph: undirected, unweighted")
                        print(f"Vertices: {self.n}")
                        header_found = True
                        continue
                except ValueError as e:
                    print(f"✗ ERROR: Invalid header at line {line_num}: {line}")
                    print(f"         {e}")
                    sys.exit(1)

            # Parse edges
            try:
                parts = line.split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    # Ignore weights - dataset is unweighted

                    # Validation
                    if u < 1 or u > self.n or v < 1 or v > self.n:
                        print(f"⚠ Warning: Edge ({u}, {v}) out of range [1, {self.n}] at line {line_num}")
                        continue

                    if u == v:  # skip self-loops
                        continue

                    # Convert to undirected edge (sorted tuple)
                    edge = tuple(sorted([u, v]))

                    if edge not in edges_set:
                        edges_set.add(edge)
                        self.edges.append(edge)

            except (ValueError, IndexError) as e:
                print(f"⚠ Warning: Invalid edge at line {line_num}: {line}")
                continue

        # Final validation
        if not header_found:
            print(f"✗ ERROR: No valid header found in MTX file")
            sys.exit(1)
        
        if self.n == 0:
            print(f"✗ ERROR: No vertices found")
            sys.exit(1)
        
        if len(self.edges) == 0:
            print(f"⚠ WARNING: No edges found (empty graph)")
        
        print(f"✓ Loaded {self.n} vertices and {len(self.edges)} edges")
    
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
        
        For standard method: creates T variables for full range 1..n-1
        For cutoff method: creates T variables only up to theoretical_ub
        For hybrid method: creates T variables based on num_replacements
        
        Returns: vpool, X_vars, Y_vars, Tx_vars, Ty_vars
        """
        vpool = IDPool()
        
        # Position variables for X,Y coordinates
        X_vars, Y_vars = create_position_variables(self.n, vpool)
        
        # Distance variables for each edge (structure depends on encoding method)
        Tx_vars = {}  # T variables for X distances
        Ty_vars = {}  # T variables for Y distances
        
        if self.encoding_method == 'cutoff':
            # Cutoff method: prepare for cutoff encoding (variables created during encoding)
            for i, (u, v) in enumerate(self.edges):
                edge_id = f'edge_{u}_{v}'
                # Store prefixes for cutoff encoding
                Tx_vars[edge_id] = {'prefix': f'Tx[{u},{v}]', 'vars': {}}
                Ty_vars[edge_id] = {'prefix': f'Ty[{u},{v}]', 'vars': {}}
        elif self.encoding_method == 'hybrid':
            # Hybrid method: prepare for hybrid encoding (variables created during encoding)
            for i, (u, v) in enumerate(self.edges):
                edge_id = f'edge_{u}_{v}'
                # Store prefixes and num_replacements for hybrid encoding
                Tx_vars[edge_id] = {'prefix': f'Tx[{u},{v}]', 'vars': {}, 'num_repl': self.num_replacements}
                Ty_vars[edge_id] = {'prefix': f'Ty[{u},{v}]', 'vars': {}, 'num_repl': self.num_replacements}
        else:
            # Standard method: create T variables for full range
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
    
    def encode_distance_constraints(self, X_vars, Y_vars, Tx_vars, Ty_vars, vpool, verbose: bool = False):
        """
        Encode distance constraints for each edge
        
        Uses standard, cutoff, or hybrid encoding based on initialization
        
        Args:
            verbose: If True, print detailed clause information during encoding
        """
        clauses = []
        
        if self.encoding_method == 'cutoff':
            # Cutoff encoding: uses theoretical UB
            if verbose:
                print(f"\n  🔍 CUTOFF ENCODING: Processing {len(self.edges)} edges")
            
            for edge_idx, (edge_id, (u, v)) in enumerate(zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges), 1):
                if verbose:
                    print(f"\n  Edge {edge_idx}/{len(self.edges)}: ({u}, {v})")
                
                # X distance encoding with cutoff
                tx_prefix = Tx_vars[edge_id]['prefix']
                tx_clauses, tx_vars = encode_abs_distance_cutoff(
                    X_vars[u], X_vars[v], self.theoretical_ub, vpool, tx_prefix
                )
                Tx_vars[edge_id]['vars'] = tx_vars
                
                if verbose:
                    print(f"    X-distance: {len(tx_clauses)} clauses, {len(tx_vars)} T vars")
                    if len(tx_clauses) <= 20:  # Show first few clauses
                        for i, clause in enumerate(tx_clauses[:5], 1):
                            print(f"      Clause {i}: {clause}")
                        if len(tx_clauses) > 5:
                            print(f"      ... ({len(tx_clauses) - 5} more clauses)")
                
                clauses.extend(tx_clauses)
                
                # Y distance encoding with cutoff
                ty_prefix = Ty_vars[edge_id]['prefix']
                ty_clauses, ty_vars = encode_abs_distance_cutoff(
                    Y_vars[u], Y_vars[v], self.theoretical_ub, vpool, ty_prefix
                )
                Ty_vars[edge_id]['vars'] = ty_vars
                
                if verbose:
                    print(f"    Y-distance: {len(ty_clauses)} clauses, {len(ty_vars)} T vars")
                    if len(ty_clauses) <= 20:
                        for i, clause in enumerate(ty_clauses[:5], 1):
                            print(f"      Clause {i}: {clause}")
                        if len(ty_clauses) > 5:
                            print(f"      ... ({len(ty_clauses) - 5} more clauses)")
                
                clauses.extend(ty_clauses)
        
        elif self.encoding_method == 'hybrid':
            # Hybrid encoding: uses incremental replacement
            if verbose:
                print(f"\n  🔍 HYBRID ENCODING: Processing {len(self.edges)} edges")
                print(f"     Replacements: {self.num_replacements}")
            
            for edge_idx, (edge_id, (u, v)) in enumerate(zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges), 1):
                if verbose:
                    print(f"\n  Edge {edge_idx}/{len(self.edges)}: ({u}, {v})")
                
                # X distance encoding with hybrid
                tx_prefix = Tx_vars[edge_id]['prefix']
                num_repl = Tx_vars[edge_id]['num_repl']
                tx_clauses, tx_vars = encode_abs_distance_hybrid(
                    X_vars[u], X_vars[v], self.n, self.theoretical_ub, vpool,
                    prefix=tx_prefix, num_replacements=num_repl
                )
                Tx_vars[edge_id]['vars'] = tx_vars
                
                if verbose:
                    print(f"    X-distance: {len(tx_clauses)} clauses, {len(tx_vars)} T vars")
                    if len(tx_clauses) <= 20:
                        for i, clause in enumerate(tx_clauses[:5], 1):
                            print(f"      Clause {i}: {clause}")
                        if len(tx_clauses) > 5:
                            print(f"      ... ({len(tx_clauses) - 5} more clauses)")
                
                clauses.extend(tx_clauses)
                
                # Y distance encoding with hybrid
                ty_prefix = Ty_vars[edge_id]['prefix']
                ty_clauses, ty_vars = encode_abs_distance_hybrid(
                    Y_vars[u], Y_vars[v], self.n, self.theoretical_ub, vpool,
                    prefix=ty_prefix, num_replacements=num_repl
                )
                Ty_vars[edge_id]['vars'] = ty_vars
                
                if verbose:
                    print(f"    Y-distance: {len(ty_clauses)} clauses, {len(ty_vars)} T vars")
                    if len(ty_clauses) <= 20:
                        for i, clause in enumerate(ty_clauses[:5], 1):
                            print(f"      Clause {i}: {clause}")
                        if len(ty_clauses) > 5:
                            print(f"      ... ({len(ty_clauses) - 5} more clauses)")
                
                clauses.extend(ty_clauses)
        
        else:
            # Standard encoding: uses full range 1..n-1
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
        Encode bandwidth <= K constraints with SYMMETRY (both directions)
        
        For each edge: (Tx<=K) ∧ (Ty<=K) ∧ (Tx>=i → Ty<=K-i) ∧ (Ty>=i → Tx<=K-i)
        
        Works with all encoding methods (standard, cutoff, hybrid)
        
        CRITICAL: The symmetry constraints (both directions) are REQUIRED for correctness.
        This matches the implementation in incremental_bandwidth_solver.py
        """
        clauses = []
        
        for edge_id in Tx_vars:
            if self.encoding_method in ['cutoff', 'hybrid']:
                # Cutoff/Hybrid encoding: T_vars is a dict {d: var_id}
                tx_vars = Tx_vars[edge_id]['vars']
                ty_vars = Ty_vars[edge_id]['vars']
                
                # Tx <= K (i.e., not Tx >= K+1)
                if (K + 1) in tx_vars:
                    clause = [-tx_vars[K + 1]]
                    clauses.append(clause)
                
                # Ty <= K (i.e., not Ty >= K+1)  
                if (K + 1) in ty_vars:
                    clause = [-ty_vars[K + 1]]
                    clauses.append(clause)
                
                # Implication: BOTH DIRECTIONS for symmetry
                for i in range(1, K + 1):
                    remaining = K - i
                    if remaining >= 0:
                        # Direction 1: Tx >= i → Ty <= K-i
                        # ¬(Tx >= i) ∨ ¬(Ty >= K-i+1)
                        if i in tx_vars and (remaining + 1) in ty_vars:
                            clause = [-tx_vars[i], -ty_vars[remaining + 1]]
                            clauses.append(clause)
                        
                        # Direction 2: Ty >= i → Tx <= K-i (SYMMETRY)
                        # ¬(Ty >= i) ∨ ¬(Tx >= K-i+1)
                        if i in ty_vars and (remaining + 1) in tx_vars:
                            clause = [-ty_vars[i], -tx_vars[remaining + 1]]
                            clauses.append(clause)
            else:
                # Standard encoding: T_vars is a list
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
                
                # Implication: BOTH DIRECTIONS for symmetry
                for i in range(1, K + 1):
                    remaining = K - i
                    if remaining >= 0:
                        # Direction 1: Tx >= i → Ty <= K-i
                        if i-1 < len(Tx) and remaining < len(Ty):
                            tx_geq_i = Tx[i-1]  # Tx >= i
                            ty_geq_rem_plus1 = Ty[remaining]  # Ty >= K-i+1
                            clause = [-tx_geq_i, -ty_geq_rem_plus1]
                            clauses.append(clause)
                        
                        # Direction 2: Ty >= i → Tx <= K-i (SYMMETRY)
                        if i-1 < len(Ty) and remaining < len(Tx):
                            ty_geq_i = Ty[i-1]  # Ty >= i
                            tx_geq_rem_plus1 = Tx[remaining]  # Tx >= K-i+1
                            clause = [-ty_geq_i, -tx_geq_rem_plus1]
                            clauses.append(clause)
        
        return clauses
    
    def test_bandwidth_k(self, K: int, solver_type: str = 'glucose42', dump_clauses: bool = False, verbose: bool = False) -> Tuple[bool, Dict]:
        """
        Test if bandwidth K is achievable using SAT solver
        
        Args:
            K: Target bandwidth to test
            solver_type: SAT solver to use ('glucose42' or 'cadical195')
            dump_clauses: If True, dump all clauses to file for debugging
            verbose: If True, print detailed clause information during encoding
            
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
        print(f"Encoding: {self.encoding_method.upper()}")
        if self.encoding_method == 'cutoff':
            print(f"Theoretical UB: {self.theoretical_ub}")
            if K > self.theoretical_ub:
                print(f"WARNING: K={K} > theoretical UB={self.theoretical_ub}")
                print(f"This K is trivially SAT (within theoretical bounds)")
        print(f"Checking feasibility of bandwidth <= {K}")
        
        if dump_clauses:
            print(f"⚠ Clause dumping ENABLED - will save clauses to file")
        
        if verbose:
            print(f"⚠ Verbose mode ENABLED - will show detailed encoding info")

        # Create variables for this K
        vpool, X_vars, Y_vars, Tx_vars, Ty_vars = self.create_variables_for_k(K)

        print(f"Preparing SAT constraints for K = {K}...")
        print(f"Using streaming approach to minimize peak RAM")

        # Create SAT solver
        print(f"Starting {solver_type.upper()} solver...")
        sat_solver = self._create_solver(solver_type)
        
        # Track total counts for reporting
        total_variables = vpool.top
        position_count = 0
        distance_count = 0
        bandwidth_count = 0
        
        # Optional: Collect all clauses for dumping
        all_clauses = [] if dump_clauses else None

        # Stream position constraints (add and count as we go)
        position_clauses = self.encode_position_constraints(X_vars, Y_vars, vpool)
        position_count = 0
        for clause in position_clauses:
            sat_solver.add_clause(clause)
            position_count += 1
            if dump_clauses:
                all_clauses.append(clause)
        print(f"  Position constraints: {position_count} clauses")

        # Stream distance constraints (add and count as we go)
        if verbose:
            print(f"\n📐 Encoding distance constraints...")
        
        distance_clauses = self.encode_distance_constraints(X_vars, Y_vars, Tx_vars, Ty_vars, vpool, verbose)
        distance_count = 0
        for clause in distance_clauses:
            sat_solver.add_clause(clause)
            distance_count += 1
            if dump_clauses:
                all_clauses.append(clause)
        print(f"  Distance constraints: {distance_count} clauses")

        # Stream bandwidth constraints (add and count as we go)
        bandwidth_clauses = self.encode_bandwidth_constraints(Tx_vars, Ty_vars, K)
        bandwidth_count = 0
        for clause in bandwidth_clauses:
            sat_solver.add_clause(clause)
            bandwidth_count += 1
            if dump_clauses:
                all_clauses.append(clause)
        print(f"  Bandwidth constraints: {bandwidth_count} clauses")

        # Calculate total after all constraints added
        total_clauses = position_count + distance_count + bandwidth_count
        print(f"Total: {total_clauses} clauses, {total_variables} variables")
        
        # Dump clauses to file if requested
        if dump_clauses:
            filename = os.path.basename(self.filename).replace('.mtx', '')
            dump_file = f"clauses_{filename}_K{K}_{self.encoding_method}.txt"
            print(f"\n📝 Dumping clauses to: {dump_file}")
            
            with open(dump_file, 'w') as f:
                f.write(f"# Clauses for {filename} with K={K}\n")
                f.write(f"# Encoding method: {self.encoding_method}\n")
                f.write(f"# Graph: {self.n} vertices, {len(self.edges)} edges\n")
                f.write(f"# Total clauses: {total_clauses}\n")
                f.write(f"# Total variables: {total_variables}\n")
                f.write(f"# Position: {position_count}, Distance: {distance_count}, Bandwidth: {bandwidth_count}\n")
                f.write(f"\n")
                
                # Write position clauses
                f.write(f"# === POSITION CONSTRAINTS ({position_count} clauses) ===\n")
                for i, clause in enumerate(all_clauses[:position_count], 1):
                    f.write(f"{clause}\n")
                
                # Write distance clauses
                f.write(f"\n# === DISTANCE CONSTRAINTS ({distance_count} clauses) ===\n")
                for i, clause in enumerate(all_clauses[position_count:position_count+distance_count], 1):
                    f.write(f"{clause}\n")
                
                # Write bandwidth clauses
                f.write(f"\n# === BANDWIDTH CONSTRAINTS ({bandwidth_count} clauses) ===\n")
                for i, clause in enumerate(all_clauses[position_count+distance_count:], 1):
                    f.write(f"{clause}\n")
            
            print(f"✓ Clauses dumped successfully")
            print(f"  File: {dump_file}")
            print(f"  Size: {len(all_clauses)} clauses")

        # Force garbage collection and memory trim
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

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
            'encoding_method': self.encoding_method,
            'graph_size': self.n,
            'num_edges': len(self.edges),
            'total_clauses': total_clauses,
            'total_variables': total_variables,
            'theoretical_ub': self.theoretical_ub,
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


def solve_custom_k(mtx_file: str, solver_type: str, K: int, encoding_method: str = 'standard', 
                   num_replacements: int = 0, dump_clauses: bool = False, verbose: bool = False) -> Dict:
    """
    Test if bandwidth K is achievable for given MTX file
    
    Args:
        mtx_file: Path to MTX file
        solver_type: SAT solver ('glucose42' or 'cadical195')
        K: Target bandwidth to test
        encoding_method: 'standard', 'cutoff', or 'hybrid' (default: 'standard')
        num_replacements: For hybrid method, number of T variables to replace (default: 0)
        dump_clauses: If True, dump all clauses to file for debugging (default: False)
        verbose: If True, print detailed clause information during encoding (default: False)
        
    Returns:
        Dictionary with test results
    """
    print(f"CUSTOM K BANDWIDTH SOLVER")
    print(f"File: {mtx_file}")
    print(f"Solver: {solver_type}")
    print(f"Target K: {K}")
    print(f"Encoding: {encoding_method}")
    if encoding_method == 'hybrid':
        print(f"Replacements: {num_replacements}")
    if dump_clauses:
        print(f"Dump clauses: ENABLED")
    if verbose:
        print(f"Verbose mode: ENABLED")
    
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
    solver = CustomKBandwidthSolver(mtx_file, encoding_method, num_replacements)
    is_sat, result_info = solver.test_bandwidth_k(K, solver_type, dump_clauses, verbose)
    
    return {
        'status': 'success',
        'filename': mtx_file,
        'K': K,
        'encoding_method': encoding_method,
        'num_replacements': num_replacements,
        'is_sat': is_sat,
        'result_info': result_info
    }


if __name__ == "__main__":
    """
    Command line usage: python custom_k_bandwidth_solver.py <mtx_file> <solver> <K> [--method=standard|cutoff|hybrid] [--replacements=N] [--dump-clauses]
    
    Arguments:
        mtx_file: Name of MTX file
        solver:   SAT solver (glucose42 or cadical195)
        K:        Target bandwidth to test
        --method: Encoding method (optional, default=standard)
                  - standard: Full distance encoding (1 to n-1)
                  - cutoff: UB-optimized encoding (1 to theoretical_ub)
                  - hybrid: Incremental T→mutual-exclusion replacement
        --replacements: For hybrid method, number of T variables to replace from UB (default=0)
                       - 0: No replacement (equivalent to standard)
                       - 1: Replace only T_UB
                       - 2: Replace T_UB and T_{UB+1}
                       - max: Replace T_UB to T_{n-1} (equivalent to cutoff)
        --dump-clauses: Dump all clauses to file for debugging (optional)
    
    Examples:
        # Standard encoding (full range)
        python custom_k_bandwidth_solver.py bcsstk01.mtx cadical195 4
        python custom_k_bandwidth_solver.py jgl009.mtx glucose42 10 --method=standard
        
        # Cutoff encoding (UB-optimized)
        python custom_k_bandwidth_solver.py ash85.mtx cadical195 25 --method=cutoff
        python custom_k_bandwidth_solver.py ck104.mtx glucose42 15 --method=cutoff
        
        # Hybrid encoding (incremental replacement)
        python custom_k_bandwidth_solver.py ash85.mtx cadical195 25 --method=hybrid --replacements=1
        python custom_k_bandwidth_solver.py ck104.mtx glucose42 15 --method=hybrid --replacements=5
        python custom_k_bandwidth_solver.py jgl009.mtx cadical195 10 --method=hybrid --replacements=100
        
        # Dump clauses for debugging
        python custom_k_bandwidth_solver.py bfw62a.mtx cadical195 3 --method=cutoff --dump-clauses
        python custom_k_bandwidth_solver.py bfw62a.mtx cadical195 3 --method=hybrid --replacements=50 --dump-clauses
    
    Output:
        SAT: K is achievable (bandwidth <= K is possible)
        UNSAT: K is not achievable (bandwidth <= K is impossible)
    """
    
    # Check arguments
    if len(sys.argv) < 4:
        print("=" * 80)
        print("CUSTOM K BANDWIDTH SOLVER")
        print("=" * 80)
        print("Usage: python custom_k_bandwidth_solver.py <mtx_file> <solver> <K> [--method=standard|cutoff|hybrid] [--replacements=N]")
        print()
        print("Arguments:")
        print("  mtx_file: Name of MTX file")
        print("  solver:   SAT solver (glucose42 or cadical195)")
        print("  K:        Target bandwidth to test")
        print("  --method: Encoding method (optional, default=standard)")
        print("            - standard: Full distance encoding (1 to n-1)")
        print("            - cutoff: UB-optimized encoding (1 to theoretical_ub)")
        print("            - hybrid: Incremental T→mutual-exclusion replacement")
        print("  --replacements: For hybrid method, number of T variables to replace (default=0)")
        print()
        print("Examples:")
        print("  # Standard encoding")
        print("  python custom_k_bandwidth_solver.py bcsstk01.mtx cadical195 4")
        print("  python custom_k_bandwidth_solver.py jgl009.mtx glucose42 10 --method=standard")
        print()
        print("  # Cutoff encoding (recommended for large K)")
        print("  python custom_k_bandwidth_solver.py ash85.mtx cadical195 25 --method=cutoff")
        print("  python custom_k_bandwidth_solver.py ck104.mtx glucose42 15 --method=cutoff")
        print()
        print("  # Hybrid encoding (performance comparison)")
        print("  python custom_k_bandwidth_solver.py ash85.mtx cadical195 25 --method=hybrid --replacements=1")
        print("  python custom_k_bandwidth_solver.py ck104.mtx glucose42 15 --method=hybrid --replacements=5")
        print("  python custom_k_bandwidth_solver.py jgl009.mtx cadical195 10 --method=hybrid --replacements=100")
        print()
        print("Features:")
        print("  - Single K testing: tests only the specified K value")
        print("  - Fresh solver: complete independence, no persistent state")
        print("  - SAT/UNSAT result: clear achievability determination")
        print("  - Solution verification: validates extracted solutions")
        print("  - SYMMETRY constraints: both directions (Tx→Ty and Ty→Tx)")
        print()
        print("Encoding Methods:")
        print("  STANDARD:")
        print("    - Full T variable range: 1 to n-1")
        print("    - Higher variable count, more clauses")
        print("    - Works for all K values")
        print()
        print("  CUTOFF:")
        print("    - Optimized T variable range: 1 to theoretical_ub")
        print("    - Reduced variables, faster encoding")
        print("    - Early infeasible elimination")
        print("    - Best for K near theoretical bounds")
        print()
        print("  HYBRID:")
        print("    - Incremental T→mutual-exclusion replacement")
        print("    - Control replacement level via --replacements parameter")
        print("    - replacements=0: equivalent to standard")
        print("    - replacements=max: equivalent to cutoff")
        print("    - Useful for performance comparison at specific K")
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
    
    # Parse encoding method and replacements (optional)
    encoding_method = 'standard'  # default
    num_replacements = 0  # default
    dump_clauses = False  # default
    verbose = False  # default
    
    for arg in sys.argv[4:]:
        if arg.startswith('--method='):
            encoding_method = arg.split('=')[1]
            if encoding_method not in ['standard', 'cutoff', 'hybrid']:
                print(f"Error: --method must be 'standard', 'cutoff', or 'hybrid', got '{encoding_method}'")
                sys.exit(1)
        elif arg.startswith('--replacements='):
            try:
                num_replacements = int(arg.split('=')[1])
                if num_replacements < 0:
                    print(f"Error: --replacements must be non-negative, got {num_replacements}")
                    sys.exit(1)
            except ValueError:
                print(f"Error: --replacements must be an integer, got '{arg.split('=')[1]}'")
                sys.exit(1)
        elif arg == '--dump-clauses' or arg == '--dump':
            dump_clauses = True
        elif arg == '--verbose' or arg == '-v':
            verbose = True
    
    # Validate replacements parameter
    if num_replacements > 0 and encoding_method != 'hybrid':
        print(f"Warning: --replacements={num_replacements} specified but method is '{encoding_method}'")
        print(f"Replacements only apply to 'hybrid' method. Ignoring --replacements parameter.")
        num_replacements = 0
    
    # Run solver
    print("=" * 80)
    print("CUSTOM K BANDWIDTH SOLVER")
    print("=" * 80)
    print(f"File: {mtx_file}")
    print(f"Solver: {solver_type.upper()}")
    print(f"Target K: {K}")
    print(f"Encoding: {encoding_method.upper()}")
    if encoding_method == 'hybrid':
        print(f"Replacements: {num_replacements}")
    if dump_clauses:
        print(f"Dump clauses: ENABLED")
    if verbose:
        print(f"Verbose mode: ENABLED")
    
    try:
        results = solve_custom_k(mtx_file, solver_type, K, encoding_method, num_replacements, dump_clauses, verbose)
        
        # Print final result
        print(f"\n" + "="*60)
        print(f"FINAL RESULT")
        print(f"="*60)
        
        status = results.get('status', 'unknown')
        if status == 'success':
            is_sat = results.get('is_sat', False)
            result_info = results.get('result_info', {})
            solve_time = result_info.get('solve_time', 0)
            encoding = results.get('encoding_method', 'unknown')
            num_repl = results.get('num_replacements', 0)
            
            if is_sat:
                print(f"✓ RESULT: SAT")
                print(f"✓ K = {K} is ACHIEVABLE")
                print(f"✓ The graph CAN be placed with bandwidth <= {K}")
                print(f"✓ Solve time: {solve_time:.3f}s")
                print(f"✓ Encoding: {encoding.upper()}")
                if encoding == 'hybrid':
                    print(f"✓ Replacements: {num_repl}")
                
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
                print(f"✗ Encoding: {encoding.upper()}")
                if encoding == 'hybrid':
                    print(f"✗ Replacements: {num_repl}")
                
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
