#!/usr/bin/env python3
"""
solver_analyze_by_distance.py
Analyze performance by gradually replacing high-distance T_d variables (d > UB) 
with mutual-exclusion constraints, one distance at a time.

Instead of replacing by edge ratio, this replaces by distance threshold:
- d_cutoff = UB: Original encoding (all T_d with activation/deactivation)
- d_cutoff = UB+1: Replace T_{UB+1} with mutual exclusion, keep T_1..T_UB
- d_cutoff = UB+5: Replace T_{UB+1}..T_{UB+5}, keep rest original
- d_cutoff = n-1: Full cutoff (all d>UB replaced)

IMPORTANT: We KEEP deactivation clauses for ALL distances, including those replaced.
This is different from pure cutoff which drops deactivation for d>UB.

Usage:
    python solver_analyze_by_distance.py bfw62a.mtx --k=3
    python solver_analyze_by_distance.py bfw62a.mtx --k=3 --d-cutoff=15
    python solver_analyze_by_distance.py bfw62a.mtx --k=3 --sweep --output=results_by_d.csv
"""

import sys
import os
import time
import argparse
import csv
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, asdict

from pysat.formula import IDPool
from pysat.solvers import Cadical195, Glucose42

# Import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from distance_encoder import encode_abs_distance_final
from distance_encoder_cutoff import calculate_theoretical_upper_bound
from position_constraints import encode_all_position_constraints, create_position_variables


@dataclass
class SolverMetricsByDistance:
    """Solver metrics for distance-based replacement analysis"""
    n: int
    num_edges: int
    k: int
    ub: int
    
    # Encoding configuration
    d_cutoff: int  # Distances > d_cutoff are replaced by mutual exclusion
    num_replaced_distances: int  # Number of T_d replaced
    num_original_distances: int  # Number of T_d kept with activation/deactivation
    
    # Formula statistics
    num_vars: int
    num_clauses: int
    num_binary_clauses: int
    num_ternary_clauses: int
    num_unary_clauses: int
    
    # Variable reduction
    saved_t_vars_per_edge: int
    total_saved_t_vars: int
    
    # Solver statistics
    result: str
    encode_time: float
    solve_time: float
    decisions: int
    conflicts: int
    propagations: int
    restarts: int
    
    memory_mb: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)


class DistanceHybridEncoder:
    """
    Hybrid encoder that replaces T_d variables by distance threshold.
    
    For d <= d_cutoff: Use original activation/deactivation
    For d > d_cutoff: Replace with mutual exclusion (¬U_i ∨ ¬V_k for |i-k|=d)
    
    KEEP deactivation for all distances (different from pure cutoff).
    """
    
    def __init__(self, n: int, edges: List[Tuple[int, int]], K: int,
                 d_cutoff: int, vpool: IDPool, solver_name: str = 'cadical195'):
        """
        Args:
            n: Number of vertices
            edges: List of edges
            K: Target bandwidth
            d_cutoff: Distance cutoff (replace T_d for d > d_cutoff with mutual exclusion)
            vpool: Variable pool
            solver_name: Solver to use
        """
        self.n = n
        self.edges = edges
        self.K = K
        self.d_cutoff = d_cutoff
        self.vpool = vpool
        self.ub = calculate_theoretical_upper_bound(n)
        
        # Create solver
        if solver_name == 'glucose42':
            self.solver = Glucose42()
        else:
            self.solver = Cadical195()
        
        # Statistics
        self.num_binary_clauses = 0
        self.num_ternary_clauses = 0
        self.num_unary_clauses = 0
        self.total_clauses = 0
        self.saved_t_vars = 0
        
        # Validate d_cutoff
        if d_cutoff < self.ub:
            print(f"WARNING: d_cutoff={d_cutoff} < UB={self.ub}")
            print(f"         This will replace some T_d in the 'near' region")
        
        print(f"\n{'='*80}")
        print(f"DISTANCE-BASED HYBRID ENCODER")
        print(f"{'='*80}")
        print(f"Graph: n={n}, |E|={len(edges)}")
        print(f"Target K: {K}")
        print(f"Theoretical UB: {self.ub}")
        print(f"Distance cutoff: d_cutoff = {d_cutoff}")
        print(f"  Keep T_1..T_{d_cutoff} with activation/deactivation")
        print(f"  Replace T_{d_cutoff+1}..T_{n-1} with mutual exclusion")
        print(f"  Number of replaced distances: {max(0, n - 1 - d_cutoff)}")
    
    def add_mutual_exclusion_for_distance(self, U_vars: List[int], V_vars: List[int], d: int):
        """
        Add mutual exclusion clauses for a SPECIFIC distance d.
        (¬U_i ∨ ¬V_k) for all (i,k) with |i-k| = d
        
        This replaces the activation clause for T_d.
        """
        count = 0
        
        for i in range(1, self.n + 1):
            # k = i - d (left direction)
            k = i - d
            if 1 <= k <= self.n:
                self.solver.add_clause([-U_vars[i-1], -V_vars[k-1]])
                count += 1
                self.num_binary_clauses += 1
            
            # k = i + d (right direction)
            k = i + d
            if 1 <= k <= self.n:
                self.solver.add_clause([-U_vars[i-1], -V_vars[k-1]])
                count += 1
                self.num_binary_clauses += 1
        
        self.total_clauses += count
        return count
    
    def encode_edge_axis(self, edge_id: int, u: int, v: int,
                        X_vars: Dict, Y_vars: Dict, axis: str):
        """
        Encode edge with distance-based hybrid:
        - Create T_1..T_{d_cutoff}
        - Add activation/deactivation for T_1..T_{d_cutoff}
        - Add mutual exclusion for distances d > d_cutoff
        - KEEP deactivation for all distances (including replaced ones)
        
        Args:
            edge_id: Edge index
            u, v: Vertex pair
            X_vars, Y_vars: Position variable dicts
            axis: 'x' or 'y'
        """
        if axis == 'x':
            U_vars, V_vars = X_vars[u], X_vars[v]
            prefix = f"Tx[{u},{v}]"
        else:
            U_vars, V_vars = Y_vars[u], Y_vars[v]
            prefix = f"Ty[{u},{v}]"
        
        # Create T variables up to d_cutoff
        T_vars_dict = {}
        for d in range(1, min(self.d_cutoff + 1, self.n)):
            T_vars_dict[d] = self.vpool.id(f'{prefix}_geq_{d}')
        
        # Track saved variables
        if self.d_cutoff < self.n - 1:
            saved = self.n - 1 - self.d_cutoff
            self.saved_t_vars += saved
        
        # ===== ACTIVATION clauses for d <= d_cutoff =====
        # (U_k ∧ V_{k-d}) → T_d  ≡  ¬U_k ∨ ¬V_{k-d} ∨ T_d
        for k in range(1, self.n + 1):
            for d in range(1, min(k, self.d_cutoff + 1)):
                u_pos = k - d
                if u_pos >= 1 and d <= self.d_cutoff:
                    # V_k ∧ U_{k-d} → T_d
                    self.solver.add_clause([-V_vars[k-1], -U_vars[u_pos-1], T_vars_dict[d]])
                    self.num_ternary_clauses += 1
                    self.total_clauses += 1
                    
                    # U_k ∧ V_{k-d} → T_d
                    self.solver.add_clause([-U_vars[k-1], -V_vars[u_pos-1], T_vars_dict[d]])
                    self.num_ternary_clauses += 1
                    self.total_clauses += 1
        
        # ===== DEACTIVATION clauses for d < d_cutoff =====
        # (U_k ∧ V_{k-d}) → ¬T_{d+1}  ≡  ¬U_k ∨ ¬V_{k-d} ∨ ¬T_{d+1}
        # IMPORTANT: Keep for all d where T_{d+1} exists
        for k in range(1, self.n + 1):
            for d in range(1, min(k, self.d_cutoff)):
                u_pos = k - d
                if u_pos >= 1 and d + 1 <= self.d_cutoff:
                    # V_k ∧ U_{k-d} → ¬T_{d+1}
                    self.solver.add_clause([-V_vars[k-1], -U_vars[u_pos-1], -T_vars_dict[d+1]])
                    self.num_ternary_clauses += 1
                    self.total_clauses += 1
                    
                    # U_k ∧ V_{k-d} → ¬T_{d+1}
                    self.solver.add_clause([-U_vars[k-1], -V_vars[u_pos-1], -T_vars_dict[d+1]])
                    self.num_ternary_clauses += 1
                    self.total_clauses += 1
        
        # ===== MONOTONIC chain: T_{d+1} → T_d for d < d_cutoff =====
        for d in range(1, min(self.d_cutoff, self.n - 1)):
            if d + 1 <= self.d_cutoff:
                self.solver.add_clause([-T_vars_dict[d+1], T_vars_dict[d]])
                self.num_binary_clauses += 1
                self.total_clauses += 1
        
        # ===== MUTUAL EXCLUSION for d > d_cutoff =====
        # Replace activation for T_{d_cutoff+1}..T_{n-1}
        for d in range(self.d_cutoff + 1, self.n):
            self.add_mutual_exclusion_for_distance(U_vars, V_vars, d)
        
        # ===== FIX: FORBID USELESS T VARIABLES (d > K) =====
        # Explicitly set T_d = FALSE for all d > K to prevent solver
        # from wasting time exploring impossible branches
        num_forbidden = 0
        for d in range(self.K + 1, min(self.d_cutoff + 1, self.n)):
            if d in T_vars_dict:
                # Force T_d = FALSE (distance >= d is not allowed)
                self.solver.add_clause([-T_vars_dict[d]])
                self.num_unary_clauses += 1
                self.total_clauses += 1
                num_forbidden += 1
        
        # Log the fix (only once per encoder instance)
        if num_forbidden > 0 and not hasattr(self, '_logged_forbidden'):
            print(f"  ✓ Added {num_forbidden} unary clauses per axis: ¬T_d for d={self.K+1}..{min(self.d_cutoff, self.n-1)}")
            print(f"    (Preventing solver from exploring useless T_d=TRUE branches)")
            self._logged_forbidden = True
        
        # Add bandwidth <= K constraint (now somewhat redundant but kept for clarity)
        if self.K <= self.d_cutoff and (self.K + 1) in T_vars_dict:
            # This clause is now redundant if we added ¬T_{K+1} above,
            # but we keep it for documentation purposes
            pass  # Already added above
        
        return T_vars_dict
    
    def encode(self):
        """Build complete distance-based hybrid encoding"""
        print(f"\n{'─'*80}")
        print("ENCODING PHASE")
        print(f"{'─'*80}")
        
        # Create position variables
        print("Creating position variables...")
        X_vars, Y_vars = create_position_variables(self.n, self.vpool)
        
        # Add position constraints
        print("Adding position constraints...")
        pos_start = time.time()
        pos_count = 0
        for clause in encode_all_position_constraints(self.n, X_vars, Y_vars, self.vpool):
            self.solver.add_clause(clause)
            pos_count += 1
            if len(clause) == 2:
                self.num_binary_clauses += 1
            elif len(clause) == 3:
                self.num_ternary_clauses += 1
        self.total_clauses += pos_count
        pos_time = time.time() - pos_start
        print(f"  Added {pos_count} position clauses in {pos_time:.2f}s")
        
        # Encode edges (all with same d_cutoff)
        print(f"\nEncoding edges with d_cutoff={self.d_cutoff}...")
        edge_start = time.time()
        
        for edge_id, (u, v) in enumerate(self.edges):
            # Encode both axes
            Tx_vars = self.encode_edge_axis(edge_id, u, v, X_vars, Y_vars, 'x')
            Ty_vars = self.encode_edge_axis(edge_id, u, v, X_vars, Y_vars, 'y')
            
            # Add cross constraints: Tx>=i → Ty<=K-i
            effective_k = min(self.K, self.d_cutoff)
            for i in range(1, effective_k + 1):
                remaining = effective_k - i
                if remaining >= 0:
                    if i in Tx_vars and (remaining + 1) in Ty_vars:
                        self.solver.add_clause([-Tx_vars[i], -Ty_vars[remaining + 1]])
                        self.num_binary_clauses += 1
                        self.total_clauses += 1
                    if i in Ty_vars and (remaining + 1) in Tx_vars:
                        self.solver.add_clause([-Ty_vars[i], -Tx_vars[remaining + 1]])
                        self.num_binary_clauses += 1
                        self.total_clauses += 1
        
        edge_time = time.time() - edge_start
        print(f"  Encoded {len(self.edges)} edges in {edge_time:.2f}s")
        
        # Print statistics
        print(f"\n{'─'*80}")
        print("ENCODING STATISTICS")
        print(f"{'─'*80}")
        print(f"Distance cutoff: d_cutoff = {self.d_cutoff}")
        print(f"  T_1..T_{self.d_cutoff}: Original (activation/deactivation)")
        print(f"  T_{self.d_cutoff+1}..T_{self.n-1}: Mutual exclusion")
        print(f"Total clauses: {self.total_clauses:,}")
        print(f"  Unary clauses: {self.num_unary_clauses:,} ({100*self.num_unary_clauses/self.total_clauses:.2f}%)")
        print(f"  Binary clauses: {self.num_binary_clauses:,} ({100*self.num_binary_clauses/self.total_clauses:.2f}%)")
        print(f"  Ternary clauses: {self.num_ternary_clauses:,} ({100*self.num_ternary_clauses/self.total_clauses:.2f}%)")
        print(f"Saved T variables: {self.saved_t_vars} (from replaced distances)")
        print(f"  Per edge: {max(0, self.n - 1 - self.d_cutoff)} vars × 2 axes = {2*max(0, self.n - 1 - self.d_cutoff)} vars")
    
    def solve(self) -> Tuple[bool, float, Dict]:
        """Solve and collect metrics"""
        print(f"\n{'─'*80}")
        print("SOLVING PHASE")
        print(f"{'─'*80}")
        
        print("Starting solver...")
        start_time = time.time()
        result = self.solver.solve()
        solve_time = time.time() - start_time
        
        # Collect solver statistics
        stats = {}
        try:
            if hasattr(self.solver, 'nof_vars'):
                stats['vars'] = self.solver.nof_vars()
            if hasattr(self.solver, 'nof_clauses'):
                stats['clauses'] = self.solver.nof_clauses()
            if hasattr(self.solver, 'decisions'):
                stats['decisions'] = self.solver.decisions()
            if hasattr(self.solver, 'conflicts'):
                stats['conflicts'] = self.solver.conflicts()
            if hasattr(self.solver, 'propagations'):
                stats['propagations'] = self.solver.propagations()
            if hasattr(self.solver, 'restarts'):
                stats['restarts'] = self.solver.restarts()
        except Exception as e:
            print(f"Warning: Could not retrieve some solver statistics: {e}")
        
        # Print results
        print(f"\nResult: {'SAT' if result else 'UNSAT'}")
        print(f"Solve time: {solve_time:.3f}s")
        
        if stats:
            print(f"\nSolver statistics:")
            if 'vars' in stats:
                print(f"  Variables: {stats['vars']:,}")
            if 'clauses' in stats:
                print(f"  Clauses: {stats['clauses']:,}")
            if 'decisions' in stats:
                print(f"  Decisions: {stats['decisions']:,}")
            if 'conflicts' in stats:
                print(f"  Conflicts: {stats['conflicts']:,}")
            if 'propagations' in stats:
                print(f"  Propagations: {stats['propagations']:,}")
            if 'restarts' in stats:
                print(f"  Restarts: {stats['restarts']:,}")
        
        return result, solve_time, stats
    
    def cleanup(self):
        """Clean up solver resources"""
        self.solver.delete()


def parse_mtx_file(filename: str) -> Tuple[int, List[Tuple[int, int]]]:
    """Parse MTX file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    header_found = False
    edges_set = set()
    n = 0
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        
        if not header_found:
            parts = line.split()
            if len(parts) >= 2:
                n = int(parts[0])
                header_found = True
                continue
        
        parts = line.split()
        if len(parts) >= 2:
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                edges_set.add((min(u, v), max(u, v)))
    
    edges = list(edges_set)
    return n, edges


def run_single_experiment(n: int, edges: List[Tuple[int, int]], K: int,
                         d_cutoff: int, solver_name: str = 'cadical195') -> SolverMetricsByDistance:
    """Run single experiment with given d_cutoff"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: d_cutoff = {d_cutoff}")
    print(f"{'='*80}")
    
    # Create encoder
    vpool = IDPool()
    encoder = DistanceHybridEncoder(n, edges, K, d_cutoff, vpool, solver_name)
    
    # Encode
    encode_start = time.time()
    encoder.encode()
    encode_time = time.time() - encode_start
    
    # Solve
    result, solve_time, stats = encoder.solve()
    
    # Build metrics
    ub = calculate_theoretical_upper_bound(n)
    num_clauses = stats.get('clauses', encoder.total_clauses)
    
    metrics = SolverMetricsByDistance(
        n=n,
        num_edges=len(edges),
        k=K,
        ub=ub,
        d_cutoff=d_cutoff,
        num_replaced_distances=max(0, n - 1 - d_cutoff),
        num_original_distances=min(d_cutoff, n - 1),
        num_vars=stats.get('vars', 0),
        num_clauses=num_clauses,
        num_binary_clauses=encoder.num_binary_clauses,
        num_ternary_clauses=encoder.num_ternary_clauses,
        num_unary_clauses=encoder.num_unary_clauses,
        saved_t_vars_per_edge=2 * max(0, n - 1 - d_cutoff),
        total_saved_t_vars=encoder.saved_t_vars,
        result='SAT' if result else 'UNSAT',
        encode_time=encode_time,
        solve_time=solve_time,
        decisions=stats.get('decisions', 0),
        conflicts=stats.get('conflicts', 0),
        propagations=stats.get('propagations', 0),
        restarts=stats.get('restarts', 0)
    )
    
    # Cleanup
    encoder.cleanup()
    
    return metrics


def run_sweep(n: int, edges: List[Tuple[int, int]], K: int,
             d_cutoffs: List[int], solver_name: str = 'cadical195') -> List[SolverMetricsByDistance]:
    """Run sweep with different d_cutoff values"""
    results = []
    
    for d_cutoff in d_cutoffs:
        metrics = run_single_experiment(n, edges, K, d_cutoff, solver_name)
        results.append(metrics)
    
    return results


def save_results_csv(results: List[SolverMetricsByDistance], output_file: str):
    """Save results to CSV"""
    if not results:
        print("No results to save")
        return
    
    fieldnames = list(results[0].to_dict().keys())
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in results:
            writer.writerow(metrics.to_dict())
    
    print(f"\n✓ Results saved to {output_file}")


def print_comparison_table(results: List[SolverMetricsByDistance]):
    """Print comparison table"""
    print(f"\n{'='*120}")
    print("RESULTS COMPARISON BY DISTANCE CUTOFF")
    print(f"{'='*120}")
    
    header = (f"{'d_cutoff':<10} {'Replaced':<10} {'Original':<10} {'Clauses':<12} "
             f"{'Binary%':<10} {'EncTime':<10} {'SolveTime':<12} {'Decisions':<12} {'Conflicts':<12}")
    print(header)
    print('-' * 120)
    
    for m in results:
        binary_pct = 100 * m.num_binary_clauses / m.num_clauses if m.num_clauses > 0 else 0
        row = (f"{m.d_cutoff:<10} "
               f"{m.num_replaced_distances:<10} "
               f"{m.num_original_distances:<10} "
               f"{m.num_clauses:<12,} "
               f"{binary_pct:<10.2f} "
               f"{m.encode_time:<10.2f} "
               f"{m.solve_time:<12.2f} "
               f"{m.decisions:<12,} "
               f"{m.conflicts:<12,}")
        print(row)
    
    print(f"{'='*120}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze performance by gradually replacing T_d variables with mutual exclusion'
    )
    parser.add_argument('mtx_file', help='MTX file to solve')
    parser.add_argument('--k', type=int, required=True, help='Target bandwidth K')
    parser.add_argument('--d-cutoff', type=int,
                       help='Distance cutoff (replace T_d for d > d_cutoff)')
    parser.add_argument('--solver', choices=['cadical195', 'glucose42'],
                       default='cadical195', help='SAT solver to use')
    parser.add_argument('--sweep', action='store_true',
                       help='Run sweep with multiple d_cutoff values')
    parser.add_argument('--output', type=str,
                       help='Output CSV file for results')
    
    args = parser.parse_args()
    
    # Find and parse MTX file
    search_paths = [
        args.mtx_file,
        f"mtx/{args.mtx_file}",
        f"mtx/oc2/{args.mtx_file}",
        f"mtx/cutoff2/{args.mtx_file}",
    ]
    
    found_file = None
    for path in search_paths:
        if os.path.exists(path):
            found_file = path
            break
    
    if not found_file:
        print(f"Error: File '{args.mtx_file}' not found")
        sys.exit(1)
    
    print(f"Reading graph from: {found_file}")
    n, edges = parse_mtx_file(found_file)
    print(f"Loaded: n={n}, |E|={len(edges)}")
    
    ub = calculate_theoretical_upper_bound(n)
    
    if args.sweep:
        # Run sweep: UB, UB+5, UB+10, ..., n-1
        d_cutoffs = []
        d_cutoffs.append(n - 1)  # Original (no replacement)
        
        # Add intermediate values
        step = max(1, (n - 1 - ub) // 5)  # ~5 steps
        for d in range(ub, n - 1, step):
            if d not in d_cutoffs:
                d_cutoffs.append(d)
        
        if ub not in d_cutoffs:
            d_cutoffs.append(ub)  # Full cutoff
        
        d_cutoffs = sorted(d_cutoffs, reverse=True)
        
        print(f"\n{'#'*80}")
        print(f"# RUNNING SWEEP: d_cutoff values = {d_cutoffs}")
        print(f"{'#'*80}")
        
        results = run_sweep(n, edges, args.k, d_cutoffs, args.solver)
        
        # Print comparison
        print_comparison_table(results)
        
        # Save to CSV
        if args.output:
            save_results_csv(results, args.output)
        else:
            print("Note: Use --output to save results to CSV")
    
    else:
        # Single run
        if args.d_cutoff is None:
            print(f"Error: --d-cutoff required for single run (or use --sweep)")
            sys.exit(1)
        
        metrics = run_single_experiment(n, edges, args.k, args.d_cutoff, args.solver)
        
        print(f"\n{'='*80}")
        print("FINAL METRICS")
        print(f"{'='*80}")
        print(f"Result: {metrics.result}")
        print(f"d_cutoff: {metrics.d_cutoff}")
        print(f"Replaced distances: {metrics.num_replaced_distances}")
        print(f"Encode time: {metrics.encode_time:.3f}s")
        print(f"Solve time: {metrics.solve_time:.3f}s")
        print(f"Binary clause ratio: {100*metrics.num_binary_clauses/metrics.num_clauses:.2f}%")
        print(f"Saved T vars: {metrics.total_saved_t_vars}")


if __name__ == '__main__':
    main()
