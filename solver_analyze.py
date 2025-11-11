#!/usr/bin/env python3
"""
solver_analyze.py
Analyze hybrid encoding performance: gradually replace high-distance (d>UB) activation
clauses with binary mutual-exclusion constraints on a per-edge basis.

Usage:
    python solver_analyze.py bfw62a.mtx --k=3
    python solver_analyze.py bfw62a.mtx --k=3 --cutoff-ratio=0.5 --seed=42
    python solver_analyze.py bfw62a.mtx --k=3 --sweep --output=results.csv
"""

import sys
import os
import time
import random
import argparse
import csv
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

from pysat.formula import IDPool
from pysat.solvers import Cadical195, Glucose42

# Import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from distance_encoder import encode_abs_distance_final
from distance_encoder_cutoff import calculate_theoretical_upper_bound
from position_constraints import encode_all_position_constraints, create_position_variables


@dataclass
class SolverMetrics:
    """Detailed solver metrics for analysis"""
    n: int
    num_edges: int
    k: int
    ub: int
    
    # Encoding configuration
    cutoff_ratio: float
    num_cutoff_edges: int
    num_original_edges: int
    
    # Formula statistics
    num_vars: int
    num_clauses: int
    num_binary_clauses: int
    num_ternary_clauses: int
    num_unary_clauses: int
    
    # High-T variable reduction
    saved_high_t_vars_per_edge: int
    total_saved_high_t_vars: int
    
    # Solver statistics
    result: str  # SAT/UNSAT/TIMEOUT
    encode_time: float
    solve_time: float
    decisions: int
    conflicts: int
    propagations: int
    restarts: int
    
    # Memory (if available)
    memory_mb: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary for CSV export"""
        return asdict(self)


class HybridBandwidthEncoder:
    """
    Hybrid encoder that allows per-edge choice between:
    - Original: Full T_1..T_{n-1} with activation/deactivation
    - Cutoff-style: T_1..T_UB + mutual exclusion for d>UB
    """
    
    def __init__(self, n: int, edges: List[Tuple[int, int]], K: int, 
                 cutoff_edge_ids: Set[int], vpool: IDPool, solver_name: str = 'cadical195'):
        """
        Args:
            n: Number of vertices
            edges: List of edges (u, v)
            K: Target bandwidth
            cutoff_edge_ids: Set of edge indices to apply cutoff-style encoding
            vpool: Variable pool
            solver_name: Solver to use
        """
        self.n = n
        self.edges = edges
        self.K = K
        self.cutoff_edge_ids = cutoff_edge_ids
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
        self.saved_high_t_vars = 0
        
        print(f"\n{'='*80}")
        print(f"HYBRID BANDWIDTH ENCODER")
        print(f"{'='*80}")
        print(f"Graph: n={n}, |E|={len(edges)}")
        print(f"Target K: {K}")
        print(f"Theoretical UB: {self.ub}")
        print(f"Cutoff edges: {len(cutoff_edge_ids)}/{len(edges)} ({100*len(cutoff_edge_ids)/len(edges):.1f}%)")
        print(f"Original edges: {len(edges) - len(cutoff_edge_ids)}")
    
    def add_mutual_exclusion_for_edge(self, U_vars: List[int], V_vars: List[int]):
        """
        Add mutual exclusion clauses for position pairs with distance > UB.
        (¬U_i ∨ ¬V_k) for all |i-k| >= UB+1
        
        This replaces T_{UB+1}..T_{n-1} activation clauses.
        """
        gap = self.ub + 1
        count = 0
        
        for i in range(1, self.n + 1):
            # Positions too far left: k < i - UB
            for k in range(1, max(1, i - gap + 1)):
                self.solver.add_clause([-U_vars[i-1], -V_vars[k-1]])
                count += 1
                self.num_binary_clauses += 1
            
            # Positions too far right: k > i + UB
            # Fix: Only iterate if start position is valid
            start = i + gap
            if start <= self.n:
                for k in range(start, self.n + 1):
                    self.solver.add_clause([-U_vars[i-1], -V_vars[k-1]])
                    count += 1
                    self.num_binary_clauses += 1
        
        self.total_clauses += count
        return count
    
    def encode_edge_original_style(self, edge_id: int, u: int, v: int, 
                                   X_vars: Dict, Y_vars: Dict, axis: str):
        """
        Encode edge with ORIGINAL style: Full T_1..T_{n-1}
        
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
        
        # Create T variables for full range T_1..T_{n-1}
        T_vars, clauses = encode_abs_distance_final(U_vars, V_vars, self.n, self.vpool, prefix)
        
        # Add all clauses (activation, deactivation, monotonic)
        for clause in clauses:
            self.solver.add_clause(clause)
            if len(clause) == 2:
                self.num_binary_clauses += 1
            elif len(clause) == 3:
                self.num_ternary_clauses += 1
            self.total_clauses += 1
        
        # Add bandwidth <= K constraints
        if self.K < len(T_vars):
            self.solver.add_clause([-T_vars[self.K]])
            self.num_unary_clauses += 1
            self.total_clauses += 1
        
        return T_vars
    
    def encode_edge_cutoff_style(self, edge_id: int, u: int, v: int,
                                 X_vars: Dict, Y_vars: Dict, axis: str):
        """
        Encode edge with CUTOFF style: T_1..T_UB + mutual exclusion for d>UB
        
        CRITICAL: 
        - Only create T_d for d <= UB
        - Keep activation/deactivation for d <= UB (important for propagation)
        - Drop deactivation for d > UB (those pairs are forbidden by mutual exclusion)
        - Add mutual exclusion for |i-k| >= UB+1
        """
        if axis == 'x':
            U_vars, V_vars = X_vars[u], X_vars[v]
            prefix = f"Tx[{u},{v}]"
        else:
            U_vars, V_vars = Y_vars[u], Y_vars[v]
            prefix = f"Ty[{u},{v}]"
        
        # Create T variables ONLY up to T_UB
        T_vars_dict = {}
        for d in range(1, self.ub + 1):
            T_vars_dict[d] = self.vpool.id(f'{prefix}_geq_{d}')
        
        # Activation clauses for d <= UB (KEEP - important for propagation)
        # (U_k ∧ V_{k-d}) → T_d  ≡  ¬U_k ∨ ¬V_{k-d} ∨ T_d
        for k in range(1, self.n + 1):
            for d in range(1, min(k, self.ub + 1)):
                u_pos = k - d
                if u_pos >= 1 and d <= self.ub:
                    # V_k ∧ U_{k-d} → T_d
                    self.solver.add_clause([-V_vars[k-1], -U_vars[u_pos-1], T_vars_dict[d]])
                    self.num_ternary_clauses += 1
                    self.total_clauses += 1
                    
                    # U_k ∧ V_{k-d} → T_d
                    self.solver.add_clause([-U_vars[k-1], -V_vars[u_pos-1], T_vars_dict[d]])
                    self.num_ternary_clauses += 1
                    self.total_clauses += 1
        
        # Deactivation clauses for d < UB (KEEP - for tight bounds)
        # (U_k ∧ V_{k-d}) → ¬T_{d+1}  ≡  ¬U_k ∨ ¬V_{k-d} ∨ ¬T_{d+1}
        for k in range(1, self.n + 1):
            for d in range(1, min(k, self.ub)):
                u_pos = k - d
                if u_pos >= 1 and d + 1 <= self.ub:
                    # V_k ∧ U_{k-d} → ¬T_{d+1}
                    self.solver.add_clause([-V_vars[k-1], -U_vars[u_pos-1], -T_vars_dict[d+1]])
                    self.num_ternary_clauses += 1
                    self.total_clauses += 1
                    
                    # U_k ∧ V_{k-d} → ¬T_{d+1}
                    self.solver.add_clause([-U_vars[k-1], -V_vars[u_pos-1], -T_vars_dict[d+1]])
                    self.num_ternary_clauses += 1
                    self.total_clauses += 1
        
        # Monotonic chain: T_{d+1} → T_d
        for d in range(1, self.ub):
            if d + 1 <= self.ub:
                self.solver.add_clause([-T_vars_dict[d+1], T_vars_dict[d]])
                self.num_binary_clauses += 1
                self.total_clauses += 1
        
        # Add mutual exclusion for d > UB (REPLACES high-T activation)
        mut_ex_count = self.add_mutual_exclusion_for_edge(U_vars, V_vars)
        
        # Add bandwidth <= K constraint
        if self.K < self.ub and (self.K + 1) in T_vars_dict:
            self.solver.add_clause([-T_vars_dict[self.K + 1]])
            self.num_unary_clauses += 1
            self.total_clauses += 1
        
        # Track saved high-T variables
        saved_vars = self.n - 1 - self.ub
        self.saved_high_t_vars += saved_vars
        
        return T_vars_dict
    
    def encode(self):
        """Build complete hybrid encoding"""
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
        
        # Encode edges (hybrid)
        print(f"\nEncoding edges (hybrid)...")
        edge_start = time.time()
        
        for edge_id, (u, v) in enumerate(self.edges):
            is_cutoff = edge_id in self.cutoff_edge_ids
            
            if is_cutoff:
                # Cutoff-style: T_1..T_UB + mutual exclusion
                Tx_vars = self.encode_edge_cutoff_style(edge_id, u, v, X_vars, Y_vars, 'x')
                Ty_vars = self.encode_edge_cutoff_style(edge_id, u, v, X_vars, Y_vars, 'y')
                
                # Add cross constraints: Tx>=i → Ty<=K-i
                effective_k = min(self.K, self.ub)
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
            else:
                # Original-style: Full T_1..T_{n-1}
                Tx_vars = self.encode_edge_original_style(edge_id, u, v, X_vars, Y_vars, 'x')
                Ty_vars = self.encode_edge_original_style(edge_id, u, v, X_vars, Y_vars, 'y')
                
                # Add cross constraints: Tx>=i → Ty<=K-i
                for i in range(1, self.K + 1):
                    remaining = self.K - i
                    if remaining >= 0:
                        if i-1 < len(Tx_vars) and remaining < len(Ty_vars):
                            self.solver.add_clause([-Tx_vars[i-1], -Ty_vars[remaining]])
                            self.num_binary_clauses += 1
                            self.total_clauses += 1
                        if i-1 < len(Ty_vars) and remaining < len(Tx_vars):
                            self.solver.add_clause([-Ty_vars[i-1], -Tx_vars[remaining]])
                            self.num_binary_clauses += 1
                            self.total_clauses += 1
        
        edge_time = time.time() - edge_start
        print(f"  Encoded {len(self.edges)} edges in {edge_time:.2f}s")
        
        # Print statistics
        print(f"\n{'─'*80}")
        print("ENCODING STATISTICS")
        print(f"{'─'*80}")
        print(f"Cutoff edges: {len(self.cutoff_edge_ids)} ({100*len(self.cutoff_edge_ids)/len(self.edges):.1f}%)")
        print(f"Original edges: {len(self.edges) - len(self.cutoff_edge_ids)}")
        print(f"Total clauses: {self.total_clauses:,}")
        print(f"  Unary clauses: {self.num_unary_clauses:,} ({100*self.num_unary_clauses/self.total_clauses:.1f}%)")
        print(f"  Binary clauses: {self.num_binary_clauses:,} ({100*self.num_binary_clauses/self.total_clauses:.1f}%)")
        print(f"  Ternary clauses: {self.num_ternary_clauses:,} ({100*self.num_ternary_clauses/self.total_clauses:.1f}%)")
        print(f"Saved high-T variables: {self.saved_high_t_vars} (from cutoff edges)")
        print(f"  Per cutoff edge: {self.n - 1 - self.ub} vars × 2 axes = {2*(self.n - 1 - self.ub)} vars")
    
    def solve(self) -> Tuple[bool, float, Dict]:
        """
        Solve the formula and collect detailed metrics.
        
        Returns:
            (result, solve_time, stats_dict)
        """
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
            # Try to get statistics from solver
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


def select_cutoff_edges(num_edges: int, ratio: float, seed: Optional[int] = None) -> Set[int]:
    """
    Select edge indices to apply cutoff-style encoding.
    
    Args:
        num_edges: Total number of edges
        ratio: Fraction of edges to cutoff (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Set of edge indices
    """
    if seed is not None:
        random.seed(seed)
    
    num_cutoff = int(num_edges * ratio)
    all_ids = list(range(num_edges))
    cutoff_ids = set(random.sample(all_ids, num_cutoff))
    
    return cutoff_ids


def run_single_experiment(n: int, edges: List[Tuple[int, int]], K: int,
                         cutoff_ratio: float, seed: Optional[int] = None,
                         solver_name: str = 'cadical195') -> SolverMetrics:
    """
    Run a single experiment with given cutoff ratio.
    
    Returns:
        SolverMetrics object with all collected data
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: Cutoff Ratio = {cutoff_ratio:.2%}")
    print(f"{'='*80}")
    
    # Select edges for cutoff
    cutoff_edge_ids = select_cutoff_edges(len(edges), cutoff_ratio, seed)
    
    # Create encoder
    vpool = IDPool()
    encoder = HybridBandwidthEncoder(n, edges, K, cutoff_edge_ids, vpool, solver_name)
    
    # Encode
    encode_start = time.time()
    encoder.encode()
    encode_time = time.time() - encode_start
    
    # Solve
    result, solve_time, stats = encoder.solve()
    
    # Build metrics
    ub = calculate_theoretical_upper_bound(n)
    
    # Use solver's clause count if available (after simplification), otherwise use internal count
    num_clauses = stats.get('clauses', encoder.total_clauses)
    
    metrics = SolverMetrics(
        n=n,
        num_edges=len(edges),
        k=K,
        ub=ub,
        cutoff_ratio=cutoff_ratio,
        num_cutoff_edges=len(cutoff_edge_ids),
        num_original_edges=len(edges) - len(cutoff_edge_ids),
        num_vars=stats.get('vars', 0),
        num_clauses=num_clauses,
        num_binary_clauses=encoder.num_binary_clauses,
        num_ternary_clauses=encoder.num_ternary_clauses,
        num_unary_clauses=encoder.num_unary_clauses,
        saved_high_t_vars_per_edge=2 * (n - 1 - ub),
        total_saved_high_t_vars=encoder.saved_high_t_vars,
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
             ratios: List[float], seed: Optional[int] = None,
             solver_name: str = 'cadical195') -> List[SolverMetrics]:
    """
    Run a sweep of experiments with different cutoff ratios.
    
    Args:
        n: Number of vertices
        edges: List of edges
        K: Target bandwidth
        ratios: List of cutoff ratios to test (e.g., [0.0, 0.25, 0.5, 0.75, 1.0])
        seed: Random seed
        solver_name: Solver to use
    
    Returns:
        List of SolverMetrics for each ratio
    """
    results = []
    
    for ratio in ratios:
        metrics = run_single_experiment(n, edges, K, ratio, seed, solver_name)
        results.append(metrics)
    
    return results


def save_results_csv(results: List[SolverMetrics], output_file: str):
    """Save experiment results to CSV"""
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


def print_comparison_table(results: List[SolverMetrics]):
    """Print comparison table of results"""
    print(f"\n{'='*100}")
    print("RESULTS COMPARISON")
    print(f"{'='*100}")
    
    header = f"{'Ratio':<8} {'Cutoff':<8} {'Orig':<6} {'Clauses':<12} {'Binary%':<10} {'Time(s)':<10} {'Decisions':<12} {'Conflicts':<12}"
    print(header)
    print('-' * 100)
    
    for m in results:
        binary_pct = 100 * m.num_binary_clauses / m.num_clauses if m.num_clauses > 0 else 0
        row = (f"{m.cutoff_ratio:<8.2f} "
               f"{m.num_cutoff_edges:<8} "
               f"{m.num_original_edges:<6} "
               f"{m.num_clauses:<12,} "
               f"{binary_pct:<10.1f} "
               f"{m.solve_time:<10.2f} "
               f"{m.decisions:<12,} "
               f"{m.conflicts:<12,}")
        print(row)
    
    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze hybrid encoding performance with varying cutoff ratios'
    )
    parser.add_argument('mtx_file', help='MTX file to solve')
    parser.add_argument('--k', type=int, required=True, help='Target bandwidth K')
    parser.add_argument('--cutoff-ratio', type=float, default=0.5,
                       help='Fraction of edges to apply cutoff encoding (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for edge selection (default: 42)')
    parser.add_argument('--solver', choices=['cadical195', 'glucose42'],
                       default='cadical195', help='SAT solver to use')
    parser.add_argument('--sweep', action='store_true',
                       help='Run sweep with ratios [0.0, 0.25, 0.5, 0.75, 1.0]')
    parser.add_argument('--output', type=str,
                       help='Output CSV file for results (required with --sweep)')
    
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
    
    if args.sweep:
        # Run sweep
        ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        print(f"\n{'#'*80}")
        print(f"# RUNNING SWEEP: Ratios = {ratios}")
        print(f"{'#'*80}")
        
        results = run_sweep(n, edges, args.k, ratios, args.seed, args.solver)
        
        # Print comparison
        print_comparison_table(results)
        
        # Save to CSV if requested
        if args.output:
            save_results_csv(results, args.output)
        else:
            print("Note: Use --output to save results to CSV")
    
    else:
        # Single run
        metrics = run_single_experiment(n, edges, args.k, args.cutoff_ratio,
                                       args.seed, args.solver)
        
        print(f"\n{'='*80}")
        print("FINAL METRICS")
        print(f"{'='*80}")
        print(f"Result: {metrics.result}")
        print(f"Solve time: {metrics.solve_time:.3f}s")
        print(f"Cutoff ratio: {metrics.cutoff_ratio:.2%}")
        print(f"Binary clause ratio: {100*metrics.num_binary_clauses/metrics.num_clauses:.1f}%")
        print(f"Saved high-T vars: {metrics.total_saved_high_t_vars}")


if __name__ == '__main__':
    main()
