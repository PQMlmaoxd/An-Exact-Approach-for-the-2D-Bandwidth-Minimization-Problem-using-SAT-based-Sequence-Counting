# analyze_mtx_realtime.py
"""
Real-time MTX analysis with PROPER incremental optimization matching incremental_bandwidth_solver.py

This version actually runs incremental optimization K=UB, K=UB-1, ... down to optimal,
reporting SAT/UNSAT for each K just like the real incremental solvers.

Usage:
    python analyze_mtx_realtime.py <mtx_file> --solver cadical195 --timeout 600
"""

import sys
import os
import time
import signal
import argparse
from collections import Counter
from pysat.formula import IDPool
from pysat.solvers import Glucose42, Cadical195

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distance_encoder import encode_abs_distance_final
from distance_encoder_cutoff import encode_abs_distance_cutoff, calculate_theoretical_upper_bound
from position_constraints import encode_all_position_constraints, create_position_variables


def parse_mtx_file(filename):
    """Parse MTX file and return n, edges"""
    print(f"\n{'='*80}")
    print(f"PARSING MTX FILE")
    print(f"{'='*80}")
    print(f"File: {filename}")
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None, None
    
    header_found = False
    edges_set = set()
    n = 0
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        if not line or line.startswith('%'):
            continue
        
        if not header_found:
            parts = line.split()
            if len(parts) >= 2:
                n = int(parts[0])
                print(f"Graph: {n} vertices, {parts[1] if len(parts) > 1 else 'unknown'} edges (from header)")
                header_found = True
                continue
        
        try:
            parts = line.split()
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                edges_set.add((min(u,v), max(u,v)))
        except (ValueError, IndexError):
            continue
    
    edges = list(edges_set)
    print(f"Loaded: {n} vertices, {len(edges)} edges")
    
    return n, edges


class TimeoutException(Exception):
    """Exception raised when solver times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Solver timeout!")


def run_incremental_original(n, edges, start_k, solver_type='cadical195', timeout_seconds=600):
    """
    Run incremental optimization with ORIGINAL encoder (like incremental_bandwidth_solver.py)
    """
    print(f"\n{'='*70}")
    print(f"ORIGINAL ENCODER - INCREMENTAL OPTIMIZATION")
    print(f"{'='*70}")
    
    vpool = IDPool()
    X_vars, Y_vars = create_position_variables(n, vpool)
    
    print(f"Generating base constraints (position + distance)...")
    base_start = time.time()
    
    # Position constraints
    position_clauses = list(encode_all_position_constraints(n, X_vars, Y_vars, vpool))
    
    # Distance constraints
    Tx_vars_dict = {}
    Ty_vars_dict = {}
    distance_clauses = []
    
    for i, (u, v) in enumerate(edges):
        if (i + 1) % 10 == 0:
            print(f"  Edge {i+1}/{len(edges)}...", end='\r')
        
        edge_id = f"edge_{u}_{v}"
        
        Tx_vars, Tx_clauses = encode_abs_distance_final(
            X_vars[u], X_vars[v], n, vpool, f"Tx_{u}_{v}"
        )
        distance_clauses.extend(Tx_clauses)
        Tx_vars_dict[edge_id] = Tx_vars
        
        Ty_vars, Ty_clauses = encode_abs_distance_final(
            Y_vars[u], Y_vars[v], n, vpool, f"Ty_{u}_{v}"
        )
        distance_clauses.extend(Ty_clauses)
        Ty_vars_dict[edge_id] = Ty_vars
    
    base_clauses = position_clauses + distance_clauses
    base_time = time.time() - base_start
    
    # Analyze base clause distribution
    clause_sizes = Counter(len(clause) for clause in base_clauses)
    print(f"\n  Base clauses: {len(base_clauses):,}")
    print(f"  Base encoding time: {base_time:.3f}s")
    print(f"\n  Base clause distribution:")
    for size in sorted(clause_sizes.keys()):
        count = clause_sizes[size]
        percentage = (count / len(base_clauses)) * 100
        print(f"    {size}-literal: {count:8,} ({percentage:5.1f}%)")
    
    total_literals = sum(len(c) for c in base_clauses)
    avg_literals = total_literals / len(base_clauses) if base_clauses else 0
    print(f"  Total literals: {total_literals:,}")
    print(f"  Avg literals/clause: {avg_literals:.2f}")
    
    # Initialize solver
    if solver_type == 'cadical195':
        solver = Cadical195()
    else:
        solver = Glucose42()
    
    for clause in base_clauses:
        solver.add_clause(clause)
    
    print(f"\n  Starting incremental optimization from K={start_k}...")
    
    # Incremental loop
    current_k = start_k
    optimal_k = None
    total_solve_time = 0.0
    solve_count = 0
    
    while current_k >= 1:
        print(f"\n  {'='*60}")
        print(f"  Testing K={current_k}")
        print(f"  {'='*60}")
        
        # Generate bandwidth constraints for this K
        bw_start = time.time()
        bandwidth_clauses = []
        
        for edge_id in Tx_vars_dict:
            Tx = Tx_vars_dict[edge_id]
            Ty = Ty_vars_dict[edge_id]
            
            # Tx <= K
            if current_k < len(Tx):
                bandwidth_clauses.append([-Tx[current_k]])
            
            # Ty <= K
            if current_k < len(Ty):
                bandwidth_clauses.append([-Ty[current_k]])
            
            # Tx >= i → Ty <= K-i
            for i in range(1, current_k + 1):
                if current_k - i >= 0:
                    if i-1 < len(Tx) and current_k-i < len(Ty):
                        bandwidth_clauses.append([-Tx[i-1], -Ty[current_k-i]])
        
        bw_time = time.time() - bw_start
        
        # Analyze bandwidth clause distribution
        bw_sizes = Counter(len(clause) for clause in bandwidth_clauses)
        print(f"  Bandwidth constraints: {len(bandwidth_clauses)} clauses ({bw_time:.3f}s)")
        bw_dist_str = ", ".join([f"{size}-lit:{bw_sizes[size]}" for size in sorted(bw_sizes.keys())])
        print(f"    Distribution: {bw_dist_str}")
        
        # Add to solver (unit clauses as assumptions, others permanent)
        assumptions = []
        for clause in bandwidth_clauses:
            if len(clause) == 1:
                assumptions.append(clause[0])
            else:
                solver.add_clause(clause)
        
        # Solve
        if timeout_seconds:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        try:
            solve_start = time.time()
            result = solver.solve(assumptions=assumptions) if assumptions else solver.solve()
            solve_time = time.time() - solve_start
            total_solve_time += solve_time
            solve_count += 1
            
            if timeout_seconds:
                signal.alarm(0)
            
            if result:
                print(f"  Result: SAT")
                print(f"  Solve time: {solve_time:.3f}s")
                optimal_k = current_k
                current_k -= 1
            else:
                print(f"  Result: UNSAT")
                print(f"  Solve time: {solve_time:.3f}s")
                print(f"  → Optimal bandwidth: {optimal_k}")
                break
                
        except TimeoutException:
            if timeout_seconds:
                signal.alarm(0)
            print(f"  Result: TIMEOUT after {timeout_seconds}s")
            break
    
    solver.delete()
    
    print(f"\n  {'='*60}")
    print(f"  ORIGINAL ENCODER SUMMARY")
    print(f"  {'='*60}")
    print(f"  Base encoding time: {base_time:.3f}s")
    print(f"  Total solve time: {total_solve_time:.3f}s ({solve_count} calls)")
    print(f"  Total time: {base_time + total_solve_time:.3f}s")
    print(f"  Optimal bandwidth: {optimal_k if optimal_k else 'NOT FOUND'}")
    
    return optimal_k, base_time, total_solve_time


def run_incremental_cutoff(n, edges, start_k, solver_type='cadical195', timeout_seconds=600):
    """
    Run incremental optimization with CUTOFF encoder (like incremental_bandwidth_solver_cutoff.py)
    """
    print(f"\n{'='*70}")
    print(f"CUTOFF ENCODER - INCREMENTAL OPTIMIZATION")
    print(f"{'='*70}")
    
    UB = calculate_theoretical_upper_bound(n)
    print(f"Using UB cutoff: {UB}")
    
    vpool = IDPool()
    X_vars, Y_vars = create_position_variables(n, vpool)
    
    print(f"Generating base constraints (position + distance with cutoff)...")
    base_start = time.time()
    
    # Position constraints
    position_clauses = list(encode_all_position_constraints(n, X_vars, Y_vars, vpool))
    
    # Distance constraints with cutoff
    Tx_vars_dict = {}
    Ty_vars_dict = {}
    distance_clauses = []
    
    for i, (u, v) in enumerate(edges):
        if (i + 1) % 10 == 0:
            print(f"  Edge {i+1}/{len(edges)}...", end='\r')
        
        edge_id = f"edge_{u}_{v}"
        
        Tx_clauses, Tx_vars = encode_abs_distance_cutoff(
            X_vars[u], X_vars[v], UB, vpool, f"Tx_{u}_{v}"
        )
        distance_clauses.extend(Tx_clauses)
        Tx_vars_dict[edge_id] = Tx_vars
        
        Ty_clauses, Ty_vars = encode_abs_distance_cutoff(
            Y_vars[u], Y_vars[v], UB, vpool, f"Ty_{u}_{v}"
        )
        distance_clauses.extend(Ty_clauses)
        Ty_vars_dict[edge_id] = Ty_vars
    
    base_clauses = position_clauses + distance_clauses
    base_time = time.time() - base_start
    
    # Analyze base clause distribution
    clause_sizes = Counter(len(clause) for clause in base_clauses)
    print(f"\n  Base clauses: {len(base_clauses):,}")
    print(f"  Base encoding time: {base_time:.3f}s")
    print(f"\n  Base clause distribution:")
    for size in sorted(clause_sizes.keys()):
        count = clause_sizes[size]
        percentage = (count / len(base_clauses)) * 100
        print(f"    {size}-literal: {count:8,} ({percentage:5.1f}%)")
    
    total_literals = sum(len(c) for c in base_clauses)
    avg_literals = total_literals / len(base_clauses) if base_clauses else 0
    print(f"  Total literals: {total_literals:,}")
    print(f"  Avg literals/clause: {avg_literals:.2f}")
    
    # Initialize solver
    if solver_type == 'cadical195':
        solver = Cadical195()
    else:
        solver = Glucose42()
    
    for clause in base_clauses:
        solver.add_clause(clause)
    
    print(f"\n  Starting incremental optimization from K={start_k}...")
    
    # Incremental loop
    current_k = min(start_k, UB)
    optimal_k = None
    total_solve_time = 0.0
    solve_count = 0
    
    while current_k >= 1:
        print(f"\n  {'='*60}")
        print(f"  Testing K={current_k}")
        print(f"  {'='*60}")
        
        # Generate bandwidth constraints for this K
        bw_start = time.time()
        bandwidth_clauses = []
        
        for edge_id in Tx_vars_dict:
            Tx = Tx_vars_dict[edge_id]  # Dict
            Ty = Ty_vars_dict[edge_id]  # Dict
            
            # Tx <= K (i.e., not Tx >= K+1)
            if current_k + 1 in Tx:
                bandwidth_clauses.append([-Tx[current_k + 1]])
            
            # Ty <= K
            if current_k + 1 in Ty:
                bandwidth_clauses.append([-Ty[current_k + 1]])
            
            # Tx >= i → Ty <= K-i
            for i in range(1, current_k + 1):
                k_minus_i = current_k - i
                if k_minus_i >= 0:
                    if i in Tx and (k_minus_i + 1) in Ty:
                        bandwidth_clauses.append([-Tx[i], -Ty[k_minus_i + 1]])
        
        bw_time = time.time() - bw_start
        print(f"  Bandwidth constraints: {len(bandwidth_clauses)} clauses ({bw_time:.3f}s)")
        
        # Add to solver
        assumptions = []
        for clause in bandwidth_clauses:
            if len(clause) == 1:
                assumptions.append(clause[0])
            else:
                solver.add_clause(clause)
        
        # Solve
        if timeout_seconds:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        try:
            solve_start = time.time()
            result = solver.solve(assumptions=assumptions) if assumptions else solver.solve()
            solve_time = time.time() - solve_start
            total_solve_time += solve_time
            solve_count += 1
            
            if timeout_seconds:
                signal.alarm(0)
            
            if result:
                print(f"  Result: SAT")
                print(f"  Solve time: {solve_time:.3f}s")
                optimal_k = current_k
                current_k -= 1
            else:
                print(f"  Result: UNSAT")
                print(f"  Solve time: {solve_time:.3f}s")
                print(f"  → Optimal bandwidth: {optimal_k}")
                break
                
        except TimeoutException:
            if timeout_seconds:
                signal.alarm(0)
            print(f"  Result: TIMEOUT after {timeout_seconds}s")
            break
    
    solver.delete()
    
    print(f"\n  {'='*60}")
    print(f"  CUTOFF ENCODER SUMMARY")
    print(f"  {'='*60}")
    print(f"  Base encoding time: {base_time:.3f}s")
    print(f"  Total solve time: {total_solve_time:.3f}s ({solve_count} calls)")
    print(f"  Total time: {base_time + total_solve_time:.3f}s")
    print(f"  Optimal bandwidth: {optimal_k if optimal_k else 'NOT FOUND'}")
    
    return optimal_k, base_time, total_solve_time


def main():
    parser = argparse.ArgumentParser(
        description='Real-time MTX analysis with proper incremental optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('mtx_file', help='Path to MTX file')
    parser.add_argument('--solver', choices=['glucose42', 'cadical195'],
                        default='cadical195',
                        help='SAT solver (default: cadical195)')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Timeout per solve call in seconds (default: 600)')
    parser.add_argument('--encoder', choices=['original', 'cutoff', 'both'],
                        default='both',
                        help='Which encoder to test (default: both)')
    
    args = parser.parse_args()
    
    mtx_file = args.mtx_file
    solver_type = args.solver
    timeout_seconds = args.timeout
    encoder_choice = args.encoder
    
    print(f"\n{'='*80}")
    print(f"REAL-TIME MTX INCREMENTAL OPTIMIZATION ANALYSIS")
    print(f"{'='*80}")
    print(f"File: {mtx_file}")
    print(f"Solver: {solver_type.upper()}")
    print(f"Timeout per solve: {timeout_seconds}s")
    print(f"Testing: {encoder_choice.upper()}")
    print(f"{'='*80}")
    
    # Search for file
    if not os.path.exists(mtx_file):
        search_paths = [
            mtx_file,
            f"mtx/{mtx_file}",
            f"mtx/group 1/{mtx_file}",
            f"mtx/group 2/{mtx_file}",
            f"mtx/group 3/{mtx_file}",
            f"mtx/regular/{mtx_file}",
        ]
        
        found = None
        for path in search_paths:
            if os.path.exists(path):
                found = path
                break
        
        if found is None:
            print(f"Error: File {mtx_file} not found")
            sys.exit(1)
        
        mtx_file = found
        print(f"Found file: {mtx_file}")
    
    # Parse
    n, edges = parse_mtx_file(mtx_file)
    if n is None or edges is None:
        print("Failed to parse MTX file")
        sys.exit(1)
    
    UB = calculate_theoretical_upper_bound(n)
    
    # Run tests
    results = {}
    
    if encoder_choice in ['original', 'both']:
        opt_orig, enc_orig, solve_orig = run_incremental_original(
            n, edges, UB, solver_type, timeout_seconds
        )
        results['original'] = (opt_orig, enc_orig, solve_orig)
    
    if encoder_choice in ['cutoff', 'both']:
        opt_cutoff, enc_cutoff, solve_cutoff = run_incremental_cutoff(
            n, edges, UB, solver_type, timeout_seconds
        )
        results['cutoff'] = (opt_cutoff, enc_cutoff, solve_cutoff)
    
    # Final comparison
    if encoder_choice == 'both':
        print(f"\n{'='*80}")
        print(f"FINAL COMPARISON")
        print(f"{'='*80}")
        
        opt_orig, enc_orig, solve_orig = results['original']
        opt_cutoff, enc_cutoff, solve_cutoff = results['cutoff']
        
        print(f"\n{'Metric':<30} {'Original':>15} {'Cutoff':>15} {'Winner':>15}")
        print(f"{'-'*80}")
        print(f"{'Optimal bandwidth':<30} {opt_orig if opt_orig else 'N/A':>15} {opt_cutoff if opt_cutoff else 'N/A':>15}")
        print(f"{'Encoding time (s)':<30} {enc_orig:>15.3f} {enc_cutoff:>15.3f} "
              f"{'Cutoff' if enc_cutoff < enc_orig else 'Original':>15}")
        print(f"{'Total solve time (s)':<30} {solve_orig:>15.3f} {solve_cutoff:>15.3f} "
              f"{'Cutoff' if solve_cutoff < solve_orig else 'Original':>15}")
        print(f"{'Total time (s)':<30} {enc_orig+solve_orig:>15.3f} {enc_cutoff+solve_cutoff:>15.3f} "
              f"{'Cutoff' if enc_cutoff+solve_cutoff < enc_orig+solve_orig else 'Original':>15}")
        
        if solve_orig > 0 and solve_cutoff > 0:
            speedup = solve_orig / solve_cutoff
            print(f"\nSolve speedup: {speedup:.2f}x ({'Cutoff' if speedup > 1 else 'Original'} faster)")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
