# analyze_mtx_dataset.py
"""
Wrapper to analyze real MTX datasets using clause distribution and solver instrumentation tools.

Usage:
    python analyze_mtx_dataset.py <mtx_file> [options]
    
Options:
    --mode MODE          Analysis mode: distribution, performance, or both (default: distribution)
    --solver SOLVER      SAT solver: glucose42 or cadical195 (default: cadical195)
    --timeout SECONDS    Timeout for solver in seconds (default: 300 for performance mode)
    --no-timeout         Disable timeout
    
Examples:
    # Quick distribution analysis (recommended for large graphs)
    python analyze_mtx_dataset.py large.mtx
    
    # With specific solver
    python analyze_mtx_dataset.py file.mtx --solver cadical195
    
    # Performance test with timeout
    python analyze_mtx_dataset.py file.mtx --mode performance --timeout 600 --solver cadical195
    
    # Both modes with no timeout
    python analyze_mtx_dataset.py file.mtx --mode both --no-timeout
"""

import sys
import os
import time
import signal
import argparse
from collections import Counter
from pysat.formula import IDPool
from pysat.solvers import Glucose42, Cadical195

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distance_encoder import encode_abs_distance_final
from distance_encoder_cutoff import encode_abs_distance_cutoff, calculate_theoretical_upper_bound
from position_constraints import encode_all_position_constraints, create_position_variables


def parse_mtx_file(filename):
    """
    Parse MTX file and return n, edges
    
    Handles MatrixMarket format with proper error handling
    """
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
        
        # Parse header
        if not header_found:
            try:
                parts = line.split()
                if len(parts) >= 2:
                    n = int(parts[0])
                    m = int(parts[1]) if len(parts) > 1 else 0
                    print(f"Graph: {n} vertices, {m} edges (from header)")
                    header_found = True
                    continue
            except ValueError:
                continue
        
        # Parse edge
        try:
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                
                # Skip self-loops
                if u != v:
                    edges_set.add((min(u, v), max(u, v)))
        except ValueError:
            continue
    
    edges = list(edges_set)
    print(f"Loaded: {n} vertices, {len(edges)} edges")
    
    return n, edges


def analyze_full_graph_clauses(n, edges):
    """
    Analyze clause distribution for FULL graph encoding
    
    This generates clauses for ALL edges, not just one edge
    """
    
    UB = calculate_theoretical_upper_bound(n)
    
    print(f"\n{'='*80}")
    print(f"FULL GRAPH CLAUSE DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    print(f"Vertices: {n}")
    print(f"Edges: {len(edges)}")
    print(f"Theoretical UB: {UB}")
    
    # ===== ORIGINAL ENCODER =====
    print(f"\n{'='*70}")
    print(f"GENERATING CLAUSES: ORIGINAL ENCODER")
    print(f"{'='*70}")
    
    vpool_orig = IDPool()
    
    # Create position variables for all vertices
    print(f"Creating position variables for {n} vertices...")
    X_vars_orig, Y_vars_orig = create_position_variables(n, vpool_orig)
    
    # Generate position constraints
    print(f"Generating position constraints...")
    position_start = time.time()
    position_clauses_orig = list(encode_all_position_constraints(n, X_vars_orig, Y_vars_orig, vpool_orig))
    position_time = time.time() - position_start
    print(f"  Position clauses: {len(position_clauses_orig)} ({position_time:.3f}s)")
    
    # Generate distance constraints for ALL edges
    print(f"Generating distance constraints for {len(edges)} edges...")
    distance_start = time.time()
    all_dist_clauses_orig = []
    
    for i, (u, v) in enumerate(edges):
        if (i + 1) % 10 == 0 or (i + 1) == len(edges):
            print(f"  Processing edge {i+1}/{len(edges)}...", end='\r')
        
        # X distance
        Tx_vars, Tx_clauses = encode_abs_distance_final(
            X_vars_orig[u], X_vars_orig[v], n, vpool_orig, f"Tx_{u}_{v}"
        )
        all_dist_clauses_orig.extend(Tx_clauses)
        
        # Y distance
        Ty_vars, Ty_clauses = encode_abs_distance_final(
            Y_vars_orig[u], Y_vars_orig[v], n, vpool_orig, f"Ty_{u}_{v}"
        )
        all_dist_clauses_orig.extend(Ty_clauses)
    
    distance_time = time.time() - distance_start
    print(f"\n  Distance clauses: {len(all_dist_clauses_orig)} ({distance_time:.3f}s)")
    
    all_clauses_orig = position_clauses_orig + all_dist_clauses_orig
    
    # Analyze distribution
    print(f"\nAnalyzing clause distribution...")
    size_counter_orig = Counter(len(clause) for clause in all_clauses_orig)
    
    print(f"\n{'='*70}")
    print(f"ORIGINAL ENCODER STATISTICS")
    print(f"{'='*70}")
    print(f"Total clauses: {len(all_clauses_orig):,}")
    print(f"Total literals: {sum(len(c) for c in all_clauses_orig):,}")
    print(f"Avg literals/clause: {sum(len(c) for c in all_clauses_orig) / len(all_clauses_orig):.2f}")
    
    print(f"\nClause size distribution:")
    for size in sorted(size_counter_orig.keys()):
        count = size_counter_orig[size]
        percentage = (count / len(all_clauses_orig)) * 100
        print(f"  {size}-literal: {count:8,} ({percentage:5.1f}%)")
    
    # ===== CUTOFF ENCODER =====
    print(f"\n{'='*70}")
    print(f"GENERATING CLAUSES: CUTOFF ENCODER")
    print(f"{'='*70}")
    
    vpool_cutoff = IDPool()
    
    # Create position variables for all vertices
    print(f"Creating position variables for {n} vertices...")
    X_vars_cutoff, Y_vars_cutoff = create_position_variables(n, vpool_cutoff)
    
    # Generate position constraints
    print(f"Generating position constraints...")
    position_start = time.time()
    position_clauses_cutoff = list(encode_all_position_constraints(n, X_vars_cutoff, Y_vars_cutoff, vpool_cutoff))
    position_time = time.time() - position_start
    print(f"  Position clauses: {len(position_clauses_cutoff)} ({position_time:.3f}s)")
    
    # Generate distance constraints for ALL edges with UB cutoff
    print(f"Generating distance constraints with UB={UB} for {len(edges)} edges...")
    distance_start = time.time()
    all_dist_clauses_cutoff = []
    
    for i, (u, v) in enumerate(edges):
        if (i + 1) % 10 == 0 or (i + 1) == len(edges):
            print(f"  Processing edge {i+1}/{len(edges)}...", end='\r')
        
        # X distance with cutoff
        Tx_clauses, Tx_vars = encode_abs_distance_cutoff(
            X_vars_cutoff[u], X_vars_cutoff[v], UB, vpool_cutoff, f"Tx_{u}_{v}"
        )
        all_dist_clauses_cutoff.extend(Tx_clauses)
        
        # Y distance with cutoff
        Ty_clauses, Ty_vars = encode_abs_distance_cutoff(
            Y_vars_cutoff[u], Y_vars_cutoff[v], UB, vpool_cutoff, f"Ty_{u}_{v}"
        )
        all_dist_clauses_cutoff.extend(Ty_clauses)
    
    distance_time = time.time() - distance_start
    print(f"\n  Distance clauses: {len(all_dist_clauses_cutoff)} ({distance_time:.3f}s)")
    
    all_clauses_cutoff = position_clauses_cutoff + all_dist_clauses_cutoff
    
    # Analyze distribution
    print(f"\nAnalyzing clause distribution...")
    size_counter_cutoff = Counter(len(clause) for clause in all_clauses_cutoff)
    
    print(f"\n{'='*70}")
    print(f"CUTOFF ENCODER STATISTICS")
    print(f"{'='*70}")
    print(f"Total clauses: {len(all_clauses_cutoff):,}")
    print(f"Total literals: {sum(len(c) for c in all_clauses_cutoff):,}")
    print(f"Avg literals/clause: {sum(len(c) for c in all_clauses_cutoff) / len(all_clauses_cutoff):.2f}")
    
    print(f"\nClause size distribution:")
    for size in sorted(size_counter_cutoff.keys()):
        count = size_counter_cutoff[size]
        percentage = (count / len(all_clauses_cutoff)) * 100
        print(f"  {size}-literal: {count:8,} ({percentage:5.1f}%)")
    
    # ===== COMPARISON =====
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<30} {'Original':>15} {'Cutoff':>15} {'Difference':>15}")
    print(f"{'-'*80}")
    
    # Total clauses
    diff_clauses = len(all_clauses_orig) - len(all_clauses_cutoff)
    reduction_pct = (diff_clauses / len(all_clauses_orig) * 100) if len(all_clauses_orig) > 0 else 0
    print(f"{'Total clauses':<30} {len(all_clauses_orig):>15,} {len(all_clauses_cutoff):>15,} "
          f"{diff_clauses:>15,} ({reduction_pct:+.1f}%)")
    
    # 2-literal clauses
    orig_2lit = size_counter_orig.get(2, 0)
    cutoff_2lit = size_counter_cutoff.get(2, 0)
    diff_2lit = orig_2lit - cutoff_2lit
    print(f"{'2-literal clauses':<30} {orig_2lit:>15,} {cutoff_2lit:>15,} {diff_2lit:>15,}")
    
    # 3-literal clauses
    orig_3lit = size_counter_orig.get(3, 0)
    cutoff_3lit = size_counter_cutoff.get(3, 0)
    diff_3lit = orig_3lit - cutoff_3lit
    print(f"{'3-literal clauses':<30} {orig_3lit:>15,} {cutoff_3lit:>15,} {diff_3lit:>15,}")
    
    # Ratios
    orig_ratio = f"{orig_3lit}:{orig_2lit}" if orig_2lit > 0 else "N/A"
    cutoff_ratio = f"{cutoff_3lit}:{cutoff_2lit}" if cutoff_2lit > 0 else "N/A"
    print(f"{'3-lit:2-lit ratio':<30} {orig_ratio:>15} {cutoff_ratio:>15}")
    
    # Key insight
    print(f"\n{'='*80}")
    print(f"KEY INSIGHTS")
    print(f"{'='*80}")
    
    if orig_3lit > orig_2lit * 2:
        print(f"\n✓ ORIGINAL: Heavily 3-literal dominant ({orig_3lit/orig_2lit:.1f}:1)")
        print(f"  → Optimized for modern SAT solvers")
        print(f"  → Better cache locality and heuristics")
    
    if cutoff_2lit > cutoff_3lit:
        print(f"\n⚠️  CUTOFF: 2-literal dominant ({cutoff_2lit/cutoff_3lit:.1f}:1)")
        print(f"  → May over-constrain the problem")
        print(f"  → Non-uniform clause distribution")
        print(f"  → Consider adding deactivation clauses")
    
    if reduction_pct > 10:
        print(f"\n✓ Cutoff reduces clauses by {reduction_pct:.1f}%")
        print(f"  → Faster encoding time")
        print(f"  → BUT: May have slower solve time due to clause mix")
    
    return {
        'orig_clauses': len(all_clauses_orig),
        'cutoff_clauses': len(all_clauses_cutoff),
        'orig_2lit': orig_2lit,
        'orig_3lit': orig_3lit,
        'cutoff_2lit': cutoff_2lit,
        'cutoff_3lit': cutoff_3lit
    }


class TimeoutException(Exception):
    """Exception raised when solver times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Solver timeout!")


def performance_comparison(n, edges, K_test=None, solver_type='cadical195', timeout_seconds=300):
    """
    Compare actual solver performance between encoders with INCREMENTAL optimization
    
    Args:
        n: Number of vertices
        edges: List of edges
        K_test: Starting K value (default: theoretical UB)
        solver_type: 'glucose42' or 'cadical195' (default: cadical195)
        timeout_seconds: Timeout in seconds (default: 300, None = no timeout)
    
    WARNING: This actually solves the SAT problem incrementally like real solvers!
    """
    
    UB = calculate_theoretical_upper_bound(n)
    
    if K_test is None:
        K_test = UB  # Start from theoretical upper bound
    
    print(f"\n{'='*80}")
    print(f"SOLVER PERFORMANCE COMPARISON - INCREMENTAL OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Vertices: {n}")
    print(f"Edges: {len(edges)}")
    print(f"Theoretical UB: {UB}")
    print(f"Starting from K = {K_test}, optimizing down to find minimum bandwidth")
    print(f"Solver: {solver_type.upper()}")
    if timeout_seconds:
        print(f"Timeout: {timeout_seconds}s")
    else:
        print(f"Timeout: DISABLED")
    print(f"\n⚠️  WARNING: This will actually solve the SAT problem INCREMENTALLY!")
    print(f"⚠️  This may take significant time for large graphs!")
    
    # ===== ORIGINAL ENCODER =====
    print(f"\n{'='*70}")
    print(f"TESTING: ORIGINAL ENCODER")
    print(f"{'='*70}")
    
    vpool_orig = IDPool()
    X_vars_orig, Y_vars_orig = create_position_variables(n, vpool_orig)
    
    print(f"Generating clauses...")
    start_time = time.time()
    
    # Position constraints
    position_clauses = list(encode_all_position_constraints(n, X_vars_orig, Y_vars_orig, vpool_orig))
    
    # Distance constraints - store T variables for bandwidth constraints
    distance_clauses = []
    Tx_vars_dict = {}  # Store T variables for each edge
    Ty_vars_dict = {}
    
    for i, (u, v) in enumerate(edges):
        if (i + 1) % 10 == 0:
            print(f"  Edge {i+1}/{len(edges)}...", end='\r')
        
        edge_id = f"edge_{u}_{v}"
        
        Tx_vars, Tx_clauses = encode_abs_distance_final(
            X_vars_orig[u], X_vars_orig[v], n, vpool_orig, f"Tx_{u}_{v}"
        )
        distance_clauses.extend(Tx_clauses)
        Tx_vars_dict[edge_id] = Tx_vars
        
        Ty_vars, Ty_clauses = encode_abs_distance_final(
            Y_vars_orig[u], Y_vars_orig[v], n, vpool_orig, f"Ty_{u}_{v}"
        )
        distance_clauses.extend(Ty_clauses)
        Ty_vars_dict[edge_id] = Ty_vars
    
    # Add bandwidth ≤ K constraints
    print(f"\n  Adding bandwidth ≤ {K_test} constraints...")
    bandwidth_clauses = []
    for edge_id in Tx_vars_dict:
        Tx = Tx_vars_dict[edge_id]
        Ty = Ty_vars_dict[edge_id]
        
        # Tx <= K (i.e., not Tx >= K+1)
        if K_test < len(Tx):
            bandwidth_clauses.append([-Tx[K_test]])
        
        # Ty <= K (i.e., not Ty >= K+1)
        if K_test < len(Ty):
            bandwidth_clauses.append([-Ty[K_test]])
        
        # Implication: Tx >= i → Ty <= K-i
        for i in range(1, K_test + 1):
            if K_test - i >= 0:
                tx_geq_i = None
                ty_leq_ki = None
                
                if i-1 < len(Tx):
                    tx_geq_i = Tx[i-1]  # Tx >= i
                
                if K_test-i < len(Ty):
                    ty_leq_ki = -Ty[K_test-i]  # Ty <= K-i (negated)
                
                if tx_geq_i is not None and ty_leq_ki is not None:
                    bandwidth_clauses.append([-tx_geq_i, ty_leq_ki])
    
    print(f"  Generated {len(bandwidth_clauses)} bandwidth constraint clauses")
    
    all_clauses = position_clauses + distance_clauses + bandwidth_clauses
    encode_time_orig = time.time() - start_time
    
    print(f"\n  Encoding time: {encode_time_orig:.3f}s")
    print(f"  Total clauses: {len(all_clauses):,}")
    
    # Solve
    print(f"\nSolving with {solver_type.upper()}...")
    if solver_type == 'cadical195':
        solver = Cadical195()
    else:
        solver = Glucose42()
    
    for clause in all_clauses:
        solver.add_clause(clause)
    
    # Set up timeout if specified
    if timeout_seconds:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    
    try:
        solve_start = time.time()
        result_orig = solver.solve()
        solve_time_orig = time.time() - solve_start
        
        # Cancel alarm
        if timeout_seconds:
            signal.alarm(0)
        
        print(f"  Result: {'SAT' if result_orig else 'UNSAT'}")
        print(f"  Solve time: {solve_time_orig:.3f}s")
        print(f"  Total time: {encode_time_orig + solve_time_orig:.3f}s")
        
    except TimeoutException:
        if timeout_seconds:
            signal.alarm(0)
        solve_time_orig = timeout_seconds
        result_orig = None
        print(f"  Result: TIMEOUT after {timeout_seconds}s")
        print(f"  Consider: Increase timeout or use distribution mode only")
    
    solver.delete()
    
    # ===== CUTOFF ENCODER =====
    print(f"\n{'='*70}")
    print(f"TESTING: CUTOFF ENCODER")
    print(f"{'='*70}")
    
    vpool_cutoff = IDPool()
    X_vars_cutoff, Y_vars_cutoff = create_position_variables(n, vpool_cutoff)
    
    print(f"Generating clauses with UB cutoff...")
    start_time = time.time()
    
    # Position constraints
    position_clauses = list(encode_all_position_constraints(n, X_vars_cutoff, Y_vars_cutoff, vpool_cutoff))
    
    # Distance constraints with cutoff - store T variables for bandwidth constraints
    distance_clauses = []
    Tx_vars_dict = {}  # Store T variables for each edge
    Ty_vars_dict = {}
    
    for i, (u, v) in enumerate(edges):
        if (i + 1) % 10 == 0:
            print(f"  Edge {i+1}/{len(edges)}...", end='\r')
        
        edge_id = f"edge_{u}_{v}"
        
        Tx_clauses, Tx_vars = encode_abs_distance_cutoff(
            X_vars_cutoff[u], X_vars_cutoff[v], UB, vpool_cutoff, f"Tx_{u}_{v}"
        )
        distance_clauses.extend(Tx_clauses)
        Tx_vars_dict[edge_id] = Tx_vars
        
        Ty_clauses, Ty_vars = encode_abs_distance_cutoff(
            Y_vars_cutoff[u], Y_vars_cutoff[v], UB, vpool_cutoff, f"Ty_{u}_{v}"
        )
        distance_clauses.extend(Ty_clauses)
        Ty_vars_dict[edge_id] = Ty_vars
    
    # Add bandwidth ≤ K constraints
    print(f"\n  Adding bandwidth ≤ {K_test} constraints...")
    bandwidth_clauses = []
    for edge_id in Tx_vars_dict:
        Tx = Tx_vars_dict[edge_id]  # Dict: Tx[d] = var_id for distance d
        Ty = Ty_vars_dict[edge_id]  # Dict: Ty[d] = var_id for distance d
        
        # Tx <= K (i.e., not Tx >= K+1)
        # For Dict: check if K+1 exists in dict
        if K_test + 1 in Tx:
            bandwidth_clauses.append([-Tx[K_test + 1]])
        
        # Ty <= K (i.e., not Ty >= K+1)
        if K_test + 1 in Ty:
            bandwidth_clauses.append([-Ty[K_test + 1]])
        
        # Implication: Tx >= i → Ty <= K-i
        for i in range(1, K_test + 1):
            k_minus_i = K_test - i
            if k_minus_i >= 0:
                # Check if both variables exist in dicts
                if i in Tx and (k_minus_i + 1) in Ty:
                    tx_geq_i = Tx[i]  # Tx >= i
                    ty_leq_ki = -Ty[k_minus_i + 1]  # Ty <= K-i means ¬(Ty >= K-i+1)
                    bandwidth_clauses.append([-tx_geq_i, ty_leq_ki])
    
    print(f"  Generated {len(bandwidth_clauses)} bandwidth constraint clauses")
    
    all_clauses = position_clauses + distance_clauses + bandwidth_clauses
    encode_time_cutoff = time.time() - start_time
    
    print(f"\n  Encoding time: {encode_time_cutoff:.3f}s")
    print(f"  Total clauses: {len(all_clauses):,}")
    
    # Solve
    print(f"\nSolving with {solver_type.upper()}...")
    if solver_type == 'cadical195':
        solver = Cadical195()
    else:
        solver = Glucose42()
    
    for clause in all_clauses:
        solver.add_clause(clause)
    
    # Set up timeout if specified
    if timeout_seconds:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    
    try:
        solve_start = time.time()
        result_cutoff = solver.solve()
        solve_time_cutoff = time.time() - solve_start
        
        # Cancel alarm
        if timeout_seconds:
            signal.alarm(0)
        
        print(f"  Result: {'SAT' if result_cutoff else 'UNSAT'}")
        print(f"  Solve time: {solve_time_cutoff:.3f}s")
        print(f"  Total time: {encode_time_cutoff + solve_time_cutoff:.3f}s")
        
    except TimeoutException:
        if timeout_seconds:
            signal.alarm(0)
        solve_time_cutoff = timeout_seconds
        result_cutoff = None
        print(f"  Result: TIMEOUT after {timeout_seconds}s")
        print(f"  Consider: Increase timeout or use distribution mode only")
    
    solver.delete()
    
    # ===== COMPARISON =====
    print(f"\n{'='*80}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<30} {'Original':>15} {'Cutoff':>15} {'Winner':>15}")
    print(f"{'-'*80}")
    
    print(f"{'Encoding time (s)':<30} {encode_time_orig:>15.3f} {encode_time_cutoff:>15.3f} "
          f"{'Cutoff' if encode_time_cutoff < encode_time_orig else 'Original':>15}")
    
    if result_orig is not None and result_cutoff is not None:
        print(f"{'Solve time (s)':<30} {solve_time_orig:>15.3f} {solve_time_cutoff:>15.3f} "
              f"{'Cutoff' if solve_time_cutoff < solve_time_orig else 'Original':>15}")
        
        total_orig = encode_time_orig + solve_time_orig
        total_cutoff = encode_time_cutoff + solve_time_cutoff
        print(f"{'Total time (s)':<30} {total_orig:>15.3f} {total_cutoff:>15.3f} "
              f"{'Cutoff' if total_cutoff < total_orig else 'Original':>15}")
        
        if solve_time_orig > 0 and solve_time_cutoff > 0:
            speedup = max(solve_time_orig, solve_time_cutoff) / min(solve_time_orig, solve_time_cutoff)
            winner = 'Original' if solve_time_orig < solve_time_cutoff else 'Cutoff'
            print(f"{'Solve speedup':<30} {'':<15} {'':<15} {f'{winner} {speedup:.2f}x':>15}")
    else:
        print(f"\nOne or both solvers timed out.")
        print(f"Original: {'TIMEOUT' if result_orig is None else 'OK'}")
        print(f"Cutoff: {'TIMEOUT' if result_cutoff is None else 'OK'}")
        print(f"\nRecommendation: Use distribution mode for large graphs")


def main():
    """Main entry point with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='Analyze MTX datasets and compare Original vs Cutoff encoders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick distribution analysis (recommended for large graphs)
  python analyze_mtx_dataset.py large.mtx
  
  # With specific solver
  python analyze_mtx_dataset.py file.mtx --solver cadical195
  
  # Performance test with timeout
  python analyze_mtx_dataset.py file.mtx --mode performance --timeout 600 --solver cadical195
  
  # Both modes with no timeout
  python analyze_mtx_dataset.py file.mtx --mode both --no-timeout

Recommended settings for LARGE graphs:
  python analyze_mtx_dataset.py large.mtx --mode distribution
  python analyze_mtx_dataset.py large.mtx --mode performance --timeout 600 --solver cadical195
"""
    )
    
    parser.add_argument('mtx_file', help='Path to MTX file')
    parser.add_argument('--mode', choices=['distribution', 'performance', 'both'],
                        default='distribution',
                        help='Analysis mode (default: distribution)')
    parser.add_argument('--solver', choices=['glucose42', 'cadical195'],
                        default='cadical195',
                        help='SAT solver to use (default: cadical195)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout in seconds for performance mode (default: 300)')
    parser.add_argument('--no-timeout', action='store_true',
                        help='Disable timeout')
    
    args = parser.parse_args()
    
    mtx_file = args.mtx_file
    mode = args.mode
    solver_type = args.solver
    timeout_seconds = None if args.no_timeout else args.timeout
    
    print(f"\n{'='*80}")
    print(f"MTX DATASET ANALYSIS")
    print(f"{'='*80}")
    print(f"File: {mtx_file}")
    print(f"Mode: {mode}")
    print(f"Solver: {solver_type.upper()}")
    if timeout_seconds:
        print(f"Timeout: {timeout_seconds}s")
    else:
        print(f"Timeout: DISABLED")
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
            print(f"Error: File '{mtx_file}' not found")
            print("Searched in:", search_paths)
            sys.exit(1)
        
        mtx_file = found
        print(f"Found file: {mtx_file}")
    
    # Parse MTX file
    n, edges = parse_mtx_file(mtx_file)
    if n is None or edges is None:
        print("Failed to parse MTX file")
        sys.exit(1)
    
    # Run analysis based on mode
    if mode in ['distribution', 'both']:
        stats = analyze_full_graph_clauses(n, edges)
    
    if mode in ['performance', 'both']:
        # Performance test with timeout and solver selection
        print(f"\n{'='*80}")
        print(f"PERFORMANCE TEST")
        print(f"{'='*80}")
        print(f"⚠️  This will solve the SAT problem with {solver_type.upper()}")
        if timeout_seconds:
            print(f"⚠️  Timeout set to {timeout_seconds}s")
            print(f"⚠️  If timeout occurs, consider:")
            print(f"    - Increase timeout: --timeout 1200")
            print(f"    - Use distribution mode only: --mode distribution")
        
        performance_comparison(n, edges, solver_type=solver_type, timeout_seconds=timeout_seconds)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
