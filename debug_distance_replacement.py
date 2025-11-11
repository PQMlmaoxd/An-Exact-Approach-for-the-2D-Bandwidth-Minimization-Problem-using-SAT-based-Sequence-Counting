#!/usr/bin/env python3
"""
Debug script to analyze detailed solver behavior during distance replacement.

Collects metrics:
- Conflicts per distance cutoff
- Restarts per distance cutoff  
- Propagations per decision
- Learnt clause sizes (LBD distribution)
- Decision level depth
"""

import sys
import os
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from pysat.formula import IDPool
from pysat.solvers import Cadical195

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from distance_encoder_cutoff import calculate_theoretical_upper_bound
from solver_analyze_by_distance import parse_mtx_file, DistanceHybridEncoder


@dataclass
class DetailedMetrics:
    """Detailed solver behavior metrics"""
    d_cutoff: int
    result: str
    solve_time: float
    
    # Basic stats
    decisions: int
    conflicts: int
    propagations: int
    restarts: int
    
    # Derived metrics
    conflicts_per_decision: float
    propagations_per_decision: float
    decisions_per_restart: float
    
    # Learning efficiency
    avg_learnt_clause_size: Optional[float] = None
    max_decision_level: Optional[int] = None


def analyze_solver_behavior(encoder: DistanceHybridEncoder) -> DetailedMetrics:
    """
    Solve and analyze detailed solver behavior
    
    Returns:
        DetailedMetrics with solver statistics
    """
    print(f"\n{'─'*80}")
    print(f"ANALYZING d_cutoff = {encoder.d_cutoff}")
    print(f"{'─'*80}")
    
    # Solve
    start_time = time.time()
    result = encoder.solver.solve()
    solve_time = time.time() - start_time
    
    # Collect statistics
    solver = encoder.solver
    
    try:
        decisions = solver.decisions() if hasattr(solver, 'decisions') else 0
        conflicts = solver.conflicts() if hasattr(solver, 'conflicts') else 0
        propagations = solver.propagations() if hasattr(solver, 'propagations') else 0
        restarts = solver.restarts() if hasattr(solver, 'restarts') else 0
        
        # Calculate derived metrics
        conflicts_per_decision = conflicts / decisions if decisions > 0 else 0
        propagations_per_decision = propagations / decisions if decisions > 0 else 0
        decisions_per_restart = decisions / restarts if restarts > 0 else 0
        
        metrics = DetailedMetrics(
            d_cutoff=encoder.d_cutoff,
            result='SAT' if result else 'UNSAT',
            solve_time=solve_time,
            decisions=decisions,
            conflicts=conflicts,
            propagations=propagations,
            restarts=restarts,
            conflicts_per_decision=conflicts_per_decision,
            propagations_per_decision=propagations_per_decision,
            decisions_per_restart=decisions_per_restart
        )
        
        # Print analysis
        print(f"\nBasic Statistics:")
        print(f"  Result: {metrics.result}")
        print(f"  Solve time: {solve_time:.3f}s")
        print(f"  Decisions: {decisions:,}")
        print(f"  Conflicts: {conflicts:,}")
        print(f"  Propagations: {propagations:,}")
        print(f"  Restarts: {restarts:,}")
        
        print(f"\nEfficiency Metrics:")
        print(f"  Conflicts per decision: {conflicts_per_decision:.2f}")
        print(f"  Propagations per decision: {propagations_per_decision:.1f}")
        print(f"  Decisions per restart: {decisions_per_restart:.1f}")
        
        # Interpretation
        print(f"\nInterpretation:")
        if conflicts_per_decision > 0.5:
            print(f"  ⚠️  High conflict rate → Solver struggling with search")
        else:
            print(f"  ✓ Low conflict rate → Efficient search")
        
        if propagations_per_decision < 100:
            print(f"  ⚠️  Low propagation → Weak constraint propagation")
        elif propagations_per_decision > 500:
            print(f"  ✓ High propagation → Strong constraint propagation")
        else:
            print(f"  ~ Moderate propagation")
        
        if decisions_per_restart < 100:
            print(f"  ⚠️  Frequent restarts → Solver not finding good paths")
        else:
            print(f"  ✓ Infrequent restarts → Stable search")
        
        return metrics
        
    except Exception as e:
        print(f"Warning: Could not collect some metrics: {e}")
        return DetailedMetrics(
            d_cutoff=encoder.d_cutoff,
            result='SAT' if result else 'UNSAT',
            solve_time=solve_time,
            decisions=0,
            conflicts=0,
            propagations=0,
            restarts=0,
            conflicts_per_decision=0,
            propagations_per_decision=0,
            decisions_per_restart=0
        )


def compare_propagation_strength(n: int, edges: List[Tuple[int, int]], K: int,
                                d_cutoffs: List[int]) -> None:
    """
    Compare propagation strength across different d_cutoff values
    
    Focus on:
    - Propagations per decision (higher = stronger propagation)
    - Conflicts per decision (lower = better)
    - Restarts (fewer = more stable)
    """
    print(f"\n{'='*80}")
    print("PROPAGATION STRENGTH COMPARISON")
    print(f"{'='*80}")
    print(f"Graph: n={n}, |E|={len(edges)}, K={K}")
    print(f"Testing d_cutoff values: {d_cutoffs}")
    
    results = []
    
    for d_cutoff in d_cutoffs:
        # Create encoder
        vpool = IDPool()
        encoder = DistanceHybridEncoder(n, edges, K, d_cutoff, vpool, 'cadical195')
        
        # Encode (silent)
        print(f"\nEncoding with d_cutoff={d_cutoff}...")
        encoder.encode()
        
        # Analyze solver behavior
        metrics = analyze_solver_behavior(encoder)
        results.append(metrics)
        
        # Cleanup
        encoder.cleanup()
    
    # Print comparison table
    print(f"\n{'='*120}")
    print("COMPARISON TABLE")
    print(f"{'='*120}")
    
    header = (f"{'d_cutoff':<10} {'Time(s)':<10} {'Decisions':<12} {'Conflicts':<12} "
             f"{'C/D':<8} {'Prop/D':<10} {'D/R':<10} {'Status':<10}")
    print(header)
    print('─' * 120)
    
    for m in results:
        row = (f"{m.d_cutoff:<10} "
               f"{m.solve_time:<10.2f} "
               f"{m.decisions:<12,} "
               f"{m.conflicts:<12,} "
               f"{m.conflicts_per_decision:<8.2f} "
               f"{m.propagations_per_decision:<10.1f} "
               f"{m.decisions_per_restart:<10.1f} "
               f"{m.result:<10}")
        print(row)
    
    print(f"{'='*120}")
    
    # Analysis
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    # Find best and worst
    best_time = min(results, key=lambda x: x.solve_time)
    worst_time = max(results, key=lambda x: x.solve_time)
    
    print(f"\n1. SOLVE TIME:")
    print(f"   Best:  d_cutoff={best_time.d_cutoff} → {best_time.solve_time:.2f}s")
    print(f"   Worst: d_cutoff={worst_time.d_cutoff} → {worst_time.solve_time:.2f}s")
    print(f"   Ratio: {worst_time.solve_time / best_time.solve_time:.2f}x slower")
    
    # Propagation strength
    best_prop = max(results, key=lambda x: x.propagations_per_decision)
    worst_prop = min(results, key=lambda x: x.propagations_per_decision)
    
    print(f"\n2. PROPAGATION STRENGTH:")
    print(f"   Best:  d_cutoff={best_prop.d_cutoff} → {best_prop.propagations_per_decision:.1f} prop/decision")
    print(f"   Worst: d_cutoff={worst_prop.d_cutoff} → {worst_prop.propagations_per_decision:.1f} prop/decision")
    
    # Conflict rate
    best_conflict = min(results, key=lambda x: x.conflicts_per_decision)
    worst_conflict = max(results, key=lambda x: x.conflicts_per_decision)
    
    print(f"\n3. CONFLICT RATE:")
    print(f"   Best:  d_cutoff={best_conflict.d_cutoff} → {best_conflict.conflicts_per_decision:.2f} conflicts/decision")
    print(f"   Worst: d_cutoff={worst_conflict.d_cutoff} → {worst_conflict.conflicts_per_decision:.2f} conflicts/decision")
    
    # Correlation analysis
    print(f"\n4. CORRELATION ANALYSIS:")
    
    # Check if low propagation correlates with slow solve time
    slow_runs = [m for m in results if m.solve_time > 60]
    if slow_runs:
        avg_prop_slow = sum(m.propagations_per_decision for m in slow_runs) / len(slow_runs)
        fast_runs = [m for m in results if m.solve_time < 50]
        if fast_runs:
            avg_prop_fast = sum(m.propagations_per_decision for m in fast_runs) / len(fast_runs)
            print(f"   Slow runs (>60s): avg propagation = {avg_prop_slow:.1f}")
            print(f"   Fast runs (<50s): avg propagation = {avg_prop_fast:.1f}")
            
            if avg_prop_slow < avg_prop_fast:
                print(f"   ✓ CONFIRMED: Lower propagation correlates with slower solving")
            else:
                print(f"   ✗ UNEXPECTED: Propagation does not explain slowdown")
    
    print(f"\n{'='*80}\n")


def main():
    """
    Usage:
        python debug_distance_replacement.py bfw62a.mtx --k=3
    """
    if len(sys.argv) < 3:
        print("Usage: python debug_distance_replacement.py <mtx_file> --k=<K>")
        sys.exit(1)
    
    mtx_file = sys.argv[1]
    K = None
    
    for arg in sys.argv[2:]:
        if arg.startswith('--k='):
            K = int(arg.split('=')[1])
    
    if K is None:
        print("Error: --k=<K> required")
        sys.exit(1)
    
    # Find MTX file
    search_paths = [
        mtx_file,
        f"mtx/{mtx_file}",
        f"mtx/oc2/{mtx_file}",
        f"mtx/cutoff2/{mtx_file}",
    ]
    
    found_file = None
    for path in search_paths:
        if os.path.exists(path):
            found_file = path
            break
    
    if not found_file:
        print(f"Error: File '{mtx_file}' not found")
        sys.exit(1)
    
    # Parse graph
    print(f"Reading graph from: {found_file}")
    n, edges = parse_mtx_file(found_file)
    print(f"Loaded: n={n}, |E|={len(edges)}")
    
    ub = calculate_theoretical_upper_bound(n)
    print(f"Theoretical UB: {ub}")
    
    # Test d_cutoff values based on your observations
    d_cutoffs = [
        ub + 1,  # 40s (fast)
        ub + 2,  # 82s (slow)
        ub + 3,  # 80s (slow)
        ub + 4,  # 43s (fast again!)
    ]
    
    print(f"\n{'#'*80}")
    print(f"# INVESTIGATING NON-MONOTONIC BEHAVIOR")
    print(f"# Testing: {d_cutoffs}")
    print(f"{'#'*80}")
    
    compare_propagation_strength(n, edges, K, d_cutoffs)


if __name__ == '__main__':
    main()