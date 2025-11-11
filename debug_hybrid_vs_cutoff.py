#!/usr/bin/env python3
# debug_hybrid_vs_cutoff.py
# Debug why hybrid with full replacement is slower than cutoff

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distance_encoder_cutoff import encode_abs_distance_cutoff, calculate_theoretical_upper_bound
from distance_encoder_hybrid import encode_abs_distance_hybrid
from pysat.formula import IDPool

def analyze_clause_structure(clauses, label):
    """Analyze structure of clauses"""
    print(f"\n{label}:")
    print(f"  Total clauses: {len(clauses)}")
    
    # Group by clause length
    length_dist = {}
    for clause in clauses:
        length = len(clause)
        length_dist[length] = length_dist.get(length, 0) + 1
    
    print(f"  Clause length distribution:")
    for length in sorted(length_dist.keys()):
        count = length_dist[length]
        print(f"    Length {length}: {count} clauses")
    
    # Count binary clauses (most common for mutual exclusion)
    binary_count = length_dist.get(2, 0)
    print(f"  Binary clauses: {binary_count} ({binary_count/len(clauses)*100:.1f}%)")
    
    return {
        'total': len(clauses),
        'length_dist': length_dist,
        'binary_count': binary_count
    }


def compare_single_edge(n: int, UB: int):
    """Compare encoding for a single edge"""
    print(f"\n{'='*80}")
    print(f"SINGLE EDGE COMPARISON (n={n}, UB={UB})")
    print(f"{'='*80}")
    
    max_replacements = n - 1 - UB
    print(f"Max replacements: {max_replacements}")
    
    # CUTOFF encoding
    vpool_cutoff = IDPool()
    U_cutoff = [vpool_cutoff.id(f'U_{i}') for i in range(1, n + 1)]
    V_cutoff = [vpool_cutoff.id(f'V_{i}') for i in range(1, n + 1)]
    
    clauses_cutoff, t_vars_cutoff = encode_abs_distance_cutoff(
        U_cutoff, V_cutoff, UB, vpool_cutoff, t_var_prefix="T[cutoff]"
    )
    
    stats_cutoff = analyze_clause_structure(clauses_cutoff, "CUTOFF Encoding")
    
    # HYBRID encoding (full replacement)
    vpool_hybrid = IDPool()
    U_hybrid = [vpool_hybrid.id(f'U_{i}') for i in range(1, n + 1)]
    V_hybrid = [vpool_hybrid.id(f'V_{i}') for i in range(1, n + 1)]
    
    clauses_hybrid, t_vars_hybrid = encode_abs_distance_hybrid(
        U_hybrid, V_hybrid, n, UB, vpool_hybrid,
        prefix="T[hybrid]", num_replacements=max_replacements
    )
    
    stats_hybrid = analyze_clause_structure(clauses_hybrid, "HYBRID Encoding (full replacement)")
    
    # COMPARISON
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nT variables:")
    print(f"  Cutoff: {len(t_vars_cutoff)}")
    print(f"  Hybrid: {len(t_vars_hybrid)}")
    print(f"  Match: {'✓' if len(t_vars_cutoff) == len(t_vars_hybrid) else '✗'}")
    
    print(f"\nTotal clauses:")
    print(f"  Cutoff: {stats_cutoff['total']}")
    print(f"  Hybrid: {stats_hybrid['total']}")
    diff = stats_hybrid['total'] - stats_cutoff['total']
    print(f"  Difference: {diff} clauses")
    
    if diff > 0:
        print(f"  ⚠ Hybrid has {diff} MORE clauses than cutoff!")
    elif diff < 0:
        print(f"  ⚠ Hybrid has {-diff} FEWER clauses than cutoff!")
    else:
        print(f"  ✓ Clause count matches")
    
    print(f"\nBinary clauses (mutual exclusion):")
    print(f"  Cutoff: {stats_cutoff['binary_count']}")
    print(f"  Hybrid: {stats_hybrid['binary_count']}")
    binary_diff = stats_hybrid['binary_count'] - stats_cutoff['binary_count']
    print(f"  Difference: {binary_diff}")
    
    if binary_diff > 0:
        print(f"  ⚠ Hybrid has {binary_diff} MORE binary clauses!")
        print(f"    This suggests redundant mutual exclusion clauses")
    
    # Check for exact clause matching
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS")
    print(f"{'='*80}")
    
    # Convert clauses to sets for comparison
    cutoff_set = set(tuple(sorted(clause)) for clause in clauses_cutoff)
    hybrid_set = set(tuple(sorted(clause)) for clause in clauses_hybrid)
    
    only_cutoff = cutoff_set - hybrid_set
    only_hybrid = hybrid_set - cutoff_set
    common = cutoff_set & hybrid_set
    
    print(f"\nClause overlap:")
    print(f"  Common clauses: {len(common)}")
    print(f"  Only in cutoff: {len(only_cutoff)}")
    print(f"  Only in hybrid: {len(only_hybrid)}")
    
    if only_cutoff:
        print(f"\n  Sample clauses ONLY in cutoff (first 5):")
        for clause in list(only_cutoff)[:5]:
            print(f"    {clause}")
    
    if only_hybrid:
        print(f"\n  Sample clauses ONLY in hybrid (first 5):")
        for clause in list(only_hybrid)[:5]:
            print(f"    {clause}")
    
    return diff != 0


def main():
    print(f"{'='*80}")
    print(f"DEBUG: HYBRID vs CUTOFF PERFORMANCE")
    print(f"{'='*80}")
    print(f"\nThis script investigates why hybrid (full replacement) is slower than cutoff")
    
    # Test different sizes
    test_cases = [
        (5, "Small"),
        (8, "Medium"),
        (10, "Large"),
    ]
    
    issues_found = False
    
    for n, label in test_cases:
        UB = calculate_theoretical_upper_bound(n)
        print(f"\n{'#'*80}")
        print(f"# TEST CASE: {label} (n={n}, UB={UB})")
        print(f"{'#'*80}")
        
        has_diff = compare_single_edge(n, UB)
        if has_diff:
            issues_found = True
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    if issues_found:
        print(f"\n⚠ ISSUES FOUND:")
        print(f"  - Hybrid produces different clause count than cutoff")
        print(f"  - This explains the performance difference")
        print(f"  - Check the encoding logic for discrepancies")
        print(f"\nPossible causes:")
        print(f"  1. Stage 3 is creating redundant mutual exclusion clauses")
        print(f"  2. Stage 2 is encoding more than necessary")
        print(f"  3. Monotonicity clauses are duplicated")
    else:
        print(f"\n✓ NO STRUCTURAL ISSUES FOUND")
        print(f"  - Clause counts match between hybrid and cutoff")
        print(f"  - Performance difference may be due to:")
        print(f"    1. Clause ordering affecting solver heuristics")
        print(f"    2. Variable numbering differences")
        print(f"    3. Implementation overhead in encoding process")


if __name__ == "__main__":
    main()
