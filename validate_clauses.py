#!/usr/bin/env python3
# validate_clauses.py
# Validate clause generation for cutoff vs hybrid encoders

import sys
import os
from collections import defaultdict, Counter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pysat.formula import IDPool
from distance_encoder_cutoff import encode_abs_distance_cutoff, calculate_theoretical_upper_bound
from distance_encoder_hybrid import encode_abs_distance_hybrid

def dump_clauses_to_file(clauses, filename, t_vars=None, vpool=None):
    """Dump clauses to file for inspection"""
    with open(filename, 'w') as f:
        f.write(f"Total clauses: {len(clauses)}\n")
        f.write(f"Total variables: {vpool.top if vpool else 'N/A'}\n")
        if t_vars:
            f.write(f"T variables: {len(t_vars)}\n")
            f.write(f"T var mapping: {t_vars}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("CLAUSES:\n")
        f.write("="*70 + "\n\n")
        
        for i, clause in enumerate(clauses, 1):
            f.write(f"Clause {i:5d}: {clause}\n")

def analyze_clause_ordering(clauses_c, clauses_h):
    """Analyze if clause ordering matches"""
    print(f"\n{'='*70}")
    print(f"CLAUSE ORDERING ANALYSIS")
    print(f"{'='*70}")
    
    # Check if first N clauses match
    check_counts = [10, 50, 100, 500, len(clauses_c)]
    
    for n in check_counts:
        if n > len(clauses_c):
            n = len(clauses_c)
        
        matches = sum(1 for i in range(n) if clauses_c[i] == clauses_h[i])
        match_pct = (matches / n) * 100 if n > 0 else 0
        
        print(f"First {n:5d} clauses: {matches:5d} matches ({match_pct:5.1f}%)")
        
        if n >= len(clauses_c):
            break

def compare_encoders_detailed(n, test_label="Test"):
    """Detailed comparison of cutoff vs hybrid encoders"""
    
    UB = calculate_theoretical_upper_bound(n)
    max_repl = n - 1 - UB
    
    print(f"\n{'='*70}")
    print(f"{test_label}: n={n}, UB={UB}, max_repl={max_repl}")
    print(f"{'='*70}")
    
    # Cutoff encoding
    vpool_c = IDPool()
    U_c = [vpool_c.id(f'U_{i}') for i in range(1, n + 1)]
    V_c = [vpool_c.id(f'V_{i}') for i in range(1, n + 1)]
    
    print("\nGenerating CUTOFF encoding...")
    clauses_c, t_c = encode_abs_distance_cutoff(U_c, V_c, UB, vpool_c, "T_c")
    print(f"  Generated: {len(clauses_c)} clauses, {len(t_c)} T vars, {vpool_c.top} total vars")
    
    # Hybrid encoding (full replacement)
    vpool_h = IDPool()
    U_h = [vpool_h.id(f'U_{i}') for i in range(1, n + 1)]
    V_h = [vpool_h.id(f'V_{i}') for i in range(1, n + 1)]
    
    print("\nGenerating HYBRID encoding (full replacement)...")
    clauses_h, t_h = encode_abs_distance_hybrid(U_h, V_h, n, UB, vpool_h, "T_h", max_repl)
    print(f"  Generated: {len(clauses_h)} clauses, {len(t_h)} T vars, {vpool_h.top} total vars")
    
    # Dump to files
    cutoff_file = f"clauses_cutoff_n{n}.txt"
    hybrid_file = f"clauses_hybrid_n{n}.txt"
    
    print(f"\nDumping clauses to files...")
    dump_clauses_to_file(clauses_c, cutoff_file, t_c, vpool_c)
    print(f"  Cutoff: {cutoff_file}")
    
    dump_clauses_to_file(clauses_h, hybrid_file, t_h, vpool_h)
    print(f"  Hybrid: {hybrid_file}")
    
    # Basic comparison
    print(f"\n{'='*70}")
    print(f"BASIC COMPARISON")
    print(f"{'='*70}")
    print(f"Clause count match: {len(clauses_c) == len(clauses_h)}")
    print(f"  Cutoff: {len(clauses_c)} clauses")
    print(f"  Hybrid: {len(clauses_h)} clauses")
    print(f"  Diff:   {len(clauses_h) - len(clauses_c)}")
    
    print(f"\nT variable count match: {len(t_c) == len(t_h)}")
    print(f"  Cutoff: {len(t_c)} T vars")
    print(f"  Hybrid: {len(t_h)} T vars")
    
    # Structural comparison
    clauses_c_set = set(tuple(sorted(c)) for c in clauses_c)
    clauses_h_set = set(tuple(sorted(c)) for c in clauses_h)
    
    common = clauses_c_set & clauses_h_set
    only_cutoff = clauses_c_set - clauses_h_set
    only_hybrid = clauses_h_set - clauses_c_set
    
    print(f"\n{'='*70}")
    print(f"STRUCTURAL COMPARISON (order-independent)")
    print(f"{'='*70}")
    print(f"Common clauses:     {len(common)}")
    print(f"Only in cutoff:     {len(only_cutoff)}")
    print(f"Only in hybrid:     {len(only_hybrid)}")
    print(f"Structural match:   {len(only_cutoff) == 0 and len(only_hybrid) == 0}")
    
    if len(only_cutoff) > 0:
        print(f"\nSample clauses ONLY in CUTOFF (first 5):")
        for i, clause in enumerate(list(only_cutoff)[:5], 1):
            print(f"  {i}. {clause}")
    
    if len(only_hybrid) > 0:
        print(f"\nSample clauses ONLY in HYBRID (first 5):")
        for i, clause in enumerate(list(only_hybrid)[:5], 1):
            print(f"  {i}. {clause}")
    
    # Ordering comparison
    analyze_clause_ordering(clauses_c, clauses_h)
    
    # Detailed first 20 clauses comparison
    print(f"\n{'='*70}")
    print(f"FIRST 20 CLAUSES DETAILED COMPARISON")
    print(f"{'='*70}")
    print(f"{'Idx':>4} | {'Cutoff Clause':^50} | {'Hybrid Clause':^50} | {'Match':^5}")
    print(f"{'-'*4}|{'-'*52}|{'-'*52}|{'-'*7}")
    
    for i in range(min(20, len(clauses_c), len(clauses_h))):
        c_clause = str(clauses_c[i])
        h_clause = str(clauses_h[i])
        match = "✓" if clauses_c[i] == clauses_h[i] else "✗"
        
        # Truncate if too long
        if len(c_clause) > 48:
            c_clause = c_clause[:45] + "..."
        if len(h_clause) > 48:
            h_clause = h_clause[:45] + "..."
        
        print(f"{i+1:4} | {c_clause:50} | {h_clause:50} | {match:^5}")
    
    return {
        'n': n,
        'structural_match': len(only_cutoff) == 0 and len(only_hybrid) == 0,
        'clause_count_match': len(clauses_c) == len(clauses_h),
        'cutoff_file': cutoff_file,
        'hybrid_file': hybrid_file
    }

def main():
    """Main validation function"""
    
    print("="*70)
    print("CLAUSE VALIDATION TOOL")
    print("Comparing cutoff vs hybrid (full replacement) encoders")
    print("="*70)
    
    # Test cases
    test_cases = [
        (10, "Small graph (n=10)"),
        (20, "Medium graph (n=20)"),
        (62, "bfw62a.mtx size (n=62)")
    ]
    
    results = []
    for n, label in test_cases:
        result = compare_encoders_detailed(n, label)
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    all_match = all(r['structural_match'] and r['clause_count_match'] for r in results)
    
    for r in results:
        status = "✓ PASS" if (r['structural_match'] and r['clause_count_match']) else "✗ FAIL"
        print(f"n={r['n']:3d}: {status}")
        print(f"       Cutoff clauses: {r['cutoff_file']}")
        print(f"       Hybrid clauses: {r['hybrid_file']}")
    
    print(f"\n{'='*70}")
    if all_match:
        print(f"✓ ALL TESTS PASSED")
        print(f"  Cutoff and Hybrid encoders produce identical clauses (structure)")
        print(f"  Performance differences must be due to:")
        print(f"    1. Clause ordering (affects solver heuristics)")
        print(f"    2. Variable numbering scheme")
        print(f"    3. Other implementation details")
    else:
        print(f"✗ SOME TESTS FAILED")
        print(f"  Encoders produce different clauses!")
        print(f"  Check the dumped files for details")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
