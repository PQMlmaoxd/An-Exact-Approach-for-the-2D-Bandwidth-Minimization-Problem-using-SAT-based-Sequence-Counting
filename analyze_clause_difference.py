#!/usr/bin/env python3
# analyze_clause_difference.py
# Deep analysis of clause differences between cutoff and hybrid encodings

import sys
import os
from collections import defaultdict, Counter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pysat.formula import IDPool
from distance_encoder_cutoff import encode_abs_distance_cutoff, calculate_theoretical_upper_bound
from distance_encoder_hybrid import encode_abs_distance_hybrid

def analyze_single_edge_deep(n, UB, num_repl):
    """Deep analysis of single edge encoding"""
    
    print(f"\n{'='*70}")
    print(f"DEEP ANALYSIS: Single Edge")
    print(f"n={n}, UB={UB}, num_repl={num_repl}")
    print(f"{'='*70}")
    
    # Cutoff encoding
    vpool_c = IDPool()
    U_c = [vpool_c.id(f'U_{i}') for i in range(1, n + 1)]
    V_c = [vpool_c.id(f'V_{i}') for i in range(1, n + 1)]
    clauses_c, t_c = encode_abs_distance_cutoff(U_c, V_c, UB, vpool_c, "T_c")
    
    # Hybrid encoding
    vpool_h = IDPool()
    U_h = [vpool_h.id(f'U_{i}') for i in range(1, n + 1)]
    V_h = [vpool_h.id(f'V_{i}') for i in range(1, n + 1)]
    clauses_h, t_h = encode_abs_distance_hybrid(U_h, V_h, n, UB, vpool_h, "T_h", num_repl)
    
    print(f"\n1. BASIC STATS:")
    print(f"   Cutoff:  {len(clauses_c)} clauses, {len(t_c)} T vars, {vpool_c.top} total vars")
    print(f"   Hybrid:  {len(clauses_h)} clauses, {len(t_h)} T vars, {vpool_h.top} total vars")
    
    # Analyze clause length distribution
    len_dist_c = Counter(len(c) for c in clauses_c)
    len_dist_h = Counter(len(c) for c in clauses_h)
    
    print(f"\n2. CLAUSE LENGTH DISTRIBUTION:")
    print(f"   Length | Cutoff | Hybrid | Diff")
    print(f"   -------|--------|--------|------")
    all_lens = sorted(set(len_dist_c.keys()) | set(len_dist_h.keys()))
    for length in all_lens:
        c_count = len_dist_c.get(length, 0)
        h_count = len_dist_h.get(length, 0)
        diff = h_count - c_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"   {length:6} | {c_count:6} | {h_count:6} | {diff_str:>6}")
    
    # Analyze by clause type (heuristic based on length and structure)
    def classify_clause(clause):
        """Classify clause type"""
        length = len(clause)
        if length == 1:
            return "unit"
        elif length == 2:
            # Check if it's implication-like
            if all(lit < 0 for lit in clause):
                return "binary_neg"
            elif all(lit > 0 for lit in clause):
                return "binary_pos"
            else:
                return "binary_mixed"
        elif length == 3:
            return "ternary"
        elif length <= 5:
            return "small"
        elif length <= 10:
            return "medium"
        else:
            return "large"
    
    type_dist_c = Counter(classify_clause(c) for c in clauses_c)
    type_dist_h = Counter(classify_clause(c) for c in clauses_h)
    
    print(f"\n3. CLAUSE TYPE DISTRIBUTION:")
    print(f"   Type         | Cutoff | Hybrid | Diff")
    print(f"   -------------|--------|--------|------")
    all_types = sorted(set(type_dist_c.keys()) | set(type_dist_h.keys()))
    for ctype in all_types:
        c_count = type_dist_c.get(ctype, 0)
        h_count = type_dist_h.get(ctype, 0)
        diff = h_count - c_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"   {ctype:12} | {c_count:6} | {h_count:6} | {diff_str:>6}")
    
    # Find structural differences
    clauses_c_set = set(tuple(sorted(c)) for c in clauses_c)
    clauses_h_set = set(tuple(sorted(c)) for c in clauses_h)
    
    only_cutoff = clauses_c_set - clauses_h_set
    only_hybrid = clauses_h_set - clauses_c_set
    common = clauses_c_set & clauses_h_set
    
    print(f"\n4. STRUCTURAL COMPARISON:")
    print(f"   Common clauses:        {len(common)}")
    print(f"   Only in cutoff:        {len(only_cutoff)}")
    print(f"   Only in hybrid:        {len(only_hybrid)}")
    
    if len(only_cutoff) > 0:
        print(f"\n   Sample clauses ONLY in CUTOFF (first 10):")
        for i, clause in enumerate(list(only_cutoff)[:10], 1):
            print(f"      {i}. {clause}")
    
    if len(only_hybrid) > 0:
        print(f"\n   Sample clauses ONLY in HYBRID (first 10):")
        for i, clause in enumerate(list(only_hybrid)[:10], 1):
            print(f"      {i}. {clause}")
    
    # Analyze variable usage
    def get_var_usage(clauses):
        """Count how many times each variable appears"""
        usage = defaultdict(int)
        for clause in clauses:
            for lit in clause:
                usage[abs(lit)] += 1
        return usage
    
    usage_c = get_var_usage(clauses_c)
    usage_h = get_var_usage(clauses_h)
    
    print(f"\n5. VARIABLE USAGE:")
    print(f"   Cutoff: {len(usage_c)} vars, avg usage = {sum(usage_c.values())/len(usage_c):.1f}")
    print(f"   Hybrid: {len(usage_h)} vars, avg usage = {sum(usage_h.values())/len(usage_h):.1f}")
    
    # Find most used variables
    top_vars_c = sorted(usage_c.items(), key=lambda x: x[1], reverse=True)[:5]
    top_vars_h = sorted(usage_h.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\n   Top 5 most used variables:")
    print(f"   Cutoff: {top_vars_c}")
    print(f"   Hybrid: {top_vars_h}")
    
    return {
        'cutoff_clauses': len(clauses_c),
        'hybrid_clauses': len(clauses_h),
        'common': len(common),
        'only_cutoff': len(only_cutoff),
        'only_hybrid': len(only_hybrid),
        'match': len(only_cutoff) == 0 and len(only_hybrid) == 0
    }


def test_multiple_sizes():
    """Test multiple graph sizes"""
    
    print(f"\n{'='*70}")
    print(f"TESTING MULTIPLE SIZES")
    print(f"{'='*70}")
    
    test_cases = [
        (10, "Small"),
        (20, "Medium"),
        (62, "Large (bfw62a)"),
    ]
    
    results = []
    for n, label in test_cases:
        UB = calculate_theoretical_upper_bound(n)
        max_repl = n - 1 - UB
        
        print(f"\n{'='*70}")
        print(f"Test: {label} (n={n}, UB={UB}, max_repl={max_repl})")
        print(f"{'='*70}")
        
        result = analyze_single_edge_deep(n, UB, max_repl)
        result['n'] = n
        result['label'] = label
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Size':20} | {'Match':5} | {'Common':8} | {'Only C':8} | {'Only H':8}")
    print(f"{'-'*20}|{'-'*7}|{'-'*10}|{'-'*10}|{'-'*10}")
    for r in results:
        match_str = "✓" if r['match'] else "✗"
        print(f"{r['label']:20} | {match_str:^5} | {r['common']:8} | {r['only_cutoff']:8} | {r['only_hybrid']:8}")


if __name__ == "__main__":
    print("CLAUSE DIFFERENCE ANALYZER")
    print("Comparing cutoff vs hybrid (full replacement)")
    
    # Test bfw62a size specifically
    n = 62
    UB = calculate_theoretical_upper_bound(n)
    max_repl = n - 1 - UB
    
    print(f"\nAnalyzing bfw62a.mtx scenario:")
    print(f"  n = {n}")
    print(f"  UB = {UB}")
    print(f"  max_repl = {max_repl}")
    
    result = analyze_single_edge_deep(n, UB, max_repl)
    
    if not result['match']:
        print(f"\n{'='*70}")
        print(f"⚠ WARNING: ENCODINGS DO NOT MATCH!")
        print(f"{'='*70}")
        print(f"This explains the performance difference!")
        print(f"Hybrid has {result['only_hybrid']} unique clauses")
        print(f"Cutoff has {result['only_cutoff']} unique clauses")
    else:
        print(f"\n{'='*70}")
        print(f"✓ ENCODINGS MATCH STRUCTURALLY")
        print(f"{'='*70}")
        print(f"The performance difference must be due to:")
        print(f"  1. Clause ordering (affects solver heuristics)")
        print(f"  2. Variable numbering scheme")
        print(f"  3. Encoding overhead/timing")
