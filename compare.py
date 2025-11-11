"""So sánh chi tiết clause structure giữa hybrid và cutoff"""

from pysat.formula import IDPool
from distance_encoder_hybrid import encode_abs_distance_hybrid
from distance_encoder_cutoff import encode_abs_distance_cutoff
import math

def compare_single_edge_clauses(n=62, UB=11):
    """So sánh clauses chi tiết cho một edge"""
    
    print(f"Comparing single edge encoding (n={n}, UB={UB})")
    print("="*80)
    
    # Hybrid encoding
    vpool_h = IDPool()
    U_h = [vpool_h.id(f'U_{i}') for i in range(1, n + 1)]
    V_h = [vpool_h.id(f'V_{i}') for i in range(1, n + 1)]
    
    clauses_h, t_vars_h = encode_abs_distance_hybrid(
        U_h, V_h, n, UB, vpool_h,
        prefix="T_hybrid",
        num_replacements=n - 1 - UB  # Full replacement
    )
    
    # Cutoff encoding
    vpool_c = IDPool()
    U_c = [vpool_c.id(f'U_{i}') for i in range(1, n + 1)]
    V_c = [vpool_c.id(f'V_{i}') for i in range(1, n + 1)]
    
    clauses_c, t_vars_c = encode_abs_distance_cutoff(
        U_c, V_c, UB, vpool_c,
        t_var_prefix="T_cutoff"
    )
    
    print(f"\nHybrid: {len(clauses_h)} clauses, {len(t_vars_h)} T vars")
    print(f"Cutoff: {len(clauses_c)} clauses, {len(t_vars_c)} T vars")
    
    # Check clause order - compare ALL clauses for exact ordering
    print(f"\n{'='*80}")
    print("CLAUSE ORDERING COMPARISON:")
    print("="*80)
    
    # Count exact matches (same clause at same position)
    total_exact_matches = sum(1 for i in range(len(clauses_h)) if clauses_h[i] == clauses_c[i])
    print(f"\nExact position matches: {total_exact_matches}/{len(clauses_h)} ({100*total_exact_matches/len(clauses_h):.1f}%)")
    
    if total_exact_matches == len(clauses_h):
        print("✓ Perfect match - all clauses in EXACT same order!")
    else:
        # Find first mismatch
        first_diff = None
        for i in range(len(clauses_h)):
            if clauses_h[i] != clauses_c[i]:
                first_diff = i
                break
        
        if first_diff is not None:
            print(f"\n✗ First difference at position {first_diff + 1}:")
            print(f"   Hybrid: {clauses_h[first_diff]}")
            print(f"   Cutoff: {clauses_c[first_diff]}")
            
            # Analyze the ordering around the breakpoint
            print(f"\nClauses around position {first_diff + 1}:")
            for i in range(max(0, first_diff - 3), min(len(clauses_h), first_diff + 5)):
                match = "✓" if clauses_h[i] == clauses_c[i] else "✗"
                print(f"  {i+1:4}. H:{clauses_h[i]}")
                print(f"        C:{clauses_c[i]} {match}")
            
            # Check which type this belongs to
            t_var_set = set(t_vars_h.values())
            clause_h = clauses_h[first_diff]
            has_t = any(abs(lit) in t_var_set for lit in clause_h)
            print(f"\nFirst mismatched clause type: {'T-variable clause' if has_t else 'Mutual exclusion'}")
    
    # Separate clauses by type
    def categorize_clauses(clauses, t_var_ids):
        """Categorize clauses by type"""
        t_var_set = set(t_var_ids.values())
        
        mutual_exclusion = []
        t_activation = []
        t_deactivation = []
        t_monotonic = []
        t_base = []
        
        for clause in clauses:
            # Check if involves T variables
            clause_t_vars = [lit for lit in clause if abs(lit) in t_var_set]
            
            if not clause_t_vars:
                # No T variables - mutual exclusion
                mutual_exclusion.append(clause)
            elif len(clause) == 2 and all(lit < 0 or abs(lit) in t_var_set for lit in clause):
                # Two literals, at least one is T - monotonic
                t_monotonic.append(clause)
            elif len(clause) == 3:
                if clause_t_vars and clause_t_vars[0] > 0:
                    # Positive T - activation
                    t_activation.append(clause)
                else:
                    # Negative T - deactivation or base
                    if any(abs(lit) in t_var_set and lit > 0 for lit in clause):
                        t_base.append(clause)
                    else:
                        t_deactivation.append(clause)
        
        return {
            'mutual_exclusion': mutual_exclusion,
            't_activation': t_activation,
            't_deactivation': t_deactivation,
            't_monotonic': t_monotonic,
            't_base': t_base
        }
    
    cats_h = categorize_clauses(clauses_h, t_vars_h)
    cats_c = categorize_clauses(clauses_c, t_vars_c)
    
    print(f"\n{'='*80}")
    print("CLAUSE TYPE DISTRIBUTION:")
    print("="*80)
    print(f"\n{'Category':<20} {'Hybrid':<10} {'Cutoff':<10} {'Diff':<10}")
    print("-"*50)
    for cat in ['mutual_exclusion', 't_activation', 't_deactivation', 't_monotonic', 't_base']:
        h_count = len(cats_h[cat])
        c_count = len(cats_c[cat])
        diff = h_count - c_count
        print(f"{cat:<20} {h_count:<10} {c_count:<10} {diff:<10}")
    
    # Check if clause sets are identical (ignoring order)
    clauses_h_set = set(tuple(sorted(c)) for c in clauses_h)
    clauses_c_set = set(tuple(sorted(c)) for c in clauses_c)
    
    only_h = clauses_h_set - clauses_c_set
    only_c = clauses_c_set - clauses_h_set
    common = clauses_h_set & clauses_c_set
    
    print(f"\n{'='*80}")
    print("SET COMPARISON (structure, ignoring order):")
    print("="*80)
    print(f"Common clauses:     {len(common)}")
    print(f"Only in hybrid:     {len(only_h)}")
    print(f"Only in cutoff:     {len(only_c)}")
    
    if len(only_h) > 0:
        print(f"\nSample clauses ONLY in HYBRID (first 5):")
        for clause in list(only_h)[:5]:
            print(f"  {clause}")
    
    if len(only_c) > 0:
        print(f"\nSample clauses ONLY in CUTOFF (first 5):")
        for clause in list(only_c)[:5]:
            print(f"  {clause}")
    
    # Final verdict
    print(f"\n{'='*80}")
    print("VERDICT:")
    print("="*80)
    if len(only_h) == 0 and len(only_c) == 0:
        print("✓ Clause sets are IDENTICAL (same structure)")
        print("✗ But clause ORDERING is different!")
        print("  → This explains the performance difference")
    else:
        print("✗ Clause sets are DIFFERENT")
        print(f"  → Hybrid has {len(only_h)} unique clauses")
        print(f"  → Cutoff has {len(only_c)} unique clauses")


if __name__ == "__main__":
    compare_single_edge_clauses(n=62, UB=11)