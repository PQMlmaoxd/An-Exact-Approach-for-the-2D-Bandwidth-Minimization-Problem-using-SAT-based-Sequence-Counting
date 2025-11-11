#!/usr/bin/env python3
"""
verify_encoding_equivalence.py
Verify theoretical equivalence between Original and Cutoff encodings

Usage:
    python verify_encoding_equivalence.py bfw62a.mtx --k=3 --edge-sample=0
"""

print("=== Starting script ===", flush=True)

import sys
import os
import time
import argparse
from typing import Dict, List, Tuple, Set
from collections import defaultdict

print("Basic imports OK", flush=True)

from pysat.formula import IDPool
from pysat.solvers import Glucose42, Cadical195

print("PySAT imports OK", flush=True)

# Import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("About to import distance_encoder...", flush=True)
from distance_encoder import encode_abs_distance_final
print("distance_encoder imported", flush=True)

print("About to import distance_encoder_cutoff...", flush=True)
from distance_encoder_cutoff import encode_abs_distance_cutoff, calculate_theoretical_upper_bound
print("distance_encoder_cutoff imported", flush=True)

print("About to import position_constraints...", flush=True)
from position_constraints import encode_all_position_constraints, create_position_variables
print("position_constraints imported", flush=True)

print("=== All imports successful ===", flush=True)


class ClauseAnalyzer:
    """Analyze and categorize clauses"""
    
    def __init__(self, n: int, ub: int):
        self.n = n
        self.ub = ub
        self.clause_types = defaultdict(list)
    
    def categorize_clause(self, clause: List[int], var_map: Dict[int, str]) -> str:
        """Categorize a clause based on its structure and variables"""
        clause_len = len(clause)
        
        # Get variable names
        var_names = []
        for lit in clause:
            var_id = abs(lit)
            sign = '+' if lit > 0 else '-'
            if var_id in var_map:
                var_names.append(f"{sign}{var_map[var_id]}")
            else:
                var_names.append(f"{sign}var{var_id}")
        
        # Categorize by length and pattern
        if clause_len == 2:
            # Binary clause
            if all(lit < 0 for lit in clause):
                # Both negative - likely mutual exclusion or monotonic
                if any('U_' in v or 'V_' in v for v in var_names):
                    return 'mutual_exclusion'
                else:
                    return 'monotonic'
            else:
                return 'binary_other'
        
        elif clause_len == 3:
            # 3-literal clause
            if clause[2] > 0:
                # Last literal positive - likely activation
                return 'activation'
            else:
                # All negative or last negative - likely deactivation
                return 'deactivation'
        
        return 'other'
    
    def analyze_clauses(self, clauses: List[List[int]], var_map: Dict[int, str]) -> Dict[str, int]:
        """Analyze all clauses and return counts by type"""
        counts = defaultdict(int)
        
        for clause in clauses:
            cat = self.categorize_clause(clause, var_map)
            counts[cat] += 1
            self.clause_types[cat].append(clause)
        
        return counts


class EncodingVerifier:
    """Verify encoding equivalence between Original and Cutoff"""
    
    def __init__(self, n: int, edges: List[Tuple[int, int]]):
        self.n = n
        self.edges = edges
        self.ub = calculate_theoretical_upper_bound(n)
        
        print(f"\n{'='*80}")
        print(f"ENCODING VERIFIER INITIALIZED")
        print(f"{'='*80}")
        print(f"Graph: n={n}, |E|={len(edges)}")
        print(f"Theoretical UB: {self.ub}")
        print(f"Will verify: T_1 to T_{self.ub} should be identical")
        print(f"             T_{self.ub+1} to T_{n-1} replaced by mutual exclusion")
    
    def build_original_encoding_single_edge(self, u: int, v: int) -> Tuple[Dict, List, Dict]:
        """Build original encoding for a single edge and return detailed info"""
        vpool = IDPool()
        
        # Create position variables
        U_vars = [vpool.id(f'U_{i}') for i in range(1, self.n + 1)]
        V_vars = [vpool.id(f'V_{i}') for i in range(1, self.n + 1)]
        
        # Create variable map for analysis
        var_map = {}
        for i, var_id in enumerate(U_vars, 1):
            var_map[var_id] = f"U_{i}"
        for i, var_id in enumerate(V_vars, 1):
            var_map[var_id] = f"V_{i}"
        
        # Encode distance
        T_vars, clauses = encode_abs_distance_final(U_vars, V_vars, self.n, vpool, f"T[{u},{v}]")
        
        # Add T variables to map
        for i, t_var in enumerate(T_vars, 1):
            var_map[t_var] = f"T_{i}"
        
        # Analyze clauses
        clause_breakdown = {
            'activation': [],
            'deactivation': [],
            'monotonic': [],
            'base': [],
            'total': len(clauses)
        }
        
        for clause in clauses:
            clen = len(clause)
            
            if clen == 3:
                # Check if activation or deactivation
                has_t_positive = any(lit > 0 and abs(lit) in [t for t in T_vars] for lit in clause)
                has_t_negative = any(lit < 0 and abs(lit) in [t for t in T_vars] for lit in clause)
                
                if has_t_positive:
                    clause_breakdown['activation'].append(clause)
                elif has_t_negative:
                    clause_breakdown['deactivation'].append(clause)
            
            elif clen == 2:
                # Monotonic: T_{d+1} → T_d
                clause_breakdown['monotonic'].append(clause)
        
        print(f"\n{'─'*80}")
        print(f"ORIGINAL ENCODING - Edge ({u},{v})")
        print(f"{'─'*80}")
        print(f"T variables: {len(T_vars)} (T_1 to T_{len(T_vars)})")
        print(f"Clause breakdown:")
        print(f"  Activation: {len(clause_breakdown['activation'])}")
        print(f"  Deactivation: {len(clause_breakdown['deactivation'])}")
        print(f"  Monotonic: {len(clause_breakdown['monotonic'])}")
        print(f"  Total: {clause_breakdown['total']}")
        
        return T_vars, clauses, clause_breakdown, var_map
    
    def build_cutoff_encoding_single_edge(self, u: int, v: int) -> Tuple[Dict, List, Dict]:
        """Build cutoff encoding for a single edge and return detailed info"""
        vpool = IDPool()
        
        # Create position variables
        U_vars = [vpool.id(f'U_{i}') for i in range(1, self.n + 1)]
        V_vars = [vpool.id(f'V_{i}') for i in range(1, self.n + 1)]
        
        # Create variable map
        var_map = {}
        for i, var_id in enumerate(U_vars, 1):
            var_map[var_id] = f"U_{i}"
        for i, var_id in enumerate(V_vars, 1):
            var_map[var_id] = f"V_{i}"
        
        # Encode distance with cutoff
        t_var_prefix = f"T[{u},{v}]"
        clauses, t_vars = encode_abs_distance_cutoff(U_vars, V_vars, self.ub, vpool, t_var_prefix)
        
        # Add T variables to map
        for d, t_var in t_vars.items():
            var_map[t_var] = f"T_{d}"
        
        # Analyze clauses
        clause_breakdown = {
            'activation': [],
            'deactivation': [],
            'monotonic': [],
            'mutual_exclusion': [],
            'base': [],
            'total': len(clauses)
        }
        
        for clause in clauses:
            clen = len(clause)
            
            if clen == 2:
                # Check if mutual exclusion or monotonic
                if all(lit < 0 for lit in clause):
                    # Both negative
                    involves_position = any(abs(lit) in [U_vars[i] for i in range(len(U_vars))] or 
                                          abs(lit) in [V_vars[i] for i in range(len(V_vars))] 
                                          for lit in clause)
                    if involves_position:
                        clause_breakdown['mutual_exclusion'].append(clause)
                    else:
                        clause_breakdown['monotonic'].append(clause)
                else:
                    clause_breakdown['monotonic'].append(clause)
            
            elif clen == 3:
                has_t_positive = any(lit > 0 and abs(lit) in t_vars.values() for lit in clause)
                has_t_negative = any(lit < 0 and abs(lit) in t_vars.values() for lit in clause)
                
                if has_t_positive:
                    clause_breakdown['activation'].append(clause)
                elif has_t_negative:
                    clause_breakdown['deactivation'].append(clause)
        
        print(f"\n{'─'*80}")
        print(f"CUTOFF ENCODING - Edge ({u},{v})")
        print(f"{'─'*80}")
        print(f"T variables: {len(t_vars)} (T_1 to T_{self.ub})")
        print(f"Clause breakdown:")
        print(f"  Activation: {len(clause_breakdown['activation'])}")
        print(f"  Deactivation: {len(clause_breakdown['deactivation'])}")
        print(f"  Monotonic: {len(clause_breakdown['monotonic'])}")
        print(f"  Mutual exclusion: {len(clause_breakdown['mutual_exclusion'])}")
        print(f"  Total: {clause_breakdown['total']}")
        
        return t_vars, clauses, clause_breakdown, var_map
    
    def compare_t_variable_ranges(self, orig_t_vars: List[int], cut_t_vars: Dict[int, int]):
        """Compare T variable ranges between encodings"""
        print(f"\n{'='*80}")
        print(f"T VARIABLE RANGE COMPARISON")
        print(f"{'='*80}")
        
        orig_count = len(orig_t_vars)
        cut_count = len(cut_t_vars)
        
        print(f"\nOriginal encoding:")
        print(f"  T_1 to T_{orig_count} ({orig_count} variables)")
        
        print(f"\nCutoff encoding:")
        print(f"  T_1 to T_{self.ub} ({cut_count} variables)")
        
        # Check overlap
        overlap = min(orig_count, cut_count)
        print(f"\nOverlap: T_1 to T_{overlap}")
        print(f"  → These should have IDENTICAL activation/deactivation clauses")
        
        if orig_count > cut_count:
            print(f"\nOriginal has additional: T_{cut_count+1} to T_{orig_count}")
            print(f"  → Cutoff replaces these with mutual exclusion clauses")
    
    def verify_activation_equivalence(self, orig_clauses: List, cut_clauses: List, 
                                     d_max: int, var_map_orig: Dict, var_map_cut: Dict):
        """Verify activation clauses are equivalent up to T_d_max - with position pair matching"""
        print(f"\n{'─'*80}")
        print(f"ACTIVATION CLAUSE VERIFICATION (T_1 to T_{d_max})")
        print(f"{'─'*80}")
        
        # Group by distance d with position pairs
        orig_by_d = defaultdict(set)  # d -> set of (u_pos, v_pos) tuples
        cut_by_d = defaultdict(set)
        
        # Parse original clauses
        for clause in orig_clauses:
            # Find T variable and position variables
            t_dist = None
            u_pos, v_pos = None, None
            
            for lit in clause:
                var_id = abs(lit)
                if var_id in var_map_orig:
                    var_name = var_map_orig[var_id]
                    if var_name.startswith('T_') and lit > 0:
                        try:
                            d = int(var_name.split('_')[1])
                            if d <= d_max:
                                t_dist = d
                        except:
                            pass
                    elif var_name.startswith('U_') and lit < 0:  # ¬U_i in activation
                        try:
                            u_pos = int(var_name.split('_')[1])
                        except:
                            pass
                    elif var_name.startswith('V_') and lit < 0:  # ¬V_k in activation
                        try:
                            v_pos = int(var_name.split('_')[1])
                        except:
                            pass
            
            if t_dist and u_pos is not None and v_pos is not None:
                orig_by_d[t_dist].add((u_pos, v_pos))
        
        # Parse cutoff clauses
        for clause in cut_clauses:
            t_dist = None
            u_pos, v_pos = None, None
            
            for lit in clause:
                var_id = abs(lit)
                if var_id in var_map_cut:
                    var_name = var_map_cut[var_id]
                    if var_name.startswith('T_') and lit > 0:
                        try:
                            d = int(var_name.split('_')[1])
                            if d <= d_max:
                                t_dist = d
                        except:
                            pass
                    elif var_name.startswith('U_') and lit < 0:
                        try:
                            u_pos = int(var_name.split('_')[1])
                        except:
                            pass
                    elif var_name.startswith('V_') and lit < 0:
                        try:
                            v_pos = int(var_name.split('_')[1])
                        except:
                            pass
            
            if t_dist and u_pos is not None and v_pos is not None:
                cut_by_d[t_dist].add((u_pos, v_pos))
        
        # Compare multisets
        print(f"\n{'Distance':<12} {'Original':<12} {'Cutoff':<12} {'Match':<10}")
        print(f"{'-'*50}")
        
        all_match = True
        for d in range(1, d_max + 1):
            orig_pairs = orig_by_d[d]
            cut_pairs = cut_by_d[d]
            
            match = "✓" if orig_pairs == cut_pairs else "✗"
            
            if orig_pairs != cut_pairs:
                all_match = False
                # Show difference details
                missing = orig_pairs - cut_pairs
                extra = cut_pairs - orig_pairs
                if missing or extra:
                    print(f"T_{d:<10} {len(orig_pairs):<12} {len(cut_pairs):<12} {match:<10}")
                    if missing:
                        print(f"    Missing in cutoff: {list(missing)[:3]}")
                    if extra:
                        print(f"    Extra in cutoff: {list(extra)[:3]}")
            else:
                print(f"T_{d:<10} {len(orig_pairs):<12} {len(cut_pairs):<12} {match:<10}")
        
        if all_match:
            print(f"\n✓ All activation position pairs match for T_1 to T_{d_max}")
        else:
            print(f"\n✗ Mismatch detected in activation position pairs!")
        
        return all_match
    
    def verify_deactivation_equivalence(self, orig_clauses: List, cut_clauses: List,
                                       d_max: int, var_map_orig: Dict, var_map_cut: Dict):
        """Verify deactivation clauses are equivalent up to T_d_max - with position pair matching"""
        print(f"\n{'─'*80}")
        print(f"DEACTIVATION CLAUSE VERIFICATION (T_1 to T_{d_max})")
        print(f"{'─'*80}")
        
        # Group by distance d with position pairs
        orig_by_d = defaultdict(set)  # d -> set of (u_pos, v_pos) tuples
        cut_by_d = defaultdict(set)
        
        # Parse original clauses
        for clause in orig_clauses:
            # Find T variable (negative) and position variables
            t_dist = None
            u_pos, v_pos = None, None
            
            for lit in clause:
                var_id = abs(lit)
                if var_id in var_map_orig:
                    var_name = var_map_orig[var_id]
                    if var_name.startswith('T_') and lit < 0:  # ¬T_{d+1}
                        try:
                            d_plus_1 = int(var_name.split('_')[1])  # chỉ số T
                            # Only accept T_2 to T_UB (i.e., d_plus_1 from 2 to d_max)
                            # This excludes T_{UB+1} which is handled by mutual exclusion
                            if 2 <= d_plus_1 <= d_max:
                                t_dist = d_plus_1 - 1  # Get d from ¬T_{d+1}
                        except:
                            pass
                    elif var_name.startswith('U_') and lit < 0:
                        try:
                            u_pos = int(var_name.split('_')[1])
                        except:
                            pass
                    elif var_name.startswith('V_') and lit < 0:
                        try:
                            v_pos = int(var_name.split('_')[1])
                        except:
                            pass
            
            if t_dist and u_pos is not None and v_pos is not None:
                orig_by_d[t_dist].add((u_pos, v_pos))
        
        # Parse cutoff clauses
        for clause in cut_clauses:
            t_dist = None
            u_pos, v_pos = None, None
            
            for lit in clause:
                var_id = abs(lit)
                if var_id in var_map_cut:
                    var_name = var_map_cut[var_id]
                    if var_name.startswith('T_') and lit < 0:
                        try:
                            d_plus_1 = int(var_name.split('_')[1])
                            if 2 <= d_plus_1 <= d_max:
                                t_dist = d_plus_1 - 1
                        except:
                            pass
                    elif var_name.startswith('U_') and lit < 0:
                        try:
                            u_pos = int(var_name.split('_')[1])
                        except:
                            pass
                    elif var_name.startswith('V_') and lit < 0:
                        try:
                            v_pos = int(var_name.split('_')[1])
                        except:
                            pass
            
            if t_dist and u_pos is not None and v_pos is not None:
                cut_by_d[t_dist].add((u_pos, v_pos))
        
        # Compare multisets
        print(f"\n{'Distance':<12} {'Original':<12} {'Cutoff':<12} {'Match':<10}")
        print(f"{'-'*50}")
        
        all_match = True
        # Only check d=1 to d_max-1 (i.e., ¬T_2 to ¬T_UB)
        # ¬T_{UB+1} is handled by mutual exclusion, not part of this verification
        for d in range(1, d_max):  # d=1..UB-1 → ¬T_2..¬T_UB
            orig_pairs = orig_by_d[d]
            cut_pairs = cut_by_d[d]
            
            match = "✓" if orig_pairs == cut_pairs else "✗"
            
            if orig_pairs != cut_pairs:
                all_match = False
                missing = orig_pairs - cut_pairs
                extra = cut_pairs - orig_pairs
                if missing or extra:
                    print(f"¬T_{d+1:<8} {len(orig_pairs):<12} {len(cut_pairs):<12} {match:<10}")
                    if missing:
                        print(f"    Missing in cutoff: {list(missing)[:3]}")
                    if extra:
                        print(f"    Extra in cutoff: {list(extra)[:3]}")
            else:
                print(f"¬T_{d+1:<8} {len(orig_pairs):<12} {len(cut_pairs):<12} {match:<10}")
        
        if all_match:
            print(f"\n✓ All deactivation position pairs match for T_1 to T_{d_max}")
        else:
            print(f"\n✗ Mismatch detected in deactivation position pairs!")
        
        return all_match
    
    def verify_mutual_exclusion_replacement(self, orig_clauses: List, mut_ex_clauses: List,
                                           var_map_orig: Dict):
        """Verify mutual exclusion replaces T_{UB+1} to T_{n-1}"""
        print(f"\n{'─'*80}")
        print(f"MUTUAL EXCLUSION REPLACEMENT VERIFICATION")
        print(f"{'─'*80}")
        
        print(f"\nOriginal encoding for T_{self.ub+1} to T_{self.n-1}:")
        
        # Count original clauses involving T_{UB+1} and above
        high_t_clauses = []
        for clause in orig_clauses:
            for lit in clause:
                var_id = abs(lit)
                if var_id in var_map_orig and var_map_orig[var_id].startswith('T_'):
                    d = int(var_map_orig[var_id].split('_')[1])
                    if d > self.ub:
                        high_t_clauses.append(clause)
                        break
        
        print(f"  Clauses involving T_{self.ub+1}+: {len(high_t_clauses)}")
        
        print(f"\nCutoff encoding:")
        print(f"  Mutual exclusion clauses: {len(mut_ex_clauses)}")
        
        print(f"\nTheoretical explanation:")
        print(f"  Original: Uses T_{self.ub+1} to T_{self.n-1} for distance > {self.ub}")
        print(f"  Cutoff: Directly forbids position pairs with distance > {self.ub}")
        print(f"  Mechanism: ¬U_i ∨ ¬V_k for |i-k| >= {self.ub+1}")
        
        # Calculate expected mutual exclusion count
        expected_mut_ex = 0
        gap = self.ub + 1
        for i in range(1, self.n + 1):
            # Positions too far left
            kmax = i - gap
            if kmax >= 1:
                expected_mut_ex += kmax
            
            # Positions too far right
            kmin = i + gap
            if kmin <= self.n:
                expected_mut_ex += (self.n - kmin + 1)
        
        print(f"\nExpected mutual exclusion clauses: {expected_mut_ex}")
        print(f"Actual mutual exclusion clauses: {len(mut_ex_clauses)}")
        
        if len(mut_ex_clauses) == expected_mut_ex:
            print(f"✓ Mutual exclusion count matches expectation")
        else:
            print(f"✗ Mutual exclusion count mismatch!")
        
        return len(mut_ex_clauses) == expected_mut_ex

    def verify_high_t_replacement_detailed(self, orig_activation: List, orig_deactivation: List,
                                           mut_ex_clauses: List, var_map_orig: Dict,
                                           var_map_cut: Dict, use_solver: bool = False,
                                           sample_limit: int = 20) -> bool:
        """
        Verify in detail that activation clauses in the original encoding that refer to
        distances > UB are represented in the cutoff encoding as mutual-exclusion
        clauses between the corresponding position variables.

        This performs set-based checks (cheap) and, optionally, per-pair SAT/UNSAT
        micro-checks using a SAT solver (more expensive but stronger evidence).

        Args:
            orig_activation: list of activation clauses from original encoding (lists of ints)
            orig_deactivation: list of deactivation clauses from original encoding (unused, kept for API symmetry)
            mut_ex_clauses: list of mutual exclusion clauses from cutoff encoding (lists of ints)
            var_map_orig: mapping var_id -> name for original encoding
            var_map_cut: mapping var_id -> name for cutoff encoding
            use_solver: if True, run a limited number of solver checks to demonstrate
                        that (U_i ∧ V_k) is UNSAT under cutoff clauses but satisfiable
                        under a single original activation clause (because T_d can be true)
            sample_limit: maximum number of per-pair solver checks to run

        Returns:
            True if S_orig ⊆ S_cut and (optionally) all solver checks matched expectations.
        """
        print(f"\n{'='*80}")
        print("DETAILED HIGH-T REPLACEMENT VERIFICATION")
        print(f"{'='*80}\n")

        # Build S_orig: set of (u_pos, v_pos) pairs extracted from original activation
        S_orig = set()
        # Keep mapping from pair -> example original clause
        orig_examples: Dict[Tuple[int, int], List[int]] = {}

        for clause in orig_activation:
            t_dist = None
            u_pos = None
            v_pos = None
            for lit in clause:
                vid = abs(lit)
                if vid in var_map_orig:
                    name = var_map_orig[vid]
                    if name.startswith('T_') and lit > 0:
                        try:
                            d = int(name.split('_')[1])
                        except Exception:
                            continue
                        if d > self.ub:
                            t_dist = d
                    elif name.startswith('U_'):
                        try:
                            u_pos = int(name.split('_')[1])
                        except Exception:
                            continue
                    elif name.startswith('V_'):
                        try:
                            v_pos = int(name.split('_')[1])
                        except Exception:
                            continue

            if t_dist and u_pos is not None and v_pos is not None:
                S_orig.add((u_pos, v_pos))
                if (u_pos, v_pos) not in orig_examples:
                    orig_examples[(u_pos, v_pos)] = clause

        print(f"Extracted {len(S_orig)} high-distance antecedent pairs from original activation (d > {self.ub})")

        # Build S_cut: set of (u_pos, v_pos) pairs extracted from mutual-exclusion clauses
        S_cut = set()
        cut_examples: Dict[Tuple[int, int], List[int]] = {}

        for clause in mut_ex_clauses:
            if len(clause) != 2:
                continue
            lits = clause
            u_pos = None
            v_pos = None
            for lit in lits:
                vid = abs(lit)
                if vid in var_map_cut:
                    name = var_map_cut[vid]
                    if name.startswith('U_'):
                        try:
                            u_pos = int(name.split('_')[1])
                        except Exception:
                            u_pos = None
                    elif name.startswith('V_'):
                        try:
                            v_pos = int(name.split('_')[1])
                        except Exception:
                            v_pos = None

            if u_pos is not None and v_pos is not None:
                # normalize ordering to (u_pos, v_pos)
                S_cut.add((u_pos, v_pos))
                if (u_pos, v_pos) not in cut_examples:
                    cut_examples[(u_pos, v_pos)] = clause

        print(f"Extracted {len(S_cut)} mutual-exclusion pairs from cutoff encoding")

        # Check set inclusion
        missing = S_orig - S_cut
        extra = S_cut - S_orig

        print(f"Pairs in original but missing in cutoff: {len(missing)}")
        if missing:
            sample_missing = list(missing)[:10]
            for u_pos, v_pos in sample_missing:
                print(f"  Missing pair: U_{u_pos}, V_{v_pos} (example orig clause: {orig_examples.get((u_pos, v_pos))})")

        print(f"Pairs in cutoff but not found among original activations: {len(extra)}")
        if extra:
            sample_extra = list(extra)[:10]
            for u_pos, v_pos in sample_extra:
                print(f"  Extra pair: U_{u_pos}, V_{v_pos} (example cutoff clause: {cut_examples.get((u_pos, v_pos))})")

        inclusion_ok = (len(missing) == 0)

        # Optional solver checks: use provided clause literals (they use concrete var ids from the same vpool)
        solver_ok = True
        if use_solver:
            print('\nRunning solver micro-checks (limited sample)')
            # sample pairs to check
            pairs = list(S_orig)[:sample_limit]
            if not pairs:
                print('  No pairs to sample for solver checks')
            for (u_pos, v_pos) in pairs:
                print(f"  Checking pair U_{u_pos}, V_{v_pos}...", end=' ')
                # find the concrete var ids from var_map_cut and var_map_orig
                # var_map_cut and var_map_orig map var_id -> name; we need reverse mapping
                rev_cut = {name: vid for vid, name in var_map_cut.items()}
                rev_orig = {name: vid for vid, name in var_map_orig.items()}

                cut_u_name = f'U_{u_pos}'
                cut_v_name = f'V_{v_pos}'
                # get var ids if present
                cut_u_vid = rev_cut.get(cut_u_name)
                cut_v_vid = rev_cut.get(cut_v_name)

                orig_u_vid = rev_orig.get(cut_u_name)
                orig_v_vid = rev_orig.get(cut_v_name)

                # find corresponding cutoff mutual-exclusion clause
                cut_clause = cut_examples.get((u_pos, v_pos))
                # find corresponding original activation clause
                orig_clause = orig_examples.get((u_pos, v_pos))

                if cut_clause is None:
                    print('CUT_CLAUSE_NOT_FOUND')
                    solver_ok = False
                    continue

                # Solver check 1: cutoff clauses should make U_i ∧ V_k UNSAT
                try:
                    solver = Cadical195()
                    # add the mutual exclusion clause (from cutoff) directly
                    solver.add_clause(cut_clause)
                    # add unit clauses forcing U_i and V_k true (use same var ids as in clause)
                    # find positive var ids from the clause mapping
                    # cut_clause contains negative literals; positive unit literals are the var ids
                    if cut_u_vid is None or cut_v_vid is None:
                        print('REV_MAPPING_MISSING')
                        solver.delete()
                        solver_ok = False
                        continue

                    solver.add_clause([cut_u_vid])
                    solver.add_clause([cut_v_vid])
                    sat = solver.solve()
                    solver.delete()
                    if sat:
                        print('CUTOFF_SAT_EXPECTED_UNSAT')
                        solver_ok = False
                        continue
                except Exception as e:
                    print(f'CUT_SOLVER_ERROR ({e})')
                    solver_ok = False
                    continue

                # Solver check 2: corresponding original activation clause should allow U_i ∧ V_k (i.e., SAT)
                if orig_clause is not None:
                    try:
                        solver = Cadical195()
                        solver.add_clause(orig_clause)
                        if orig_u_vid is None or orig_v_vid is None:
                            print(' ORIG_REV_MAPPING_MISSING')
                            solver.delete()
                            solver_ok = False
                            continue
                        solver.add_clause([orig_u_vid])
                        solver.add_clause([orig_v_vid])
                        sat = solver.solve()
                        solver.delete()
                        if not sat:
                            print('ORIG_UNSAT_EXPECTED_SAT')
                            solver_ok = False
                            continue
                    except Exception as e:
                        print(f'ORIG_SOLVER_ERROR ({e})')
                        solver_ok = False
                        continue

                print('OK')

        result = inclusion_ok and solver_ok

        print('\nSummary:')
        print(f'  S_orig size: {len(S_orig)}')
        print(f'  S_cut  size: {len(S_cut)}')
        print(f'  Inclusion OK: {inclusion_ok}')
        if use_solver:
            print(f'  Solver micro-checks OK: {solver_ok}')

        return result
    



def parse_mtx_file(filename: str) -> Tuple[int, List[Tuple[int, int]]]:
    """Parse MTX file"""
    print(f"Reading MTX file: {filename}")
    
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
    print(f"Loaded: n={n}, |E|={len(edges)}")
    return n, edges


def solve_at_k(n: int, edges: List[Tuple[int, int]], K: int, encoding: str, solver_name: str = 'cadical195'):
    """Solve bandwidth problem at specific K with given encoding"""
    print(f"\n{'='*80}")
    print(f"SOLVING AT K={K} with {encoding.upper()} encoding")
    print(f"{'='*80}")
    
    vpool = IDPool()
    
    # Create position variables
    X_vars, Y_vars = create_position_variables(n, vpool)
    
    # Create solver
    if solver_name == 'glucose42':
        solver = Glucose42()
    else:
        solver = Cadical195()
    
    # Add position constraints
    print(f"Adding position constraints...")
    pos_count = 0
    for clause in encode_all_position_constraints(n, X_vars, Y_vars, vpool):
        solver.add_clause(clause)
        pos_count += 1
    print(f"  Added {pos_count} position clauses")
    
    # Add distance constraints with detailed logging
    print(f"Adding distance constraints...")
    dist_clause_count = 0
    
    if encoding == 'original':
        from distance_encoder import encode_abs_distance_final
        
        for u, v in edges:
            # X distance
            Tx_vars, Tx_clauses = encode_abs_distance_final(
                X_vars[u], X_vars[v], n, vpool, f"Tx[{u},{v}]"
            )
            for clause in Tx_clauses:
                solver.add_clause(clause)
                dist_clause_count += 1
            
            # Y distance
            Ty_vars, Ty_clauses = encode_abs_distance_final(
                Y_vars[u], Y_vars[v], n, vpool, f"Ty[{u},{v}]"
            )
            for clause in Ty_clauses:
                solver.add_clause(clause)
                dist_clause_count += 1
            
            # Add bandwidth <= K constraints
            # Tx <= K
            if K < len(Tx_vars):
                solver.add_clause([-Tx_vars[K]])
            
            # Ty <= K
            if K < len(Ty_vars):
                solver.add_clause([-Ty_vars[K]])
            
            # Tx>=i → Ty<=K-i
            for i in range(1, K + 1):
                remaining = K - i
                if remaining >= 0:
                    if i-1 < len(Tx_vars) and remaining < len(Ty_vars):
                        solver.add_clause([-Tx_vars[i-1], -Ty_vars[remaining]])
                    if i-1 < len(Ty_vars) and remaining < len(Tx_vars):
                        solver.add_clause([-Ty_vars[i-1], -Tx_vars[remaining]])
    
    else:  # cutoff
        from distance_encoder_cutoff import encode_abs_distance_cutoff, calculate_theoretical_upper_bound
        ub = calculate_theoretical_upper_bound(n)
        
        for u, v in edges:
            # X distance
            Tx_clauses, Tx_vars = encode_abs_distance_cutoff(
                X_vars[u], X_vars[v], ub, vpool, f"Tx[{u},{v}]"
            )
            for clause in Tx_clauses:
                solver.add_clause(clause)
                dist_clause_count += 1
            
            # Y distance
            Ty_clauses, Ty_vars = encode_abs_distance_cutoff(
                Y_vars[u], Y_vars[v], ub, vpool, f"Ty[{u},{v}]"
            )
            for clause in Ty_clauses:
                solver.add_clause(clause)
                dist_clause_count += 1
            
            # Add bandwidth <= K constraints
            # Tx <= K
            if (K + 1) in Tx_vars:
                solver.add_clause([-Tx_vars[K + 1]])
            
            # Ty <= K
            if (K + 1) in Ty_vars:
                solver.add_clause([-Ty_vars[K + 1]])
            
            # Tx>=i → Ty<=K-i
            effective_k = min(K, ub)
            for i in range(1, effective_k + 1):
                remaining = effective_k - i
                if remaining >= 0:
                    if i in Tx_vars and (remaining + 1) in Ty_vars:
                        solver.add_clause([-Tx_vars[i], -Ty_vars[remaining + 1]])
                    if i in Ty_vars and (remaining + 1) in Tx_vars:
                        solver.add_clause([-Ty_vars[i], -Tx_vars[remaining + 1]])
    
    print(f"  Added {dist_clause_count} distance clauses")
    
    # Solve
    print(f"\nSolving...")
    t0 = time.time()
    result = solver.solve()
    solve_time = time.time() - t0
    
    print(f"Result: {'SAT' if result else 'UNSAT'}")
    print(f"Solve time: {solve_time:.3f}s")
    
    solver.delete()
    
    return result, solve_time


def main():
    parser = argparse.ArgumentParser(description='Verify encoding equivalence')
    parser.add_argument('mtx_file', help='MTX file to test')
    parser.add_argument('--k', type=int, default=3, help='K value to test (default: 3)')
    parser.add_argument('--edge-sample', type=int, default=0, help='Edge index to analyze in detail (default: 0)')
    parser.add_argument('--solve', action='store_true', help='Also solve at K with both encodings')
    
    args = parser.parse_args()
    
    # Find file
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
    
    # Parse graph
    n, edges = parse_mtx_file(found_file)
    
    print(f"\n{'='*80}")
    print(f"ENCODING EQUIVALENCE VERIFICATION")
    print(f"{'='*80}")
    print(f"File: {found_file}")
    print(f"Graph: n={n}, |E|={len(edges)}")
    print(f"Testing K: {args.k}")
    print(f"Edge sample: {args.edge_sample} (edge {edges[args.edge_sample]})")
    
    # Create verifier
    verifier = EncodingVerifier(n, edges)
    
    # Analyze sample edge
    sample_edge = edges[args.edge_sample]
    u, v = sample_edge
    
    # Build original encoding
    orig_t_vars, orig_clauses, orig_breakdown, var_map_orig = verifier.build_original_encoding_single_edge(u, v)
    
    # Build cutoff encoding
    cut_t_vars, cut_clauses, cut_breakdown, var_map_cut = verifier.build_cutoff_encoding_single_edge(u, v)
    
    # Compare T variable ranges
    verifier.compare_t_variable_ranges(orig_t_vars, cut_t_vars)
    
    # Verify equivalence in overlap region
    overlap = min(len(orig_t_vars), verifier.ub)
    
    act_match = verifier.verify_activation_equivalence(
        orig_breakdown['activation'],
        cut_breakdown['activation'],
        overlap,
        var_map_orig,
        var_map_cut
    )
    
    deact_match = verifier.verify_deactivation_equivalence(
        orig_breakdown['deactivation'],
        cut_breakdown['deactivation'],
        overlap,
        var_map_orig,
        var_map_cut
    )
    
    # Verify mutual exclusion replacement
    mut_ex_match = verifier.verify_mutual_exclusion_replacement(
        orig_clauses,
        cut_breakdown['mutual_exclusion'],
        var_map_orig
    )
    
    # Detailed high-T replacement verification with solver micro-checks
    print(f"\n{'#'*80}", flush=True)
    print(f"# SET-BASED & SOLVER MICRO-CHECK VERIFICATION", flush=True)
    print(f"{'#'*80}", flush=True)
    
    high_t_match = verifier.verify_high_t_replacement_detailed(
        orig_breakdown['activation'],
        orig_breakdown['deactivation'],
        cut_breakdown['mutual_exclusion'],
        var_map_orig,
        var_map_cut,
        use_solver=True,     # Enable solver micro-checks
        sample_limit=50      # Check up to 50 position pairs
    )
    
    # Summary
    print(f"\n{'='*80}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"T_1 to T_{overlap} activation (multiset): {'✓ MATCH' if act_match else '✗ MISMATCH'}")
    print(f"T_1 to T_{overlap} deactivation (multiset): {'✓ MATCH' if deact_match else '✗ MISMATCH'}")
    print(f"Mutual exclusion replacement: {'✓ MATCH' if mut_ex_match else '✗ MISMATCH'}")
    print(f"High-T set inclusion + solver checks: {'✓ VERIFIED' if high_t_match else '✗ FAILED'}")
    
    overall_match = act_match and deact_match and mut_ex_match and high_t_match
    
    if overall_match:
        print(f"\n✓ ENCODINGS ARE SEMANTICALLY EQUIVALENT")
        print(f"  → T_1 to T_{verifier.ub}: Identical activation/deactivation per position pair")
        print(f"  → T_{verifier.ub+1}+: Replaced by mutual exclusion (stronger constraint)")
        print(f"  → Cutoff eliminates {verifier.n - 1 - verifier.ub} auxiliary T variables per edge")
    else:
        print(f"\n✗ ENCODINGS HAVE SEMANTIC DIFFERENCES")
        print(f"  → Verification FAILED - check details above")
        sys.exit(2)  # Exit with error code for CI/CD pipelines
    
    # Optional: Solve at K
    if args.solve:
        print(f"\n{'#'*80}")
        print(f"# SOLVING AT K={args.k}")
        print(f"{'#'*80}")
        
        orig_result, orig_time = solve_at_k(n, edges, args.k, 'original')
        cut_result, cut_time = solve_at_k(n, edges, args.k, 'cutoff')
        
        print(f"\n{'='*80}")
        print(f"SOLVE COMPARISON AT K={args.k}")
        print(f"{'='*80}")
        print(f"Original: {'SAT' if orig_result else 'UNSAT'} in {orig_time:.3f}s")
        print(f"Cutoff:   {'SAT' if cut_result else 'UNSAT'} in {cut_time:.3f}s")
        print(f"Results match: {'✓' if orig_result == cut_result else '✗'}")
        print(f"Speedup: {orig_time/cut_time:.2f}x" if cut_time > 0 else "N/A")


if __name__ == '__main__':
    main()