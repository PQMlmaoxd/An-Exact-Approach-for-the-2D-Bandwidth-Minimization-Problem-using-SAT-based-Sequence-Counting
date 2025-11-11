#!/usr/bin/env python3
# quick_test_hybrid_cutoff.py
# Quick test to verify clause count match

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distance_encoder_cutoff import encode_abs_distance_cutoff, calculate_theoretical_upper_bound
from distance_encoder_hybrid import encode_abs_distance_hybrid
from pysat.formula import IDPool

n = 10
UB = calculate_theoretical_upper_bound(n)
max_repl = n - 1 - UB

print(f"Quick test: n={n}, UB={UB}, max_repl={max_repl}")

# Cutoff
vpool_c = IDPool()
U_c = [vpool_c.id(f'U_{i}') for i in range(1, n + 1)]
V_c = [vpool_c.id(f'V_{i}') for i in range(1, n + 1)]
clauses_c, t_c = encode_abs_distance_cutoff(U_c, V_c, UB, vpool_c, "T_c")

# Hybrid
vpool_h = IDPool()
U_h = [vpool_h.id(f'U_{i}') for i in range(1, n + 1)]
V_h = [vpool_h.id(f'V_{i}') for i in range(1, n + 1)]
clauses_h, t_h = encode_abs_distance_hybrid(U_h, V_h, n, UB, vpool_h, "T_h", max_repl)

print(f"\nCutoff: {len(clauses_c)} clauses, {len(t_c)} T vars")
print(f"Hybrid: {len(clauses_h)} clauses, {len(t_h)} T vars")
print(f"Diff: {len(clauses_h) - len(clauses_c)} clauses")

if len(clauses_h) != len(clauses_c):
    print(f"\n⚠ PROBLEM: Clause counts don't match!")
    print(f"Hybrid has {len(clauses_h) - len(clauses_c)} extra clauses")
else:
    print(f"\n✓ Clause counts match!")
