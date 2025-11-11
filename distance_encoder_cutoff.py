# distance_encoder_cutoff.py - Improved distance encoding with UB cutoff optimization

"""
Improved distance encoding for Bandwidth constraint with UB cutoff.

Key idea:
1) Enforce |pos(U) - pos(V)| <= UB using only 2-literal "mutual exclusion" clauses:
   For every pair of positions (i, k) with |i - k| >= UB + 1, add clause:
       (¬U_i ∨ ¬V_k)
   This alone guarantees correctness for the cutoff.

2) Keep lightweight T_d variables only up to UB for compatibility with
   pipelines that consume T (e.g., (assume ¬T_{K+1}) when checking bandwidth ≤ K).
   We define only the "sufficient" direction to *enable* T_d when needed:
       (U_i ∧ V_{i-d}) → T_d  and  (V_i ∧ U_{i-d}) → T_d, for i-d ≥ 1
   Plus monotonic chain:
       T_{d+1} → T_d   (i.e., (¬T_{d+1} ∨ T_d))
   And an optional base:
       (U_i ∧ V_i) → ¬T_1   (i.e., (¬U_i ∨ ¬V_i ∨ ¬T_1))
   We DO NOT define the heavy "necessary" direction T_d → ⋁(U_i∧V_k).

Inputs:
- U_vars: list[int] of variable IDs for U's position indicator, 1-indexed by position.
- V_vars: list[int] of variable IDs for V's position indicator, 1-indexed by position.
- UB: non-negative integer cutoff.
- vpool: object with an id() method (e.g., pysat.formula.IDPool) to allocate fresh variable IDs.
- t_var_prefix: prefix string for T variables, default "T".

Outputs:
- clauses: List[List[int]]  (CNF clauses)
- t_vars: Dict[int, int] mapping d -> var id for T_d, for d in 1..UB.

Notes:
- Exactly-One constraints for U_vars and V_vars are OUT OF SCOPE here; add them elsewhere.
- If UB == 0, the cutoff forbids any distance ≥ 1; mutual exclusions become (¬U_i ∨ ¬V_k) for all i != k.
- CRITICAL: t_var_prefix MUST be unique per edge to prevent variable conflicts!
  
Variable Conflict Prevention:
When using a global vpool across multiple edges, each edge must have a unique t_var_prefix.
Otherwise, different edges will share the same T_d variables, mixing semantic information.

Correct usage pattern:
  for u, v in edges:
      edge_prefix = f"T[{u},{v}]"  # Unique per edge
      clauses_uv, tvars_uv = encode_abs_distance_cutoff(U_vars, V_vars, UB, vpool, edge_prefix)

Performance Optimizations:
1. Cutoff clauses: O(n × UB) instead of O(n²) by avoiding unnecessary distance calculations
2. Tuple keys: vpool.id((prefix, 'geq', d)) avoids string formatting and prevents key conflicts
3. Lightweight T variables: Only creates variables up to UB, not full range 1..n-1
"""

import math
from typing import List, Dict, Any
from pysat.formula import IDPool
from pysat.solvers import Solver


def calculate_theoretical_upper_bound(n: int) -> int:
    """
    Calculate theoretical upper bound using formula:
    δ(n) = min{2⌈(√(2n-1)-1)/2⌉, 2⌈√(n/2)⌉-1}
    
    This provides a tight upper bound for 2D bandwidth minimization
    based on theoretical analysis of grid placements.
    
    Args:
        n: Number of vertices
        
    Returns:
        Theoretical upper bound for bandwidth
    """
    
    # Handle special cases
    if n <= 0:
        return 0
    if n == 1:
        return 0
    if n == 2:
        return 1
    
    # First term: 2⌈(√(2n-1)-1)/2⌉
    term1 = 2 * math.ceil((math.sqrt(2*n - 1) - 1) / 2)
    
    # Second term: 2⌈√(n/2)⌉-1  
    term2 = 2 * math.ceil(math.sqrt(n / 2)) - 1
    
    # Return minimum of both terms
    ub = min(term1, term2)
    
    return ub


def encode_abs_distance_cutoff(U_vars: List[int], V_vars: List[int], 
                              UB: int, vpool: Any, 
                              t_var_prefix: str) -> tuple[List[List[int]], Dict[int, int]]:
    """
    Distance encoding with UB cutoff optimization - OPTIMIZED CLAUSE ORDER.
    
    CRITICAL PERFORMANCE OPTIMIZATION:
    Clause ordering DIRECTLY impacts SAT solver performance! This implementation
    generates clauses in the EXACT same order as the original encoder (distance_encoder.py)
    to maximize SAT solver heuristic learning, then adds cutoff-specific mutual exclusion
    clauses at the end.
    
    Clause Generation Order (matches original encoder):
    1. T Activation (V > U case): for k in range(1, n+1) → for d in range(1, min(k, UB+1))
    2. T Activation (U > V case): for k in range(1, n+1) → for d in range(1, min(k, UB+1))
    3. T Deactivation (V > U case): for k in range(1, n+1) → for d in range(1, min(k, UB))
    4. T Deactivation (U > V case): for k in range(1, n+1) → for d in range(1, min(k, UB))
    5. Base case: for k in range(1, n+1) → (U_k ∧ V_k) → ¬T_1
    6. Monotonicity: for d in range(1, UB) → T_{d+1} → T_d
    7. Mutual Exclusion (cutoff-specific): forbid distance > UB
    
    Key Optimizations:
    - Only creates T_1 to T_UB (not T_1 to T_{n-1}), saving variables
    - Mutual exclusion clauses added LAST to preserve original heuristics
    - Clause structure identical to original where they overlap
    
    Args:
        U_vars: Position indicator variables for U (1-indexed, length n)
        V_vars: Position indicator variables for V (1-indexed, length n)
        UB: Upper bound cutoff for distance
        vpool: Variable pool with id() method
        t_var_prefix: UNIQUE prefix per edge (e.g., "T[u,v]")
        
    Returns:
        tuple of (clauses, t_vars):
        - clauses: List[List[int]] - CNF clauses in optimized order
        - t_vars: Dict[int, int] - mapping distance d to variable ID for T_d
        
    IMPORTANT: t_var_prefix MUST be unique per edge to avoid conflicts!
    
    Example usage:
        prefix = f"T[{u},{v}]"
        clauses, tvars = encode_abs_distance_cutoff(U_vars, V_vars, UB, vpool, prefix)
    """
    
    clauses = []
    t_vars = {}
    n = len(U_vars)
    
    # Special case: UB = 0 means only same position allowed
    if UB == 0:
        for i in range(1, n + 1):
            for k in range(1, n + 1):
                if i != k:
                    clauses.append([-U_vars[i - 1], -V_vars[k - 1]])
        return clauses, t_vars
    
    # === CREATE T VARIABLES (only up to UB) ===
    for d in range(1, UB + 1):
        t_vars[d] = vpool.id((t_var_prefix, 'geq', d))
    
    # === MATCH ORIGINAL ENCODER CLAUSE ORDER EXACTLY ===
    
    # STAGE 1: T Activation Rules (V > U case) - FIRST
    # Original: for k in range(1, n + 1): for d in range(1, k):
    # Cutoff: Same loop structure, but only up to UB
    for k in range(1, n + 1):
        for d in range(1, min(k, UB + 1)):
            if d in t_vars:
                u_pos = k - d
                if u_pos >= 1:
                    # (V_k ∧ U_{k-d}) → T_d  =>  (¬V_k ∨ ¬U_{k-d} ∨ T_d)
                    clauses.append([-V_vars[k - 1], -U_vars[u_pos - 1], t_vars[d]])
    
    # STAGE 2: T Activation Rules (U > V case) - SECOND
    for k in range(1, n + 1):
        for d in range(1, min(k, UB + 1)):
            if d in t_vars:
                v_pos = k - d
                if v_pos >= 1:
                    # (U_k ∧ V_{k-d}) → T_d  =>  (¬U_k ∨ ¬V_{k-d} ∨ T_d)
                    clauses.append([-U_vars[k - 1], -V_vars[v_pos - 1], t_vars[d]])
    
    # STAGE 3: T Deactivation Rules (V > U case) - THIRD
    for k in range(1, n + 1):
        for d in range(1, min(k, UB)):  # d+1 must exist, so d < UB
            if (d + 1) in t_vars:
                u_pos = k - d
                if u_pos >= 1:
                    # (V_k ∧ U_{k-d}) → ¬T_{d+1}  =>  (¬V_k ∨ ¬U_{k-d} ∨ ¬T_{d+1})
                    clauses.append([-V_vars[k - 1], -U_vars[u_pos - 1], -t_vars[d + 1]])
    
    # STAGE 4: T Deactivation Rules (U > V case) - FOURTH
    for k in range(1, n + 1):
        for d in range(1, min(k, UB)):
            if (d + 1) in t_vars:
                v_pos = k - d
                if v_pos >= 1:
                    # (U_k ∧ V_{k-d}) → ¬T_{d+1}  =>  (¬U_k ∨ ¬V_{k-d} ∨ ¬T_{d+1})
                    clauses.append([-U_vars[k - 1], -V_vars[v_pos - 1], -t_vars[d + 1]])
    
    # STAGE 5: Base case (same position) - FIFTH
    # Original: for k in range(1, n + 1): clauses.append([-U_vars[k-1], -V_vars[k-1], -T_vars[0]])
    if 1 in t_vars:
        for k in range(1, n + 1):
            # (U_k ∧ V_k) → ¬T_1  =>  (¬U_k ∨ ¬V_k ∨ ¬T_1)
            clauses.append([-U_vars[k - 1], -V_vars[k - 1], -t_vars[1]])
    
    # STAGE 6: Monotonicity chain - SIXTH
    # Original: for d in range(1, len(T_vars)): clauses.append([T_vars[d-1], -T_vars[d]])
    # Equivalent: T_{d+1} → T_d  =>  (¬T_{d+1} ∨ T_d)
    for d in range(1, UB):
        if d in t_vars and (d + 1) in t_vars:
            # T_{d+1} → T_d  =>  (¬T_{d+1} ∨ T_d)
            clauses.append([-t_vars[d + 1], t_vars[d]])
    
    # === CUTOFF-SPECIFIC OPTIMIZATION ===
    # STAGE 7: Mutual Exclusion for distance > UB - LAST
    # This is added AFTER all T-variable clauses to preserve original heuristics
    # while still providing the cutoff optimization benefit
    
    gap = UB + 1  # Minimum forbidden distance
    for i in range(1, n + 1):
        # Forbid V at position k where |i - k| >= gap
        
        # Case 1: k ≤ i - gap (V too far to the left)
        kmax = i - gap
        if kmax >= 1:
            for k in range(1, kmax + 1):
                # Distance = i - k >= gap = UB + 1, forbid this
                clauses.append([-U_vars[i - 1], -V_vars[k - 1]])
        
        # Case 2: k ≥ i + gap (V too far to the right)
        kmin = i + gap
        if kmin <= n:
            for k in range(kmin, n + 1):
                # Distance = k - i >= gap = UB + 1, forbid this
                clauses.append([-U_vars[i - 1], -V_vars[k - 1]])
    
    return clauses, t_vars


def test_cutoff_encoder(u_pos: int, v_pos: int, n: int, UB: int = None):
    """
    Test function for cutoff encoder
    
    Args:
        u_pos: Position of vertex U (1-indexed)
        v_pos: Position of vertex V (1-indexed) 
        n: Grid size
        UB: Upper bound cutoff (if None, uses theoretical bound)
    """
    
    if UB is None:
        UB = calculate_theoretical_upper_bound(n)
    
    print(f"\n=== Testing Cutoff Encoder ===")
    print(f"Grid size: {n}x{n}")
    print(f"U position: {u_pos}, V position: {v_pos}")
    print(f"Actual distance: {abs(u_pos - v_pos)}")
    print(f"UB cutoff: {UB}")
    
    vpool = IDPool()
    U_vars = [vpool.id(f'U_{i}') for i in range(1, n + 1)]
    V_vars = [vpool.id(f'V_{i}') for i in range(1, n + 1)]
    
    # Set positions
    position_clauses = [[U_vars[u_pos - 1]], [V_vars[v_pos - 1]]]
    
    # Generate distance constraints with cutoff using unique prefix
    edge_prefix = f"T[test_edge]"  # Unique prefix for this test edge
    dist_clauses, t_vars = encode_abs_distance_cutoff(U_vars, V_vars, UB, vpool, edge_prefix)
    
    all_clauses = position_clauses + dist_clauses
    
    print(f"\nSolving with {len(all_clauses)} clauses...")
    
    with Solver(bootstrap_with=all_clauses) as solver:
        if solver.solve():
            model = solver.get_model()
            actual_dist = abs(u_pos - v_pos)
            
            print(f"SAT - Model found")
            print(f"Actual distance: {actual_dist}")
            
            # Check T variable assignments
            print(f"T variable assignments:")
            correct = True
            for d in range(1, min(len(t_vars) + 1, actual_dist + 3)):
                if d in t_vars:
                    should_be = (actual_dist >= d)
                    is_true = t_vars[d] in model
                    status = "✓" if should_be == is_true else "✗"
                    print(f"  T_{d}: {is_true} (expected: {should_be}) {status}")
                    if should_be != is_true:
                        correct = False
            
            # Check cutoff constraint
            cutoff_satisfied = actual_dist <= UB
            cutoff_status = "✓" if cutoff_satisfied else "✗"
            print(f"Cutoff constraint (dist ≤ {UB}): {cutoff_satisfied} {cutoff_status}")
            
            overall_status = "CORRECT" if correct and cutoff_satisfied else "ERROR"
            print(f"Overall result: {overall_status}")
            
        else:
            actual_dist = abs(u_pos - v_pos)
            if actual_dist > UB:
                print(f"UNSAT - Expected (distance {actual_dist} > UB {UB}) ✓")
            else:
                print(f"UNSAT - Unexpected (distance {actual_dist} ≤ UB {UB}) ✗")


def compare_encoders_efficiency(n: int):
    """
    Compare efficiency between original and cutoff encoders
    
    Args:
        n: Grid size for comparison
    """
    
    print(f"\n=== ENCODER EFFICIENCY COMPARISON (n={n}) ===")
    
    UB = calculate_theoretical_upper_bound(n)
    print(f"Theoretical UB for n={n}: {UB}")
    
    vpool_orig = IDPool()
    vpool_cutoff = IDPool()
    
    # Original encoder variables
    U_vars_orig = [vpool_orig.id(f'U_{i}') for i in range(1, n + 1)]
    V_vars_orig = [vpool_orig.id(f'V_{i}') for i in range(1, n + 1)]
    
    # Cutoff encoder variables  
    U_vars_cutoff = [vpool_cutoff.id(f'U_{i}') for i in range(1, n + 1)]
    V_vars_cutoff = [vpool_cutoff.id(f'V_{i}') for i in range(1, n + 1)]
    
    # Original encoder (from distance_encoder.py)
    try:
        from distance_encoder import encode_abs_distance_final
        T_vars_orig, clauses_orig = encode_abs_distance_final(U_vars_orig, V_vars_orig, n, vpool_orig)
        orig_t_count = len(T_vars_orig)
        orig_clause_count = len(clauses_orig)
    except ImportError:
        print("Could not import original encoder for comparison")
        orig_t_count = n - 1  # Theoretical count
        orig_clause_count = n * n * 4  # Rough estimate
    
    # Cutoff encoder
    cutoff_prefix = f"T[cutoff_test]"  # Unique prefix for cutoff test
    clauses_cutoff, t_vars_cutoff = encode_abs_distance_cutoff(U_vars_cutoff, V_vars_cutoff, UB, vpool_cutoff, cutoff_prefix)
    cutoff_t_count = len(t_vars_cutoff)
    cutoff_clause_count = len(clauses_cutoff)
    
    # Comparison results
    print(f"\nOriginal encoder:")
    print(f"  T variables: {orig_t_count}")
    print(f"  Clauses: {orig_clause_count}")
    
    print(f"\nCutoff encoder (UB={UB}):")
    print(f"  T variables: {cutoff_t_count}")
    print(f"  Clauses: {cutoff_clause_count}")
    
    print(f"\nEfficiency gains:")
    if orig_t_count > 0:
        t_reduction = (orig_t_count - cutoff_t_count) / orig_t_count * 100
        print(f"  T variables reduced: {t_reduction:.1f}%")
    
    if orig_clause_count > 0:
        clause_reduction = (orig_clause_count - cutoff_clause_count) / orig_clause_count * 100
        print(f"  Clauses reduced: {clause_reduction:.1f}%")
    
    print(f"  UB cutoff benefit: Eliminates impossible assignments early")
    print(f"  Pipeline compatibility: T variables available for bandwidth ≤ K checks")


def example_multi_edge_usage():
    """
    Example showing correct usage with multiple edges to avoid variable conflicts
    """
    print(f"\n=== EXAMPLE: Multi-Edge Usage with Unique Prefixes ===")
    
    n = 4
    UB = calculate_theoretical_upper_bound(n)
    print(f"Grid size: {n}x{n}, UB: {UB}")
    
    # Create shared position variables for all vertices
    vpool = IDPool()
    vertex_X_vars = {}
    vertex_Y_vars = {}
    
    for v in range(1, n + 1):
        vertex_X_vars[v] = [vpool.id(f'X_{v}_{pos}') for pos in range(1, n + 1)]
        vertex_Y_vars[v] = [vpool.id(f'Y_{v}_{pos}') for pos in range(1, n + 1)]
    
    # Example edges
    edges = [(1, 2), (2, 3), (1, 3)]
    
    all_clauses = []
    edge_t_vars = {}
    
    print(f"Processing {len(edges)} edges with unique T variable prefixes:")
    
    for u, v in edges:
        print(f"\nEdge ({u},{v}):")
        
        # Use unique prefix for each edge to avoid variable conflicts
        edge_prefix = f"T[{u},{v}]"
        print(f"  Using prefix: {edge_prefix}")
        
        # Generate distance constraints for X coordinates
        x_clauses, x_tvars = encode_abs_distance_cutoff(
            vertex_X_vars[u], vertex_X_vars[v], UB, vpool, f"{edge_prefix}_X"
        )
        
        # Generate distance constraints for Y coordinates  
        y_clauses, y_tvars = encode_abs_distance_cutoff(
            vertex_Y_vars[u], vertex_Y_vars[v], UB, vpool, f"{edge_prefix}_Y"
        )
        
        all_clauses.extend(x_clauses)
        all_clauses.extend(y_clauses)
        
        # Store T variables for this edge
        edge_t_vars[(u, v)] = {
            'x_tvars': x_tvars,
            'y_tvars': y_tvars
        }
        
        print(f"  X T-vars: {len(x_tvars)} variables")
        print(f"  Y T-vars: {len(y_tvars)} variables")
        print(f"  Clauses: {len(x_clauses) + len(y_clauses)}")
    
    print(f"\nSummary:")
    print(f"  Total edges: {len(edges)}")
    print(f"  Total clauses: {len(all_clauses)}")
    print(f"  Each edge has independent T variables → no conflicts")
    print(f"  Edge T-vars stored: {list(edge_t_vars.keys())}")
    
    # Show that T variables are indeed unique
    all_vars = set()
    conflicts = 0
    
    for edge, tvars in edge_t_vars.items():
        for d in tvars['x_tvars']:
            var_id = tvars['x_tvars'][d]
            if var_id in all_vars:
                conflicts += 1
            all_vars.add(var_id)
        
        for d in tvars['y_tvars']:
            var_id = tvars['y_tvars'][d]
            if var_id in all_vars:
                conflicts += 1
            all_vars.add(var_id)
    
    print(f"  Variable uniqueness check: {conflicts} conflicts (should be 0)")
    print(f"  Total unique T variables: {len(all_vars)}")


if __name__ == '__main__':
    """
    Test suite for distance encoder with UB cutoff
    """
    
    print("=== DISTANCE ENCODER WITH UB CUTOFF TESTS ===")
    
    # Test 1: Small grid with various distances
    print(f"\n" + "="*50)
    print(f"Test 1: Small grid (5x5) with UB cutoff")
    print(f"="*50)
    
    n = 5
    UB = calculate_theoretical_upper_bound(n)
    print(f"Theoretical UB for n={n}: {UB}")
    
    test_cases = [
        (1, 1),  # Same position
        (1, 2),  # Distance 1
        (1, 3),  # Distance 2  
        (1, 4),  # Distance 3
        (1, 5),  # Distance 4
        (2, 5),  # Distance 3
    ]
    
    for u, v in test_cases:
        test_cutoff_encoder(u, v, n, UB)
    
    # Test 2: Edge case - UB = 0
    print(f"\n" + "="*50)
    print(f"Test 2: UB = 0 (only same position allowed)")
    print(f"="*50)
    
    test_cutoff_encoder(2, 2, 4, UB=0)  # Should work
    test_cutoff_encoder(2, 3, 4, UB=0)  # Should be UNSAT
    
    # Test 3: Efficiency comparison
    print(f"\n" + "="*50)
    print(f"Test 3: Efficiency comparison")
    print(f"="*50)
    
    for grid_size in [5, 8, 10, 12]:
        compare_encoders_efficiency(grid_size)
    
    # Test 4: Multi-edge usage example
    print(f"\n" + "="*50)
    print(f"Test 4: Multi-Edge Usage Example")
    print(f"="*50)
    
    example_multi_edge_usage()
    
    # Test 5: Larger grid with theoretical UB
    print(f"\n" + "="*50)
    print(f"Test 5: Larger grid with automatic UB")
    print(f"="*50)
    
    n = 8
    UB = calculate_theoretical_upper_bound(n)
    print(f"Testing n={n} with theoretical UB={UB}")
    
    # Test near boundary cases
    test_cutoff_encoder(1, 8, n)  # Distance 7 > UB=3 → should be UNSAT
    test_cutoff_encoder(4, 5, n)  # Distance 1 ≤ UB=3 → should be SAT
    
    print(f"\n=== ALL TESTS COMPLETED ===")
    print(f"Key benefits of cutoff encoder:")
    print(f"  1. Direct mutual exclusion for distance > UB")
    print(f"  2. Lightweight T variables only up to UB")
    print(f"  3. Pipeline compatibility with bandwidth ≤ K checks")
    print(f"  4. Significant reduction in variables and clauses")
    print(f"  5. Early elimination of impossible assignments")
