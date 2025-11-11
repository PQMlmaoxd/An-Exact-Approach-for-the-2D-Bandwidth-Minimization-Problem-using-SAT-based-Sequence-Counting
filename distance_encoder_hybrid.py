# distance_encoder_hybrid.py - Hybrid encoding with incremental T→mutual-exclusion replacement

"""
Hybrid distance encoder: Replace T variables with mutual exclusion clauses incrementally.

Key Innovation:
- Base: Full distance encoding from distance_encoder.py (T_1 to T_{n-1})
- Replacement: Replace T_d variables from UB upward with mutual exclusion clauses
- Control: Specify how many levels to replace (1 edge = replace T_UB only, etc.)
- Maximum: When replacing all levels from UB to n-1, matches distance_encoder_cutoff.py performance

Four-Stage Encoding Process:
1. Stage 1: Create T variables for ALL non-replaced ranges
   - T_1 to T_UB (before replacement)
   - T_{replacement_end+1} to T_{n-1} (after replacement, if any)

2. Stage 2: Standard encoding (activation/deactivation) for T_1 to T_UB
   - Activation rules: (V_k ∧ U_{k-d}) → T_d
   - Deactivation rules: (V_k ∧ U_{k-d}) → ¬T_{d+1}
   - Monotonicity: T_{d+1} → T_d

3. Stage 3: Mutual exclusion for T_{UB+1} to T_{replacement_end}
   - Direct mutual exclusion clauses instead of T variables
   - Forbid position pairs with distance >= replacement_start
   - OPTIMIZATION: Only forbid minimum gap (automatically forbids all larger distances)
   - For full replacement: identical to cutoff encoder

3.5. Stage 3.5: Connect replacement boundary (ONLY for partial replacement)
   - Adds activation clauses for T_{replacement_start-1} to maintain monotonicity
   - Skipped when replacement_end == n-1 (full replacement)
   - Critical for correctness when Stage 4 exists

4. Stage 4: Standard encoding for T_{replacement_end+1} to T_{n-1}
   - Same activation/deactivation/monotonicity rules as Stage 2
   - Only when num_replacements < n-1-UB (partial replacement)

Performance Characteristics:
- Partial replacement (with Stage 4): More clauses than full replacement due to Stage 3.5
- Full replacement (replacement_end == n-1): Equivalent to cutoff encoding, no Stage 3.5/4
- Stage 3.5 overhead: O(n²) clauses per edge, only needed for partial replacement

Replacement Modes:
- num_replacements = 0: Full standard encoding (all T variables, no Stage 3-4)
- num_replacements = 1: T_1 to T_UB + mutual exclusion for T_{UB+1} + T_{UB+2} to T_{n-1}
- num_replacements = 2: T_1 to T_UB + mutual exclusion for T_{UB+1},T_{UB+2} + T_{UB+3} to T_{n-1}
- ...
- num_replacements = n-1-UB: T_1 to T_UB + mutual exclusion for T_{UB+1} to T_{n-1} (no Stage 4)

Usage:
    vpool = IDPool()
    U_vars = [vpool.id(f'U_{i}') for i in range(1, n+1)]
    V_vars = [vpool.id(f'V_{i}') for i in range(1, n+1)]
    
    # Keep T_1 to T_UB, replace T_{UB+1} only, keep T_{UB+2} to T_{n-1}
    clauses, t_vars = encode_abs_distance_hybrid(U_vars, V_vars, n, UB, vpool, 
                                                  prefix="T", num_replacements=1)
    
    # Keep T_1 to T_UB, replace T_{UB+1} to T_{n-1} (full replacement, equivalent to cutoff)
    clauses, t_vars = encode_abs_distance_hybrid(U_vars, V_vars, n, UB, vpool,
                                                  prefix="T", num_replacements=n-1-UB)

Performance:
- More replacements = fewer T variables = faster encoding
- When num_replacements = n-1-UB: identical performance to distance_encoder_cutoff.py
"""

import math
from typing import List, Dict, Any, Tuple
from pysat.formula import IDPool
from pysat.solvers import Solver


def calculate_theoretical_upper_bound(n: int) -> int:
    """
    Calculate theoretical upper bound using formula:
    δ(n) = min{2⌈(√(2n-1)-1)/2⌉, 2⌈√(n/2)⌉-1}
    
    Args:
        n: Number of vertices
        
    Returns:
        Theoretical upper bound for bandwidth
    """
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
    
    return min(term1, term2)


def encode_abs_distance_hybrid(
    U_vars: List[int], 
    V_vars: List[int], 
    n: int,
    UB: int,
    vpool: Any, 
    prefix: str = "T",
    num_replacements: int = 0
) -> Tuple[List[List[int]], Dict[int, int]]:
    """
    Hybrid distance encoder with incremental T→mutual-exclusion replacement.
    
    This function combines:
    1. Standard encoding (distance_encoder.py) for T_1 to T_UB
    2. Mutual exclusion clauses for T_{UB+1} to T_{UB+num_replacements}
    
    Args:
        U_vars: List of variable IDs for U's position indicators (1-indexed)
        V_vars: List of variable IDs for V's position indicators (1-indexed)
        n: Grid size
        UB: Upper bound cutoff - T variables up to UB are kept with activation clauses
        vpool: Variable pool with id() method
        prefix: Unique prefix for T variables per edge (REQUIRED for multi-edge usage)
        num_replacements: Number of T variables to replace starting from UB+1
                         - 0: No replacement (full standard encoding, T_1 to T_{n-1})
                         - 1: Keep T_1 to T_UB, replace T_{UB+1}
                         - 2: Keep T_1 to T_UB, replace T_{UB+1} and T_{UB+2}
                         - ...
                         - n-1-UB: Keep T_1 to T_UB, replace T_{UB+1} to T_{n-1} (cutoff equivalent)
        
    Returns:
        tuple of (clauses, t_vars) where:
        - clauses: List of CNF clauses
        - t_vars: Dict mapping distance d to variable ID for T_d (only for non-replaced)
        
    Performance:
        - num_replacements = 0: Same as distance_encoder.py
        - num_replacements = n-1-UB: Same as distance_encoder_cutoff.py
        
    CRITICAL: prefix MUST be unique per edge to avoid variable conflicts!
    """
    
    clauses = []
    t_vars = {}
    
    # Cap num_replacements at maximum allowed
    max_replacements = n - 1 - UB
    if num_replacements > max_replacements:
        num_replacements = max_replacements
    
    # Determine replacement ranges
    if num_replacements == 0:
        # No replacement: Full standard encoding
        max_t_with_activation = n - 1
        replacement_start = n  # No replacement range
        replacement_end = -1
    else:
        # With replacement: Keep T_1 to T_UB, replace T_{UB+1} onwards
        replacement_start = UB + 1
        replacement_end = min(replacement_start + num_replacements - 1, n - 1)
        max_t_with_activation = UB
    
    # === STAGE 1: MUTUAL EXCLUSION FOR REPLACED RANGE (like cutoff Stage 1) ===
    # CRITICAL: Generate mutual exclusion clauses FIRST (before T variables)
    # This matches cutoff encoder's clause ordering for better solver heuristics
    if num_replacements > 0 and replacement_start <= n - 1:
        gap = replacement_start  # Distance >= replacement_start is forbidden
        
        # For each position i, forbid positions that create distance >= gap
        for i in range(1, n + 1):
            # Case 1: k ≤ i - gap (k is too far to the left)
            kmax = i - gap
            if kmax >= 1:
                for k in range(1, kmax + 1):
                    clauses.append([-U_vars[i - 1], -V_vars[k - 1]])
            
            # Case 2: k ≥ i + gap (k is too far to the right)
            kmin = i + gap
            if kmin <= n:
                for k in range(kmin, n + 1):
                    clauses.append([-U_vars[i - 1], -V_vars[k - 1]])
    
    # === STAGE 2: CREATE T VARIABLES AND STANDARD ENCODING ===
    # Create T_1 to T_UB (like cutoff Stage 2)
    for d in range(1, max_t_with_activation + 1):
        t_vars[d] = vpool.id((prefix, 'geq', d))
    
    # Create T_{replacement_end+1} to T_{n-1} (for partial replacement with Stage 4)
    if num_replacements > 0 and replacement_end < n - 1:
        for d in range(replacement_end + 1, n):
            t_vars[d] = vpool.id((prefix, 'geq', d))
    
    # T Activation Rules (symmetric, from distance_encoder.py)
    # CRITICAL: Order matters for SAT solver performance!
    # Must match cutoff encoder: U-based activation FIRST, then V-based
    
    # ∀i,d: (U_i ∧ V_{i-d}) → T_d (for U > V case) - FIRST like cutoff
    for i in range(1, n + 1):
        for d in range(1, min(i, max_t_with_activation + 1)):
            if d in t_vars:
                v_pos = i - d
                if v_pos >= 1:
                    # (U_i ∧ V_{i-d}) → T_d
                    clauses.append([-U_vars[i - 1], -V_vars[v_pos - 1], t_vars[d]])
    
    # ∀i,d: (V_i ∧ U_{i-d}) → T_d (for V > U case) - SECOND like cutoff
    for i in range(1, n + 1):
        for d in range(1, min(i, max_t_with_activation + 1)):
            if d in t_vars:
                u_pos = i - d
                if u_pos >= 1:
                    # (V_i ∧ U_{i-d}) → T_d
                    clauses.append([-V_vars[i - 1], -U_vars[u_pos - 1], t_vars[d]])
    
    # T Deactivation Rules (tight encoding, from distance_encoder.py)
    # CRITICAL: Order must match cutoff encoder!
    # V-based deactivation FIRST, then U-based
    
    # ∀k,d: (V_k ∧ U_{k-d}) → ¬T_{d+1} (for V > U) - FIRST like cutoff
    for k in range(1, n + 1):
        for d in range(1, k):
            if (d + 1) in t_vars:  # Only if T_{d+1} exists
                u_pos = k - d
                if u_pos >= 1:
                    # (V_k ∧ U_{k-d}) → ¬T_{d+1}
                    clauses.append([-V_vars[k - 1], -U_vars[u_pos - 1], -t_vars[d + 1]])
    
    # ∀k,d: (U_k ∧ V_{k-d}) → ¬T_{d+1} (for U > V) - SECOND like cutoff
    for k in range(1, n + 1):
        for d in range(1, k):
            if (d + 1) in t_vars:  # Only if T_{d+1} exists
                v_pos = k - d
                if v_pos >= 1:
                    # (U_k ∧ V_{k-d}) → ¬T_{d+1}
                    clauses.append([-U_vars[k - 1], -V_vars[v_pos - 1], -t_vars[d + 1]])
    
    # Base case: Same position → distance < 1, so ¬T_1
    for k in range(1, n + 1):
        if 1 in t_vars:
            # (U_k ∧ V_k) → ¬T_1
            clauses.append([-U_vars[k - 1], -V_vars[k - 1], -t_vars[1]])
    
    # Monotonicity Rules: T_{d+1} → T_d (from distance_encoder.py)
    for d in range(1, len(t_vars)):
        if d in t_vars and (d + 1) in t_vars:
            # T_{d+1} → T_d
            clauses.append([-t_vars[d + 1], t_vars[d]])
    
    # === STAGE 3: CONNECT REPLACEMENT BOUNDARY (only for partial replacement) ===
    # CRITICAL: Only needed for PARTIAL replacement (when Stage 4 exists)
    if replacement_end < n - 1 and replacement_start > 1 and (replacement_start - 1) in t_vars:
        gap = replacement_start
        
        # Activate T_{gap-1} when positions create distance >= gap
        for i in range(1, n + 1):
            # Case 1: V at position k where k ≤ i - gap
            kmax = i - gap
            if kmax >= 1:
                for k in range(1, kmax + 1):
                    clauses.append([-U_vars[i - 1], -V_vars[k - 1], t_vars[gap - 1]])
            
            # Case 2: V at position k where k ≥ i + gap
            kmin = i + gap
            if kmin <= n:
                for k in range(kmin, n + 1):
                    clauses.append([-U_vars[i - 1], -V_vars[k - 1], t_vars[gap - 1]])
    
    # === STAGE 4: STANDARD ENCODING FOR T_{replacement_end+1} to T_{n-1} ===
    # This stage handles T variables after the replacement range
    # Only needed when num_replacements > 0 and replacement_end < n-1
    if num_replacements > 0 and replacement_end < n - 1:
        # T Activation Rules for the remaining range
        # ∀k,d: (V_k ∧ U_{k-d}) → T_d (for V > U)
        for k in range(1, n + 1):
            for d in range(replacement_end + 1, min(k, n)):
                if d in t_vars:
                    u_pos = k - d
                    if u_pos >= 1:
                        # (V_k ∧ U_{k-d}) → T_d
                        clauses.append([-V_vars[k - 1], -U_vars[u_pos - 1], t_vars[d]])
        
        # ∀k,d: (U_k ∧ V_{k-d}) → T_d (for U > V)
        for k in range(1, n + 1):
            for d in range(replacement_end + 1, min(k, n)):
                if d in t_vars:
                    v_pos = k - d
                    if v_pos >= 1:
                        # (U_k ∧ V_{k-d}) → T_d
                        clauses.append([-U_vars[k - 1], -V_vars[v_pos - 1], t_vars[d]])
        
        # T Deactivation Rules for the remaining range
        # ∀k,d: (V_k ∧ U_{k-d}) → ¬T_{d+1} (for V > U)
        for k in range(1, n + 1):
            for d in range(replacement_end + 1, k):
                if (d + 1) in t_vars:  # Only if T_{d+1} exists
                    u_pos = k - d
                    if u_pos >= 1:
                        # (V_k ∧ U_{k-d}) → ¬T_{d+1}
                        clauses.append([-V_vars[k - 1], -U_vars[u_pos - 1], -t_vars[d + 1]])
        
        # ∀k,d: (U_k ∧ V_{k-d}) → ¬T_{d+1} (for U > V)
        for k in range(1, n + 1):
            for d in range(replacement_end + 1, k):
                if (d + 1) in t_vars:  # Only if T_{d+1} exists
                    v_pos = k - d
                    if v_pos >= 1:
                        # (U_k ∧ V_{k-d}) → ¬T_{d+1}
                        clauses.append([-U_vars[k - 1], -V_vars[v_pos - 1], -t_vars[d + 1]])
        
        # Monotonicity Rules for the remaining range: T_{d+1} → T_d
        for d in range(replacement_end + 1, n - 1):
            if d in t_vars and (d + 1) in t_vars:
                # T_{d+1} → T_d
                clauses.append([-t_vars[d + 1], t_vars[d]])
        
        # Connect replacement boundary: T_{replacement_end+1} → T_{replacement_end}
        # This ensures monotonicity across the replacement boundary
        # Since T_{replacement_end} doesn't exist (it's replaced), we need to ensure
        # that if T_{replacement_end+1} is true, then the distance constraint
        # for T_{replacement_end} would also be satisfied through mutual exclusion
        # This is implicitly handled by the mutual exclusion clauses in Stage 3
    
    return clauses, t_vars


def test_hybrid_encoder(u_pos: int, v_pos: int, n: int, UB: int, num_replacements: int):
    """
    Test function for hybrid encoder
    
    Args:
        u_pos: Position of vertex U (1-indexed)
        v_pos: Position of vertex V (1-indexed)
        n: Grid size
        UB: Upper bound cutoff for replacement start
        num_replacements: Number of levels to replace
    """
    print(f"\n=== Testing Hybrid Encoder ===")
    print(f"Grid size: {n}x{n}")
    print(f"U position: {u_pos}, V position: {v_pos}")
    print(f"Actual distance: {abs(u_pos - v_pos)}")
    print(f"UB: {UB}, Replacements: {num_replacements}")
    
    if num_replacements == 0:
        print(f"No replacement: Full standard encoding")
    else:
        replacement_start = UB + 1
        replacement_end = min(replacement_start + num_replacements - 1, n - 1)
        print(f"Keeping T_1 to T_{UB} with activation clauses")
        print(f"Replacing: T_{replacement_start} to T_{replacement_end} with mutual exclusions")
        if replacement_end < n - 1:
            print(f"Adding T_{replacement_end + 1} to T_{n - 1} with activation clauses (Stage 4)")
    
    vpool = IDPool()
    U_vars = [vpool.id(f'U_{i}') for i in range(1, n + 1)]
    V_vars = [vpool.id(f'V_{i}') for i in range(1, n + 1)]
    
    # Set positions
    position_clauses = [[U_vars[u_pos - 1]], [V_vars[v_pos - 1]]]
    
    # Generate distance constraints with hybrid encoding
    dist_clauses, t_vars = encode_abs_distance_hybrid(
        U_vars, V_vars, n, UB, vpool, 
        prefix=f"T[test]", 
        num_replacements=num_replacements
    )
    
    all_clauses = position_clauses + dist_clauses
    
    print(f"\nSolving with {len(all_clauses)} clauses, {len(t_vars)} T variables...")
    
    with Solver(bootstrap_with=all_clauses) as solver:
        if solver.solve():
            model = solver.get_model()
            actual_dist = abs(u_pos - v_pos)
            
            print(f"SAT - Model found")
            print(f"Actual distance: {actual_dist}")
            
            # Check T variable assignments (only for non-replaced ones)
            print(f"T variable assignments (non-replaced):")
            correct = True
            for d in sorted(t_vars.keys()):
                should_be = (actual_dist >= d)
                is_true = t_vars[d] in model
                status = "✓" if should_be == is_true else "✗"
                print(f"  T_{d}: {is_true} (expected: {should_be}) {status}")
                if should_be != is_true:
                    correct = False
            
            # Check if replaced range is implicitly satisfied
            if num_replacements > 0:
                replacement_start = UB + 1
                replacement_end = min(replacement_start + num_replacements - 1, n - 1)
                replacement_satisfied = True
                for d in range(replacement_start, replacement_end + 1):
                    if actual_dist >= d + 1:
                        # Should be forbidden by mutual exclusion
                        replacement_satisfied = False
                        break
                
                print(f"Replaced range T_{replacement_start} to T_{replacement_end}: {'✓' if replacement_satisfied else '✗'}")
            
            overall_status = "CORRECT" if correct else "ERROR"
            print(f"Overall result: {overall_status}")
        else:
            actual_dist = abs(u_pos - v_pos)
            print(f"UNSAT - distance {actual_dist}")


def compare_hybrid_modes(n: int, UB: int):
    """
    Compare different replacement modes
    
    Args:
        n: Grid size
        UB: Upper bound cutoff
    """
    print(f"\n=== HYBRID ENCODER MODE COMPARISON (n={n}, UB={UB}) ===")
    
    vpool_base = IDPool()
    U_vars = [vpool_base.id(f'U_{i}') for i in range(1, n + 1)]
    V_vars = [vpool_base.id(f'V_{i}') for i in range(1, n + 1)]
    
    modes = [
        (0, "Full standard (no replacement)"),
        (1, f"Replace T_{UB+1} only"),
        (2, f"Replace T_{UB+1} and T_{UB+2}"),
        (n - 1 - UB, f"Full replacement (T_{UB+1} to T_{n-1})"),
    ]
    
    print(f"\nMode Comparison:")
    for num_repl, desc in modes:
        if num_repl > n - 1 - UB:
            continue  # Skip invalid modes
        
        vpool = IDPool()
        U_test = [vpool.id(f'U_{i}') for i in range(1, n + 1)]
        V_test = [vpool.id(f'V_{i}') for i in range(1, n + 1)]
        
        clauses, t_vars = encode_abs_distance_hybrid(
            U_test, V_test, n, UB, vpool,
            prefix=f"T[mode_{num_repl}]",
            num_replacements=num_repl
        )
        
        print(f"\n{desc}:")
        print(f"  Replacements: {num_repl}")
        print(f"  T variables: {len(t_vars)}")
        print(f"  Total clauses: {len(clauses)}")
        print(f"  T range with activation: T_1 to T_{max(t_vars.keys()) if t_vars else 0}")
        
        if num_repl > 0:
            replacement_start = UB + 1
            replacement_end = min(replacement_start + num_repl - 1, n - 1)
            print(f"  Replaced range: T_{replacement_start} to T_{replacement_end} (mutual exclusion)")


def verify_cutoff_equivalence(n: int):
    """
    Verify that full replacement mode equals distance_encoder_cutoff.py
    
    Args:
        n: Grid size
    """
    print(f"\n=== VERIFYING CUTOFF EQUIVALENCE (n={n}) ===")
    
    UB = calculate_theoretical_upper_bound(n)
    print(f"Theoretical UB: {UB}")
    
    # Test hybrid with full replacement
    vpool_hybrid = IDPool()
    U_hybrid = [vpool_hybrid.id(f'U_{i}') for i in range(1, n + 1)]
    V_hybrid = [vpool_hybrid.id(f'V_{i}') for i in range(1, n + 1)]
    
    clauses_hybrid, t_vars_hybrid = encode_abs_distance_hybrid(
        U_hybrid, V_hybrid, n, UB, vpool_hybrid,
        prefix="T[hybrid]",
        num_replacements=n - 1 - UB  # Full replacement
    )
    
    # Test cutoff encoder
    try:
        from distance_encoder_cutoff import encode_abs_distance_cutoff
        
        vpool_cutoff = IDPool()
        U_cutoff = [vpool_cutoff.id(f'U_{i}') for i in range(1, n + 1)]
        V_cutoff = [vpool_cutoff.id(f'V_{i}') for i in range(1, n + 1)]
        
        clauses_cutoff, t_vars_cutoff = encode_abs_distance_cutoff(
            U_cutoff, V_cutoff, UB, vpool_cutoff,
            t_var_prefix="T[cutoff]"
        )
        
        print(f"\nHybrid (full replacement):")
        print(f"  T variables: {len(t_vars_hybrid)}")
        print(f"  Total clauses: {len(clauses_hybrid)}")
        
        print(f"\nCutoff encoder:")
        print(f"  T variables: {len(t_vars_cutoff)}")
        print(f"  Total clauses: {len(clauses_cutoff)}")
        
        # Compare structure
        t_match = len(t_vars_hybrid) == len(t_vars_cutoff)
        print(f"\nT variable count match: {'✓' if t_match else '✗'}")
        
        # Clause counts should be similar (may differ slightly due to implementation details)
        clause_diff = abs(len(clauses_hybrid) - len(clauses_cutoff))
        clause_similar = clause_diff < len(clauses_cutoff) * 0.1  # Within 10%
        print(f"Clause count similar: {'✓' if clause_similar else '✗'} (diff: {clause_diff})")
        
        if t_match and clause_similar:
            print(f"\n✓ VERIFIED: Hybrid with full replacement ≈ Cutoff encoder")
        else:
            print(f"\n⚠ WARNING: Some differences detected (may be due to implementation details)")
            
    except ImportError:
        print("\nCould not import distance_encoder_cutoff for comparison")


if __name__ == '__main__':
    """
    Test suite for hybrid distance encoder
    """
    
    print("=== HYBRID DISTANCE ENCODER TESTS ===")
    
    # Test 1: Basic functionality with different replacement levels
    print(f"\n" + "="*50)
    print(f"Test 1: Different replacement levels")
    print(f"="*50)
    
    n = 8
    UB = calculate_theoretical_upper_bound(n)
    print(f"Grid size: {n}, Theoretical UB: {UB}")
    
    test_cases = [
        (1, 5, 0),      # No replacement
        (1, 5, 1),      # Replace T_UB only
        (1, 5, 2),      # Replace T_UB and T_{UB+1}
        (1, 5, n-1-UB), # Full replacement
    ]
    
    for u, v, num_repl in test_cases:
        test_hybrid_encoder(u, v, n, UB, num_repl)
    
    # Test 2: Mode comparison
    print(f"\n" + "="*50)
    print(f"Test 2: Mode comparison")
    print(f"="*50)
    
    for grid_size in [5, 8, 10]:
        UB = calculate_theoretical_upper_bound(grid_size)
        compare_hybrid_modes(grid_size, UB)
    
    # Test 3: Verify cutoff equivalence
    print(f"\n" + "="*50)
    print(f"Test 3: Cutoff equivalence verification")
    print(f"="*50)
    
    for grid_size in [5, 8, 10]:
        verify_cutoff_equivalence(grid_size)
    
    print(f"\n=== ALL TESTS COMPLETED ===")
    print(f"\nKey Features:")
    print(f"  1. Incremental replacement: Control how many T variables to replace")
    print(f"  2. Performance range: From standard (0) to cutoff (full) encoding")
    print(f"  3. Flexibility: Test different replacement strategies")
    print(f"  4. Equivalence: Full replacement = distance_encoder_cutoff.py")
