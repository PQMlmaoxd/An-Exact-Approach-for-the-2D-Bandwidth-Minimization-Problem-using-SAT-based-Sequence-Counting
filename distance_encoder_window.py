# distance_encoder_window.py
# Window-cutoff encoding for per-axis distance with lightweight T variables (to UB)

from typing import Dict, List, Tuple, Optional
from pysat.formula import IDPool
from pysat.solvers import Solver


def _tvar(vpool: IDPool, prefix: str, d: int) -> int:
    """
    Generate a unique variable ID for T_d indicator.
    
    Args:
        vpool: IDPool for variable management
        prefix: Unique prefix per-edge per-axis (e.g., "Tx[u,v]" or "Ty[u,v]")
        d: Distance value (1..UB)
    
    Returns:
        Variable ID for T_d
    """
    return vpool.id(f"{prefix}_T_{d}")


def _mk_window_clause(U_row: List[int], V_row: List[int], i: int, UB: int) -> List[int]:
    """
    Create a window clause: (¬U_i ∨ V_{L(i)} ∨ ... ∨ V_{R(i)})
    where L(i) = max(1, i-UB) and R(i) = min(n, i+UB).
    
    This enforces that if u is at position i, then v must be within window [i-UB, i+UB].
    
    Args:
        U_row: Position variables for vertex u (indexed 0..n-1, representing positions 1..n)
        V_row: Position variables for vertex v (indexed 0..n-1, representing positions 1..n)
        i: Current position (1-indexed)
        UB: Upper bound for distance (window size)
    
    Returns:
        CNF clause as list of literals
    """
    n = len(U_row)
    L = max(1, i - UB)
    R = min(n, i + UB)
    clause = [-U_row[i - 1]]  # ¬U_i
    clause.extend(V_row[k - 1] for k in range(L, R + 1))  # V_L ∨ ... ∨ V_R
    return clause


def encode_abs_distance_window_cutoff(
    U_of_u: List[int],
    U_of_v: List[int],
    UB: int,
    vpool: IDPool,
    prefix: str,
    *,
    add_T: bool = True,
    add_monotonic: bool = True,
    add_base_samepos: bool = True,
    symmetric_window: bool = True
) -> Tuple[List[List[int]], Dict[int, int]]:
    """
    Encode absolute distance between two vertices u and v with window cutoff optimization.
    
    This function creates a more efficient encoding than the O(n²) approach by using
    window constraints that limit the search space to positions within UB distance.
    
    Args:
        U_of_u: Position literals for vertex u (positions 1..n)
        U_of_v: Position literals for vertex v (positions 1..n)
        UB: Theoretical upper bound for distance on this axis
        vpool: Variable pool for generating fresh variables
        prefix: Unique identifier for T variables (e.g., "Tx_u5_v7")
        add_T: If True, create T_d indicator variables and add forward constraints
        add_monotonic: If True, add monotonicity constraints (T_{d+1} → T_d)
        add_base_samepos: If True, add constraint (U_i ∧ V_i) → ¬T_1
        symmetric_window: If True, add reverse window constraints (¬V_k ∨ U_window(k))
    
    Returns:
        clauses: List of CNF clauses
        t_vars: Dictionary mapping distance d to T_d variable ID (1..UB)
    
    Core Idea:
        Instead of encoding all O(n²) position pairs, we use window constraints:
        - For each position i: if u is at i, then v must be in [i-UB, i+UB]
        - T_d indicates that distance ≥ d
        - Forward links: (U_i ∧ V_{i-d}) → T_d establishes exact distance
        - Monotonicity: T_{d+1} → T_d ensures proper ordering
    """
    n = len(U_of_u)
    clauses: List[List[int]] = []
    t_vars: Dict[int, int] = {}
    
    # ===== 1. WINDOW CUTOFF CONSTRAINTS =====
    # Replace the O(n²) binary grid with O(n·UB) window constraints
    # For each position i: (¬U_i ∨ V_{i-UB} ∨ ... ∨ V_{i+UB})
    for i in range(1, n + 1):
        clauses.append(_mk_window_clause(U_of_u, U_of_v, i, UB))
    
    # Optional symmetric window: (¬V_k ∨ U_{k-UB} ∨ ... ∨ U_{k+UB})
    # This can help with propagation but is not strictly necessary
    if symmetric_window:
        for k in range(1, n + 1):
            nL = max(1, k - UB)
            nR = min(n, k + UB)
            clause = [-U_of_v[k - 1]]
            clause.extend(U_of_u[t - 1] for t in range(nL, nR + 1))
            clauses.append(clause)
    
    # If we don't need T variables, return early
    if not add_T:
        return clauses, t_vars
    
    # ===== 2. CREATE T_d INDICATOR VARIABLES =====
    # T_d = 1 iff distance ≥ d (for d = 1..UB)
    for d in range(1, UB + 1):
        t_vars[d] = _tvar(vpool, prefix, d)
    
    # ===== 3. FORWARD ACTIVATION LINKS =====
    # Establish when T_d must be true based on actual positions
    # (U_i ∧ V_{i-d}) → T_d: if u at i and v at i-d, distance is exactly d, so T_d = true
    # (V_i ∧ U_{i-d}) → T_d: symmetric case
    # CNF: (¬U_i ∨ ¬V_{i-d} ∨ T_d) and (¬V_i ∨ ¬U_{i-d} ∨ T_d)
    for d in range(1, UB + 1):
        Td = t_vars[d]
        for i in range(d + 1, n + 1):
            # Case 1: u at position i, v at position i-d (u is ahead)
            clauses.append([-U_of_u[i - 1], -U_of_v[i - d - 1], Td])
            # Case 2: v at position i, u at position i-d (v is ahead)
            clauses.append([-U_of_v[i - 1], -U_of_u[i - d - 1], Td])
    
    # ===== 4. DEACTIVATION LINKS (TIGHTENING) =====
    # Establish when T_{d+1} must be false based on exact distance = d
    # (U_i ∧ V_{i-d}) → ¬T_{d+1}: if distance is exactly d, then distance < d+1
    # (V_i ∧ U_{i-d}) → ¬T_{d+1}: symmetric case
    # CNF: (¬U_i ∨ ¬V_{i-d} ∨ ¬T_{d+1}) and (¬V_i ∨ ¬U_{i-d} ∨ ¬T_{d+1})
    for d in range(1, UB):  # d from 1 to UB-1 (since we need T_{d+1})
        if (d + 1) in t_vars:  # Ensure T_{d+1} exists
            Td_plus_1 = t_vars[d + 1]
            for i in range(d + 1, n + 1):
                # Case 1: u at position i, v at position i-d (distance = d)
                clauses.append([-U_of_u[i - 1], -U_of_v[i - d - 1], -Td_plus_1])
                # Case 2: v at position i, u at position i-d (distance = d)
                clauses.append([-U_of_v[i - 1], -U_of_u[i - d - 1], -Td_plus_1])
    
    # ===== 5. MONOTONICITY CONSTRAINTS =====
    # T_{d+1} → T_d: if distance ≥ d+1, then distance ≥ d
    # CNF: (¬T_{d+1} ∨ T_d)
    # This ensures proper ordering of the distance indicators
    if add_monotonic and UB >= 2:
        for d in range(1, UB):
            clauses.append([-t_vars[d + 1], t_vars[d]])
    
    # ===== 6. SAME POSITION BASE CASE =====
    # (U_i ∧ V_i) → ¬T_1: if both vertices at same position, distance = 0 < 1
    # CNF: (¬U_i ∨ ¬V_i ∨ ¬T_1)
    # This is optional but can help with early propagation
    if add_base_samepos and UB >= 1:
        T1 = t_vars[1]
        for i in range(1, n + 1):
            clauses.append([-U_of_u[i - 1], -U_of_v[i - 1], -T1])
    
    return clauses, t_vars


def k_aware_reverse_clauses(
    U_of_u: List[int],
    U_of_v: List[int],
    K: int,
    UB: int,
    vpool: IDPool,
    prefix: str,
    *,
    symmetric: bool = True
) -> List[List[int]]:
    """
    Generate K-aware reverse bridge clauses for enhanced propagation.
    
    Creates clauses of form: (T_{K+1} ∨ ¬U_i ∨ V_{i-K} ∨ ... ∨ V_{i+K})
    
    Under assumption ¬T_{K+1} (distance ≤ K), this reduces to a tighter window
    constraint that can accelerate SAT solving by pushing constraints directly
    into the region we care about.
    
    Args:
        U_of_u: Position literals for vertex u
        U_of_v: Position literals for vertex v
        K: Target bandwidth bound being tested
        UB: Upper bound for distance (must be > K)
        vpool: Variable pool
        prefix: Prefix for T variables
        symmetric: If True, also add symmetric clauses for V
    
    Returns:
        List of CNF clauses
    
    Note:
        Only meaningful when 0 <= K < UB. If K >= UB, T_{K+1} is outside
        the defined range and these clauses are skipped.
    """
    clauses: List[List[int]] = []
    n = len(U_of_u)
    
    # Validate K range
    if K < 0 or K >= UB:
        # T_{K+1} is outside valid range [1..UB], skip these clauses
        return clauses
    
    Tk1 = _tvar(vpool, prefix, K + 1)
    
    # Forward direction: (T_{K+1} ∨ ¬U_i ∨ V_{i-K} ∨ ... ∨ V_{i+K})
    for i in range(1, n + 1):
        L = max(1, i - K)
        R = min(n, i + K)
        clause = [Tk1, -U_of_u[i - 1]]
        clause.extend(U_of_v[k - 1] for k in range(L, R + 1))
        clauses.append(clause)
    
    # Optional symmetric direction: (T_{K+1} ∨ ¬V_k ∨ U_{k-K} ∨ ... ∨ U_{k+K})
    if symmetric:
        for k in range(1, n + 1):
            L = max(1, k - K)
            R = min(n, k + K)
            clause = [Tk1, -U_of_v[k - 1]]
            clause.extend(U_of_u[t - 1] for t in range(L, R + 1))
            clauses.append(clause)
    
    return clauses


def test_window_encoder(u_pos: int, v_pos: int, n: int, UB: int):
    """
    Test the window-based distance encoder.
    
    Args:
        u_pos: Position of vertex u (1-indexed)
        v_pos: Position of vertex v (1-indexed)
        n: Total number of positions
        UB: Upper bound for distance
    """
    vpool = IDPool()
    
    # Create position variables
    U_vars = [vpool.id(f'U_{i}') for i in range(1, n + 1)]
    V_vars = [vpool.id(f'V_{i}') for i in range(1, n + 1)]
    
    # Fix positions: u at u_pos, v at v_pos
    clauses = [[U_vars[u_pos - 1]], [V_vars[v_pos - 1]]]
    
    # Encode distance with window cutoff
    dist_clauses, T_vars = encode_abs_distance_window_cutoff(
        U_vars, V_vars, UB, vpool, prefix="T",
        add_T=True, add_monotonic=True, add_base_samepos=True, symmetric_window=True
    )
    clauses.extend(dist_clauses)
    
    # Solve and check
    with Solver(bootstrap_with=clauses) as solver:
        if solver.solve():
            model = solver.get_model()
            actual_dist = abs(u_pos - v_pos)
            print(f"U={u_pos}, V={v_pos}, actual_dist={actual_dist}, UB={UB}")
            
            # Verify T variables
            correct = True
            for d in range(1, min(UB + 1, actual_dist + 3)):
                should_be = (actual_dist >= d)
                is_true = T_vars[d] in model
                status = "✓" if should_be == is_true else "✗"
                print(f"  T_{d}: expected={should_be}, actual={is_true} {status}")
                if should_be != is_true:
                    correct = False
            
            print(f"Result: {'CORRECT ✓' if correct else 'ERROR ✗'}\n")
        else:
            print(f"U={u_pos}, V={v_pos}, UB={UB}: UNSAT - ERROR ✗\n")


def test_k_aware_clauses(u_pos: int, v_pos: int, n: int, K: int, UB: int):
    """
    Test K-aware reverse clauses functionality.
    
    Args:
        u_pos: Position of vertex u
        v_pos: Position of vertex v
        n: Total number of positions
        K: Target bandwidth bound
        UB: Upper bound for distance
    """
    vpool = IDPool()
    
    U_vars = [vpool.id(f'U_{i}') for i in range(1, n + 1)]
    V_vars = [vpool.id(f'V_{i}') for i in range(1, n + 1)]
    
    clauses = [[U_vars[u_pos - 1]], [V_vars[v_pos - 1]]]
    
    # Basic encoding
    dist_clauses, T_vars = encode_abs_distance_window_cutoff(
        U_vars, V_vars, UB, vpool, prefix="T",
        add_T=True, add_monotonic=True
    )
    clauses.extend(dist_clauses)
    
    # Add K-aware clauses
    k_clauses = k_aware_reverse_clauses(
        U_vars, V_vars, K, UB, vpool, prefix="T", symmetric=True
    )
    clauses.extend(k_clauses)
    
    # Test with assumption ¬T_{K+1} (distance ≤ K)
    actual_dist = abs(u_pos - v_pos)
    print(f"K-aware test: U={u_pos}, V={v_pos}, dist={actual_dist}, K={K}, UB={UB}")
    
    if K + 1 <= UB:
        assumptions = [-T_vars[K + 1]]  # Assume distance ≤ K
        with Solver(bootstrap_with=clauses) as solver:
            satisfiable = solver.solve(assumptions=assumptions)
            should_be_sat = (actual_dist <= K)
            
            if satisfiable == should_be_sat:
                print(f"  With ¬T_{K+1}: {'SAT' if satisfiable else 'UNSAT'} - CORRECT ✓\n")
            else:
                print(f"  With ¬T_{K+1}: {'SAT' if satisfiable else 'UNSAT'} - ERROR ✗")
                print(f"  Expected: {'SAT' if should_be_sat else 'UNSAT'}\n")
    else:
        print(f"  K+1 > UB, skipping assumption test\n")


if __name__ == '__main__':
    print("=" * 60)
    print("WINDOW-BASED DISTANCE ENCODER TESTS")
    print("=" * 60)
    
    print("\n--- Basic Window Encoding Tests ---\n")
    # Test cases: (u_pos, v_pos, n, UB)
    test_cases = [
        (5, 2, 10, 5),   # distance = 3, within UB
        (3, 8, 10, 7),   # distance = 5, within UB
        (7, 7, 10, 5),   # distance = 0, same position
        (1, 10, 10, 9),  # distance = 9, maximum distance
        (4, 6, 10, 3),   # distance = 2, small UB
    ]
    
    for u, v, n, ub in test_cases:
        test_window_encoder(u, v, n, ub)
    
    print("\n--- K-Aware Reverse Clauses Tests ---\n")
    # Test K-aware functionality: (u_pos, v_pos, n, K, UB)
    k_test_cases = [
        (5, 2, 10, 2, 6),   # dist=3, K=2, should be UNSAT with ¬T_3
        (3, 8, 10, 5, 8),   # dist=5, K=5, should be SAT with ¬T_6
        (1, 4, 10, 3, 6),   # dist=3, K=3, should be SAT with ¬T_4
        (7, 9, 10, 1, 5),   # dist=2, K=1, should be UNSAT with ¬T_2
    ]
    
    for u, v, n, k, ub in k_test_cases:
        test_k_aware_clauses(u, v, n, k, ub)
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
