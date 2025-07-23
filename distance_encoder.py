# distance_encoder.py - O(n²) optimized version

from pysat.formula import IDPool
from pysat.solvers import Solver

def encode_abs_distance_final(U_vars, V_vars, n, vpool, prefix="T"):
    
    T_vars = [vpool.id(f'{prefix}_geq_{d}') for d in range(1, n)]
    clauses = []
    
    # T Activation Rules (symmetric, O(n²))
    # ∀k,d: (V_k ∧ U_{k-d}) → T_d (for V > U)
    for k in range(1, n + 1):  # O(n)
        for d in range(1, k):  # O(n) - d from 1 to k-1
            if d - 1 < len(T_vars):  # Ensure T_d exists
                u_pos = k - d  # U position needed for distance = d
                if u_pos >= 1:
                    # (V_k ∧ U_{k-d}) → T_d
                    # Equivalent: ¬V_k ∨ ¬U_{k-d} ∨ T_d
                    clauses.append([-V_vars[k - 1], -U_vars[u_pos - 1], T_vars[d - 1]])
    
    # ∀k,d: (U_k ∧ V_{k-d}) → T_d (for U > V)
    for k in range(1, n + 1):  # O(n)
        for d in range(1, k):  # O(n) - d from 1 to k-1
            if d - 1 < len(T_vars):  # Ensure T_d exists
                v_pos = k - d  # V position needed for distance = d
                if v_pos >= 1:
                    # (U_k ∧ V_{k-d}) → T_d
                    # Equivalent: ¬U_k ∨ ¬V_{k-d} ∨ T_d
                    clauses.append([-U_vars[k - 1], -V_vars[v_pos - 1], T_vars[d - 1]])
    
    # T Deactivation Rules (tight, O(n²))
    # ∀k,d: (V_k ∧ U_{k-d}) → ¬T_{d+1} (for V > U)
    for k in range(1, n + 1):  # O(n)
        for d in range(1, k):  # O(n) - d from 1 to k-1
            if d < len(T_vars):  # Ensure T_{d+1} exists
                u_pos = k - d  # U position for distance = d
                if u_pos >= 1:
                    # (V_k ∧ U_{k-d}) → ¬T_{d+1}
                    # Equivalent: ¬V_k ∨ ¬U_{k-d} ∨ ¬T_{d+1}
                    clauses.append([-V_vars[k - 1], -U_vars[u_pos - 1], -T_vars[d]])
    
    # ∀k,d: (U_k ∧ V_{k-d}) → ¬T_{d+1} (for U > V)
    for k in range(1, n + 1):  # O(n)
        for d in range(1, k):  # O(n) - d from 1 to k-1
            if d < len(T_vars):  # Ensure T_{d+1} exists
                v_pos = k - d  # V position for distance = d
                if v_pos >= 1:
                    # (U_k ∧ V_{k-d}) → ¬T_{d+1}
                    # Equivalent: ¬U_k ∨ ¬V_{k-d} ∨ ¬T_{d+1}
                    clauses.append([-U_vars[k - 1], -V_vars[v_pos - 1], -T_vars[d]])
    
    # CRITICAL FIX: Add constraints for distance = 0 case
    # If U and V are at same position, distance < 1, so T_geq_1 = False
    for k in range(1, n + 1):
        if len(T_vars) > 0:  # Ensure T_geq_1 exists
            # (U_k ∧ V_k) → ¬T_1 (same position → distance < 1)
            clauses.append([-U_vars[k - 1], -V_vars[k - 1], -T_vars[0]])
    
    # Monotonicity Rules: ¬T_d → ¬T_{d+1} (bit propagation from left to right)
    for d in range(1, len(T_vars)):
        clauses.append([T_vars[d - 1], -T_vars[d]])
    
    return T_vars, clauses

def test_final_encoder(u_pos, v_pos, n):
    """Test function"""
    vpool = IDPool()
    U_vars = [vpool.id(f'U_{i}') for i in range(1, n + 1)]
    V_vars = [vpool.id(f'V_{i}') for i in range(1, n + 1)]
    
    clauses = [[U_vars[u_pos - 1]], [V_vars[v_pos - 1]]]
    T_vars, dist_clauses = encode_abs_distance_final(U_vars, V_vars, n, vpool)
    clauses.extend(dist_clauses)
    
    with Solver(bootstrap_with=clauses) as solver:
        if solver.solve():
            model = solver.get_model()
            actual_dist = abs(u_pos - v_pos)
            print(f"U={u_pos}, V={v_pos}, dist={actual_dist}")
            
            correct = True
            for d in range(1, min(len(T_vars) + 1, actual_dist + 3)):
                should_be = (actual_dist >= d)
                is_true = T_vars[d - 1] in model
                if should_be != is_true:
                    correct = False
                    break
            
            print("CORRECT" if correct else "ERROR")
        else:
            print("UNSAT - ERROR")

if __name__ == '__main__':
    test_cases = [(5, 2, 10), (3, 8, 10), (7, 7, 10), (1, 10, 10)]
    for u, v, n in test_cases:
        test_final_encoder(u, v, n)