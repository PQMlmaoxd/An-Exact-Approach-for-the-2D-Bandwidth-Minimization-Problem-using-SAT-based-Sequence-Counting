# distance_encoder.py - O(n²) optimized version

from pysat.formula import IDPool
from pysat.solvers import Solver

def encode_abs_distance_final(U_vars, V_vars, n, vpool):
    """Mã hóa khoảng cách O(n²) theo luật đối xứng"""
    
    T_vars = [vpool.id(f'T_geq_{d}') for d in range(1, n)]
    clauses = []
    
    # Luật Bật T (đối xứng, O(n²))
    # ∀k,d: (V_k ∧ U_{k-d}) → T_d (cho V > U)
    for k in range(1, n + 1):  # O(n)
        for d in range(1, k):  # O(n) - d từ 1 đến k-1
            if d - 1 < len(T_vars):  # Đảm bảo T_d tồn tại
                u_pos = k - d  # Vị trí U cần có để khoảng cách = d
                if u_pos >= 1:
                    # (V_k ∧ U_{k-d}) → T_d
                    # Tương đương: ¬V_k ∨ ¬U_{k-d} ∨ T_d
                    clauses.append([-V_vars[k - 1], -U_vars[u_pos - 1], T_vars[d - 1]])
    
    # ∀k,d: (U_k ∧ V_{k-d}) → T_d (cho U > V)
    for k in range(1, n + 1):  # O(n)
        for d in range(1, k):  # O(n) - d từ 1 đến k-1
            if d - 1 < len(T_vars):  # Đảm bảo T_d tồn tại
                v_pos = k - d  # Vị trí V cần có để khoảng cách = d
                if v_pos >= 1:
                    # (U_k ∧ V_{k-d}) → T_d
                    # Tương đương: ¬U_k ∨ ¬V_{k-d} ∨ T_d
                    clauses.append([-U_vars[k - 1], -V_vars[v_pos - 1], T_vars[d - 1]])
    
    # Luật Tắt T (chặt chẽ, O(n²))
    # ∀k,d: (V_k ∧ U_{k-d}) → ¬T_{d+1} (cho V > U)
    for k in range(1, n + 1):  # O(n)
        for d in range(1, k):  # O(n) - d từ 1 đến k-1
            if d < len(T_vars):  # Đảm bảo T_{d+1} tồn tại
                u_pos = k - d  # Vị trí U để khoảng cách = d
                if u_pos >= 1:
                    # (V_k ∧ U_{k-d}) → ¬T_{d+1}
                    # Tương đương: ¬V_k ∨ ¬U_{k-d} ∨ ¬T_{d+1}
                    clauses.append([-V_vars[k - 1], -U_vars[u_pos - 1], -T_vars[d]])
    
    # ∀k,d: (U_k ∧ V_{k-d}) → ¬T_{d+1} (cho U > V)
    for k in range(1, n + 1):  # O(n)
        for d in range(1, k):  # O(n) - d từ 1 đến k-1
            if d < len(T_vars):  # Đảm bảo T_{d+1} tồn tại
                v_pos = k - d  # Vị trí V để khoảng cách = d
                if v_pos >= 1:
                    # (U_k ∧ V_{k-d}) → ¬T_{d+1}
                    # Tương đương: ¬U_k ∨ ¬V_{k-d} ∨ ¬T_{d+1}
                    clauses.append([-U_vars[k - 1], -V_vars[v_pos - 1], -T_vars[d]])
    
    # Luật Đơn điệu: T_d → T_{d-1}
    for d in range(2, len(T_vars) + 1):
        clauses.append([-T_vars[d - 1], T_vars[d - 2]])
    
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
            
            print("✓ CORRECT" if correct else "✗ ERROR")
        else:
            print("UNSAT - ERROR")

if __name__ == '__main__':
    test_cases = [(5, 2, 10), (3, 8, 10), (7, 7, 10), (1, 10, 10)]
    for u, v, n in test_cases:
        test_final_encoder(u, v, n)