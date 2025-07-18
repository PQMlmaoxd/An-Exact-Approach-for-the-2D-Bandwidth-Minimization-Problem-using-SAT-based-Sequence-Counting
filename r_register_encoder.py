# r_register_encoder.py - Encoder sử dụng thanh R để biểu diễn khoảng cách

from pysat.formula import IDPool
from pysat.solvers import Solver

def encode_distance_with_r_register(U_vars, V_vars, n, vpool):
    """
    Mã hóa khoảng cách sử dụng thanh R register
    
    Ý tưởng:
    - Thanh R: R_n |--------- ... ---| (điểm Rk) ---| (điểm Rk-1) ----... | R2----|R1
    - Từ R1->Rk-1 sẽ là bit 0, từ Rk trở đi sẽ là bit 1
    - Điểm k trên thanh R tương ứng với V_x^k = 1
    - Thanh T biểu diễn khoảng cách: T_n-1 |-----...--|(K)----|(K-1)-----....|T2-----T1|
    
    Args:
        U_vars: List các biến U_1, U_2, ..., U_n
        V_vars: List các biến V_1, V_2, ..., V_n  
        n: Kích thước của thanh
        vpool: ID Pool để tạo biến mới
        
    Returns:
        R_vars: List các biến R_1, R_2, ..., R_n
        T_vars: List các biến T_1, T_2, ..., T_{n-1}
        clauses: List các mệnh đề SAT
    """
    
    # Tạo các biến cho thanh R và thanh T
    R_vars = [vpool.id(f'R_{i}') for i in range(1, n + 1)]
    T_vars = [vpool.id(f'T_{i}') for i in range(1, n)]
    
    clauses = []
    
    # Ràng buộc 1: Chỉ có một U_x^l = 1 (exactly one constraint)
    # ∑_{l=1}^{n} U_x^l = 1
    clauses.append(U_vars[:])  # Ít nhất một U_i = 1
    for i in range(n):
        for j in range(i + 1, n):
            clauses.append([-U_vars[i], -U_vars[j]])  # Tối đa một U_i = 1
    
    # Ràng buộc 2: Chỉ có một V_x^l = 1 (exactly one constraint)
    # ∑_{l=1}^{n} V_x^l = 1
    clauses.append(V_vars[:])  # Ít nhất một V_i = 1
    for i in range(n):
        for j in range(i + 1, n):
            clauses.append([-V_vars[i], -V_vars[j]])  # Tối đa một V_i = 1
    
    # Ràng buộc 3: Kết nối U_vars với R_vars
    # Nếu U_x^k = 1 thì R1...R_{k-1} = 0 và R_k...R_n = 1
    for k in range(1, n + 1):  # k từ 1 đến n
        # U_x^k = 1 → R_1 = 0, R_2 = 0, ..., R_{k-1} = 0
        for i in range(1, k):  # i từ 1 đến k-1
            clauses.append([-U_vars[k-1], -R_vars[i-1]])  # ¬U_k ∨ ¬R_i
        
        # U_x^k = 1 → R_k = 1, R_{k+1} = 1, ..., R_n = 1
        for i in range(k, n + 1):  # i từ k đến n
            clauses.append([-U_vars[k-1], R_vars[i-1]])  # ¬U_k ∨ R_i
    
    # Ràng buộc 4: Luật mã hóa khoảng cách chính
    # Cho mỗi vị trí k của V_x^k
    for k in range(1, n + 1):  # k từ 1 đến n
        
        # Phần 1: V_x^k ∧ R_{k-1} → T_1, V_x^k ∧ R_{k-2} → T_2, ..., V_x^k ∧ R_1 → T_{k-1}
        for i in range(1, k):  # i từ 1 đến k-1
            t_index = k - i  # T_{k-i}
            if t_index <= len(T_vars):
                # V_x^k ∧ R_i → T_{k-i}
                # Tương đương: ¬V_x^k ∨ ¬R_i ∨ T_{k-i}
                clauses.append([-V_vars[k-1], -R_vars[i-1], T_vars[t_index-1]])
        
        # Phần 2: Luật tắt T từ bên phải
        # V_x^k ∧ R_k ∧ ¬R_{k-1} → ¬T_1
        if k > 1:
            clauses.append([-V_vars[k-1], -R_vars[k-1], R_vars[k-2], -T_vars[0]])
        
        # V_x^k ∧ ¬R_k → T_1
        clauses.append([-V_vars[k-1], R_vars[k-1], T_vars[0]])
        
        # V_x^k ∧ ¬R_{k+1} → T_2, ..., V_x^k ∧ ¬R_{n-1} → T_{n-k}
        for i in range(k + 1, n + 1):  # i từ k+1 đến n
            t_index = i - k + 1  # T_{i-k+1}
            if t_index <= len(T_vars):
                # V_x^k ∧ ¬R_i → T_{i-k+1}
                # Tương đương: ¬V_x^k ∨ R_i ∨ T_{i-k+1}
                clauses.append([-V_vars[k-1], R_vars[i-1], T_vars[t_index-1]])
    
    # Ràng buộc 5: Luật bổ sung để đảm bảo tính chính xác
    # ¬T_i → ¬T_{i+1} (tính đơn điệu của thanh T)
    for i in range(1, len(T_vars)):
        clauses.append([T_vars[i-1], -T_vars[i]])  # T_i ∨ ¬T_{i+1}
    
    # Ràng buộc 6: Luật bổ sung từ yêu cầu
    for k in range(1, n + 1):  # k từ 1 đến n
        
        # For all i >= k: V_x^k ∧ ¬R_i ∧ R_{i+1} → ¬T_{i+1-k+1}
        for i in range(k, n):  # i từ k đến n-1 (để có R_{i+1})
            t_index = i + 1 - k + 1  # T_{i+1-k+1} = T_{i-k+2}
            if t_index >= 1 and t_index <= len(T_vars):
                # V_x^k ∧ ¬R_i ∧ R_{i+1} → ¬T_{i-k+2}
                # Tương đương: ¬V_x^k ∨ R_i ∨ ¬R_{i+1} ∨ ¬T_{i-k+2}
                clauses.append([-V_vars[k-1], R_vars[i-1], -R_vars[i], -T_vars[t_index-1]])
        
        # For all i < k: V_x^k ∧ ¬R_i ∧ R_{i+1} → ¬T_{k-i}
        for i in range(1, k):  # i từ 1 đến k-1 (để có R_{i+1})
            t_index = k - i  # T_{k-i}
            if t_index >= 1 and t_index <= len(T_vars) and i < n:
                # V_x^k ∧ ¬R_i ∧ R_{i+1} → ¬T_{k-i}
                # Tương đương: ¬V_x^k ∨ R_i ∨ ¬R_{i+1} ∨ ¬T_{k-i}
                clauses.append([-V_vars[k-1], R_vars[i-1], -R_vars[i], -T_vars[t_index-1]])
    
    return R_vars, T_vars, clauses

def test_r_register_encoder(u_pos, v_pos, n):
    """
    Test function cho R register encoder
    """
    vpool = IDPool()
    U_vars = [vpool.id(f'U_{i}') for i in range(1, n + 1)]
    V_vars = [vpool.id(f'V_{i}') for i in range(1, n + 1)]
    
    # Thiết lập U_pos và V_pos
    clauses = [[U_vars[u_pos - 1]], [V_vars[v_pos - 1]]]
    
    # Áp dụng encoder
    R_vars, T_vars, dist_clauses = encode_distance_with_r_register(U_vars, V_vars, n, vpool)
    clauses.extend(dist_clauses)
    
    with Solver(bootstrap_with=clauses) as solver:
        if solver.solve():
            model = solver.get_model()
            actual_dist = abs(u_pos - v_pos)
            
            print(f"Test: U={u_pos}, V={v_pos}, Expected distance={actual_dist}")
            
            # In trạng thái thanh R
            r_states = []
            for i, r_var in enumerate(R_vars):
                r_states.append('1' if r_var in model else '0')
            print(f"R register: {''.join(r_states)} (R1 to R{n})")
            
            # In trạng thái thanh T
            t_states = []
            for i, t_var in enumerate(T_vars):
                t_states.append('1' if t_var in model else '0')
            print(f"T register: {''.join(t_states)} (T1 to T{n-1})")
            
            # Kiểm tra tính đúng đắn
            # Đếm số bit 1 liên tiếp từ T1
            encoded_dist = 0
            for i, t_var in enumerate(T_vars):
                if t_var in model:
                    encoded_dist = i + 1
                else:
                    break
            
            print(f"Encoded distance: {encoded_dist}")
            print("✓ CORRECT" if encoded_dist == actual_dist else "✗ ERROR")
            print("-" * 50)
            
        else:
            print(f"Test: U={u_pos}, V={v_pos} - UNSAT ERROR")
            print("-" * 50)

def demonstrate_r_register_concept():
    """
    Minh họa concept của R register encoder
    """
    print("=== R Register Encoder Concept Demo ===")
    print("Ý tưởng: Sử dụng thanh R để biểu diễn vị trí của Ux")
    print("Thanh R: R_n |---...---| Rk |---...---| R2 |---| R1")
    print("Nếu U_x^k = 1 thì: R1...R_{k-1} = 0, R_k...R_n = 1")
    print("Thanh T biểu diễn khoảng cách |abs(Ux - Vx)|")
    print("=" * 60)

if __name__ == '__main__':
    demonstrate_r_register_concept()
    
    # Test cases
    test_cases = [
        (5, 3, 8),  # distance = 2
        (3, 5, 8),  # distance = 2 (symmetric)
        (1, 8, 8),  # distance = 7 (maximum)
        (4, 4, 8),  # distance = 0
        (2, 6, 8),  # distance = 4
    ]
    
    for u, v, n in test_cases:
        test_r_register_encoder(u, v, n)
