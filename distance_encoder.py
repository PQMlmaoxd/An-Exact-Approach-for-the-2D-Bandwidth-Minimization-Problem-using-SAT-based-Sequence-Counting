# distance_encoder.py
#
# Mục đích: Mã hóa ràng buộc khoảng cách T = abs(Ux - Vx)
# Đầu vào: Biến one-hot cho vị trí của U và V trên một đường thẳng.
# Đầu ra: Các biến cho thanh ghi khoảng cách T và các mệnh đề CNF định nghĩa chúng.

from pysat.solvers import Solver
from pysat.formula import IDPool
from itertools import combinations

def encode_absolute_distance(Ux_vars, Vx_vars, vpool):
    """
    Mã hóa khoảng cách T = abs(Ux - Vx) theo ý tưởng thermometer encoding.
    
    Ý tưởng chính:
    - R_i biểu diễn Ux >= i (thermometer code cho vị trí Ux)
    - T_d biểu diễn abs(Ux - Vx) >= d (thermometer code cho khoảng cách)
    - Hai cách mã hóa tùy thuộc vào vị trí tương đối của Ux và Vx

    Args:
        Ux_vars (list of int): Danh sách n biến one-hot cho vị trí của U. 
                               Ux_vars[l-1] là biến U_x^l (U ở vị trí l).
        Vx_vars (list of int): Danh sách n biến one-hot cho vị trí của V.
        vpool (IDPool): Đối tượng quản lý biến để tạo biến phụ.

    Returns:
        tuple: (T_vars, R_vars, clauses)
            T_vars (list of int): Danh sách n-1 biến cho thanh ghi T (T_1, ..., T_{n-1}).
            R_vars (list of int): Danh sách n biến cho thanh ghi R của Ux.
            clauses (list of list of int): Danh sách các mệnh đề CNF đã tạo.
    """
    n = len(Ux_vars)
    if n != len(Vx_vars):
        raise ValueError("Ux_vars và Vx_vars phải có cùng độ dài")

    # =================================================================
    # BƯỚC 1: TẠO BIẾN PHỤ (AUXILIARY VARIABLES)
    # =================================================================
    group_id = hash(str(Ux_vars) + str(Vx_vars)) 
    
    # R_i: Thanh ghi thermometer cho vị trí Ux (Ux >= i)
    # R1, R2, ..., Rn với R_i = True nếu Ux >= i
    R_vars = [vpool.id(f'R_{group_id}_{i}') for i in range(1, n + 1)]
    
    # T_d: Thanh ghi thermometer cho khoảng cách (abs(Ux-Vx) >= d)
    T_vars = [vpool.id(f'T_{group_id}_{d}') for d in range(1, n)]

    clauses = []

    # =================================================================
    # BƯỚC 2: ĐỊNH NGHĨA THANH GHI R (THERMOMETER CODE CHO Ux)
    # =================================================================
    # R_i <--> (Ux >= i) <--> OR_{l=i to n} (Ux_l)
    for i in range(1, n + 1):
        # Chiều thuận: Ux_l -> R_i (với mọi l >= i)
        for l in range(i, n + 1):
            clauses.append([-Ux_vars[l - 1], R_vars[i - 1]])
        
        # Chiều ngược: R_i -> OR_{l=i to n} (Ux_l)
        vars_for_geq_i = [Ux_vars[l - 1] for l in range(i, n + 1)]
        if vars_for_geq_i:
            clauses.append([-R_vars[i - 1]] + vars_for_geq_i)
        else:
            clauses.append([-R_vars[i - 1]])

    # Tính chất thermometer cho R: R_i -> R_{i-1}
    for i in range(2, n + 1):
        clauses.append([-R_vars[i - 1], R_vars[i - 2]])

    # =================================================================
    # BƯỚC 3: MÃ HÓA KHOẢNG CÁCH THEO HAI CÁCH (CORE LOGIC)
    # =================================================================
    
    # CÁCH 1: Khi Vx = k và Ux < k (tức là Ux nằm bên trái Vx)
    # Khoảng cách = k - Ux
    # Vx^k AND NOT R_i -> T_{k-i} (với i < k)
    # Nghĩa là: nếu V ở vị trí k và U không đạt được vị trí i, thì khoảng cách >= k-i
    for k in range(1, n + 1):
        vk_var = Vx_vars[k - 1]  # Vx^k
        for i in range(1, k):  # i < k
            distance = k - i
            if distance < n:  # Đảm bảo T_{distance} tồn tại
                # Vx^k AND NOT R_i -> T_{k-i}
                # Tương đương: -Vx^k OR R_i OR T_{k-i}
                clauses.append([-vk_var, R_vars[i - 1], T_vars[distance - 1]])
    
    # CÁCH 2: Khi Vx = k và Ux >= k (tức là Ux nằm bên phải hoặc trùng Vx)
    # Khoảng cách = Ux - k
    # Vx^k AND R_i AND NOT R_{i-1} -> T_{i-k} (với i >= k)
    # Nghĩa là: nếu V ở vị trí k và U ở đúng vị trí i, thì khoảng cách >= i-k
    for k in range(1, n + 1):
        vk_var = Vx_vars[k - 1]  # Vx^k
        for i in range(k, n + 1):  # i >= k
            distance = i - k
            if distance > 0 and distance < n:  # Đảm bảo T_{distance} tồn tại và distance > 0
                if i == 1:
                    # Trường hợp đặc biệt: R_1 AND NOT R_0 (nhưng R_0 không tồn tại)
                    # Vx^k AND R_1 -> T_{i-k}
                    clauses.append([-vk_var, -R_vars[i - 1], T_vars[distance - 1]])
                else:
                    # Vx^k AND R_i AND NOT R_{i-1} -> T_{i-k}
                    # Tương đương: -Vx^k OR -R_i OR R_{i-1} OR T_{i-k}
                    clauses.append([-vk_var, -R_vars[i - 1], R_vars[i - 2], T_vars[distance - 1]])

    # =================================================================
    # BƯỚC 4: CÁC RÀNG BUỘC BỔ SUNG THEO Ý TƯỞNG CỦA BẠN
    # =================================================================
    
    # Tính chất thermometer cho T: T_i -> T_{i-1}
    # (Nếu khoảng cách >= i thì cũng >= i-1)
    for i in range(2, n):
        clauses.append([-T_vars[i - 1], T_vars[i - 2]])
    
    # Ràng buộc bổ sung: For all i >= k: Vx^k AND NOT R_i AND R_{i+1} -> NOT T_{i+1-k+1}
    # Điều này đảm bảo tính chính xác của mã hóa
    for k in range(1, n + 1):
        vk_var = Vx_vars[k - 1]
        for i in range(k, n):  # i >= k và i < n (để R_{i+1} tồn tại)
            target_t_index = i + 1 - k + 1 - 1  # T_{i+1-k+1} có index là i+1-k
            if target_t_index >= 0 and target_t_index < len(T_vars):
                # Vx^k AND NOT R_i AND R_{i+1} -> NOT T_{i+1-k+1}
                # Tương đương: -Vx^k OR R_i OR -R_{i+1} OR -T_{i+1-k+1}
                clauses.append([-vk_var, R_vars[i - 1], -R_vars[i], -T_vars[target_t_index]])
    
    # Ràng buộc bổ sung: For all i < k: Vx^k AND NOT R_i AND R_{i+1} -> NOT T_{k-i}
    for k in range(1, n + 1):
        vk_var = Vx_vars[k - 1]
        for i in range(1, min(k, n)):  # i < k và i < n (để R_{i+1} tồn tại)
            target_t_index = k - i - 1  # T_{k-i} có index là k-i-1
            if target_t_index >= 0 and target_t_index < len(T_vars) and i < n:
                # Vx^k AND NOT R_i AND R_{i+1} -> NOT T_{k-i}
                # Tương đương: -Vx^k OR R_i OR -R_{i+1} OR -T_{k-i}
                clauses.append([-vk_var, R_vars[i - 1], -R_vars[i], -T_vars[target_t_index]])

    return T_vars, R_vars, clauses

# =================================================================
# PHẦN TÍCH HỢP VỚI NSC ENCODER - RÀNG BUỘC KHOẢNG CÁCH
# =================================================================

def encode_distance_at_least_k(Ux_vars, Vx_vars, k, vpool):
    """
    Mã hóa ràng buộc: abs(Ux - Vx) >= k
    Sử dụng kết hợp distance encoder và NSC encoder.
    
    Args:
        Ux_vars, Vx_vars: Biến one-hot cho vị trí U và V
        k: Ngưỡng khoảng cách tối thiểu
        vpool: Quản lý biến
    
    Returns:
        tuple: (T_vars, R_vars, all_clauses)
    """
    from nsc_encoder import encode_nsc_at_least_k
    
    # Tạo mã hóa khoảng cách cơ bản
    T_vars, R_vars, distance_clauses = encode_absolute_distance(Ux_vars, Vx_vars, vpool)
    
    if k <= 0:
        return T_vars, R_vars, distance_clauses
    if k >= len(Ux_vars):
        # Khoảng cách không thể >= n, trả về UNSAT
        return T_vars, R_vars, distance_clauses + [[1], [-1]]
    
    # Sử dụng NSC để đảm bảo ít nhất k biến T đầu tiên là True
    # T_1, T_2, ..., T_k phải có ít nhất k biến True
    # Nhưng do tính chất thermometer, T_k = True đã đủ đảm bảo T_1...T_{k-1} = True
    nsc_clauses = encode_nsc_at_least_k(T_vars[:k], k, vpool)
    
    all_clauses = distance_clauses + nsc_clauses
    return T_vars, R_vars, all_clauses

def encode_distance_at_most_k(Ux_vars, Vx_vars, k, vpool):
    """
    Mã hóa ràng buộc: abs(Ux - Vx) <= k
    Tương đương với: NOT (abs(Ux - Vx) >= k+1)
    
    Args:
        Ux_vars, Vx_vars: Biến one-hot cho vị trí U và V
        k: Ngưỡng khoảng cách tối đa
        vpool: Quản lý biến
    
    Returns:
        tuple: (T_vars, R_vars, all_clauses)
    """
    from nsc_encoder import encode_nsc_at_most_k
    
    # Tạo mã hóa khoảng cách cơ bản
    T_vars, R_vars, distance_clauses = encode_absolute_distance(Ux_vars, Vx_vars, vpool)
    
    if k < 0:
        # Khoảng cách không thể <= -1, trả về UNSAT
        return T_vars, R_vars, distance_clauses + [[1], [-1]]
    if k >= len(Ux_vars) - 1:
        # Khoảng cách luôn <= n-1, không cần ràng buộc thêm
        return T_vars, R_vars, distance_clauses
    
    # Đảm bảo T_{k+1} = False (tức là khoảng cách < k+1, hay <= k)
    if k < len(T_vars):
        distance_clauses.append([-T_vars[k]])
    
    return T_vars, R_vars, distance_clauses

def encode_distance_exactly_k(Ux_vars, Vx_vars, k, vpool):
    """
    Mã hóa ràng buộc: abs(Ux - Vx) = k
    Kết hợp At-Most-K và At-Least-K
    
    Args:
        Ux_vars, Vx_vars: Biến one-hot cho vị trí U và V
        k: Khoảng cách chính xác
        vpool: Quản lý biến
    
    Returns:
        tuple: (T_vars, R_vars, all_clauses)
    """
    # Tạo mã hóa khoảng cách cơ bản
    T_vars, R_vars, distance_clauses = encode_absolute_distance(Ux_vars, Vx_vars, vpool)
    
    if k < 0 or k >= len(Ux_vars):
        # Khoảng cách không hợp lệ
        return T_vars, R_vars, distance_clauses + [[1], [-1]]
    
    if k == 0:
        # Khoảng cách = 0, tất cả T_i phải False
        for t_var in T_vars:
            distance_clauses.append([-t_var])
    else:
        # Khoảng cách = k: T_k = True và T_{k+1} = False (nếu tồn tại)
        if k <= len(T_vars):
            distance_clauses.append([T_vars[k - 1]])  # T_k = True
        if k < len(T_vars):
            distance_clauses.append([-T_vars[k]])     # T_{k+1} = False
    
    return T_vars, R_vars, distance_clauses

# =================================================================
# HÀM HỖ TRỢ - TẠO RÀNG BUỘC ONE-HOT
# =================================================================

def add_one_hot_constraints(variables, vpool):
    """
    Thêm ràng buộc exactly-one cho một tập biến.
    Sử dụng NSC encoder để tối ưu.
    """
    from nsc_encoder import encode_nsc_exactly_k
    return encode_nsc_exactly_k(variables, 1, vpool)

def create_complete_distance_encoding(n, u_pos, v_pos, distance_constraint, k, vpool):
    """
    Tạo mã hóa hoàn chỉnh cho bài toán khoảng cách với ràng buộc cụ thể.
    
    Args:
        n: Số vị trí trên đường thẳng
        u_pos, v_pos: Vị trí cố định của U và V (để test)
        distance_constraint: 'at_least', 'at_most', hoặc 'exactly'
        k: Ngưỡng khoảng cách
        vpool: Quản lý biến
    
    Returns:
        tuple: (T_vars, R_vars, all_clauses, Ux_vars, Vx_vars)
    """
    # Tạo biến one-hot cho vị trí
    Ux_vars = [vpool.id(f'Ux_{i}') for i in range(1, n + 1)]
    Vx_vars = [vpool.id(f'Vx_{i}') for i in range(1, n + 1)]
    
    # Thêm ràng buộc exactly-one cho U và V
    clauses = []
    clauses.extend(add_one_hot_constraints(Ux_vars, vpool))
    clauses.extend(add_one_hot_constraints(Vx_vars, vpool))
    
    # Cố định vị trí (để test)
    if u_pos is not None:
        clauses.append([Ux_vars[u_pos - 1]])
    if v_pos is not None:
        clauses.append([Vx_vars[v_pos - 1]])
    
    # Thêm ràng buộc khoảng cách
    if distance_constraint == 'at_least':
        T_vars, R_vars, distance_clauses = encode_distance_at_least_k(Ux_vars, Vx_vars, k, vpool)
    elif distance_constraint == 'at_most':
        T_vars, R_vars, distance_clauses = encode_distance_at_most_k(Ux_vars, Vx_vars, k, vpool)
    elif distance_constraint == 'exactly':
        T_vars, R_vars, distance_clauses = encode_distance_exactly_k(Ux_vars, Vx_vars, k, vpool)
    else:
        raise ValueError("distance_constraint phải là 'at_least', 'at_most', hoặc 'exactly'")
    
    clauses.extend(distance_clauses)
    return T_vars, R_vars, clauses, Ux_vars, Vx_vars

# =================================================================
# PHẦN TEST - Kiểm tra và xác minh với các ràng buộc mới
# =================================================================

def run_basic_test(n_test, u_pos, v_pos, expected_dist):
    """Hàm test cơ bản cho mã hóa khoảng cách."""
    print(f"\n--- TEST CƠ BẢN: n={n_test}, Ux={u_pos}, Vx={v_pos} (Khoảng cách: {expected_dist}) ---")
    
    vpool_test = IDPool()
    ux_test_vars = [vpool_test.id(f'Ux_{i}') for i in range(1, n_test + 1)]
    vx_test_vars = [vpool_test.id(f'Vx_{i}') for i in range(1, n_test + 1)]

    T_vars, R_vars, generated_clauses = encode_absolute_distance(ux_test_vars, vx_test_vars, vpool_test)
    
    # Thêm ràng buộc exactly-one và cố định vị trí
    test_clauses = list(generated_clauses)
    test_clauses.extend(add_one_hot_constraints(ux_test_vars, vpool_test))
    test_clauses.extend(add_one_hot_constraints(vx_test_vars, vpool_test))
    test_clauses.append([ux_test_vars[u_pos - 1]])
    test_clauses.append([vx_test_vars[v_pos - 1]])

    # Giải và kiểm tra
    with Solver(name='g3', bootstrap_with=test_clauses) as s:
        if s.solve():
            model = s.get_model()
            print("  Kết quả: SAT")
            correct = True
            for d in range(1, n_test):
                if d - 1 < len(T_vars):
                    var_t = T_vars[d-1]
                    is_true = var_t in model
                    expected_true = (d <= expected_dist)
                    print(f"    T_{d}: {is_true} (Mong đợi: {expected_true})", end="")
                    if is_true != expected_true:
                        print(" <-- LỖI!")
                        correct = False
                    else:
                        print()
            if correct:
                print("  => KẾT QUẢ ĐÚNG!")
            else:
                print("  => CÓ LỖI TRONG MÃ HÓA!")
        else:
            print("  Kết quả: UNSAT - Có lỗi trong bộ mã hóa!")

def run_constraint_test(n_test, u_pos, v_pos, constraint_type, k, expected_satisfiable):
    """Test các ràng buộc khoảng cách với NSC."""
    print(f"\n--- TEST RÀNG BUỘC: n={n_test}, Ux={u_pos}, Vx={v_pos}, {constraint_type} {k} ---")
    
    vpool_test = IDPool()
    actual_distance = abs(u_pos - v_pos)
    
    T_vars, R_vars, test_clauses, Ux_vars, Vx_vars = create_complete_distance_encoding(
        n_test, u_pos, v_pos, constraint_type, k, vpool_test
    )
    
    with Solver(name='g3', bootstrap_with=test_clauses) as s:
        is_sat = s.solve()
        print(f"  Khoảng cách thực tế: {actual_distance}")
        print(f"  Ràng buộc: {constraint_type} {k}")
        print(f"  Kết quả: {'SAT' if is_sat else 'UNSAT'}")
        print(f"  Mong đợi: {'SAT' if expected_satisfiable else 'UNSAT'}")
        
        if is_sat == expected_satisfiable:
            print("  => KẾT QUẢ ĐÚNG!")
            
            if is_sat:
                model = s.get_model()
                print("  Giá trị T_vars:")
                for d in range(1, min(len(T_vars) + 1, n_test)):
                    if d - 1 < len(T_vars):
                        var_t = T_vars[d-1]
                        is_true = var_t in model
                        print(f"    T_{d}: {is_true}")
        else:
            print("  => LỖI! Kết quả không khớp với mong đợi!")

def run_comprehensive_tests():
    """Chạy bộ test toàn diện."""
    print("="*60)
    print("CHẠY BỘ TEST TOÀN DIỆN CHO DISTANCE ENCODER")
    print("="*60)
    
    # Test cơ bản
    run_basic_test(n_test=10, u_pos=7, v_pos=3, expected_dist=4)
    run_basic_test(n_test=10, u_pos=3, v_pos=9, expected_dist=6)
    run_basic_test(n_test=10, u_pos=5, v_pos=5, expected_dist=0)
    run_basic_test(n_test=5, u_pos=1, v_pos=5, expected_dist=4)
    
    # Test ràng buộc At-Least-K
    run_constraint_test(n_test=10, u_pos=3, v_pos=7, constraint_type='at_least', k=3, expected_satisfiable=True)  # distance=4 >= 3
    run_constraint_test(n_test=10, u_pos=3, v_pos=7, constraint_type='at_least', k=5, expected_satisfiable=False) # distance=4 < 5
    
    # Test ràng buộc At-Most-K  
    run_constraint_test(n_test=10, u_pos=2, v_pos=5, constraint_type='at_most', k=5, expected_satisfiable=True)   # distance=3 <= 5
    run_constraint_test(n_test=10, u_pos=2, v_pos=5, constraint_type='at_most', k=2, expected_satisfiable=False)  # distance=3 > 2
    
    # Test ràng buộc Exactly-K
    run_constraint_test(n_test=8, u_pos=2, v_pos=6, constraint_type='exactly', k=4, expected_satisfiable=True)   # distance=4 == 4
    run_constraint_test(n_test=8, u_pos=2, v_pos=6, constraint_type='exactly', k=3, expected_satisfiable=False)  # distance=4 != 3
    run_constraint_test(n_test=6, u_pos=3, v_pos=3, constraint_type='exactly', k=0, expected_satisfiable=True)   # distance=0 == 0

if __name__ == '__main__':
    run_comprehensive_tests()