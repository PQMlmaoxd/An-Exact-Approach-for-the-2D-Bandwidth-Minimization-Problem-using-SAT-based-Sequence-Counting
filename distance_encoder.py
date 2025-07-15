# final_distance_encoder.py

from pysat.formula import IDPool
from pysat.solvers import Solver

def encode_abs_distance_final(U_vars, V_vars, n, vpool):
    print("\n--- Bắt đầu mã hóa khoảng cách ---")
    
    T_vars = [vpool.id(f'T_geq_{d}') for d in range(1, n)]
    print(f"Đã tạo {len(T_vars)} biến T.")

    clauses = []
    
    # Duyệt qua tất cả các cặp vị trí có thể của U và V
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            u_i = U_vars[i - 1]
            v_j = V_vars[j - 1]
            
            dist = abs(i - j)
            
            # =========================================================
            # LUẬT 1: "BẬT ĐÈN" (Chiều thuận)
            # (U_i AND V_j) --> T_d  cho mọi d <= dist
            # =========================================================
            if dist > 0:
                for d in range(1, dist + 1):
                    td = T_vars[d - 1]
                    clauses.append([-u_i, -v_j, td])

            # =========================================================
            # LUẬT 2: "TẮT ĐÈN" (Chiều ngược / Đặt chốt)
            # (U_i AND V_j) --> NOT T_d cho mọi d > dist
            # =========================================================
            # Tối ưu: Chỉ cần đặt chốt bit 0 đầu tiên là đủ
            first_zero_idx = dist
            if first_zero_idx < len(T_vars):
                not_t_first_zero = -T_vars[first_zero_idx]
                clauses.append([-u_i, -v_j, not_t_first_zero])
                
    # =================================================================
    # LUẬT 3: "LAN TRUYỀN" (Tính đơn điệu)
    # NOT T_d --> NOT T_{d+1}  (hay T_{d+1} -> T_d)
    # Luật này giúp solver hiệu quả hơn nhưng không bắt buộc nếu có luật 1 & 2
    # =================================================================
    for d in range(2, n):
        t_d = T_vars[d - 1]
        t_d_minus_1 = T_vars[d - 2]
        clauses.append([-t_d, t_d_minus_1])
        
    print(f"Đã tạo tổng cộng {len(clauses)} mệnh đề định nghĩa T.")

    return T_vars, clauses

# =================================================================
# PHẦN KIỂM TRA (SỬ DỤNG BỘ MÃ HÓA CUỐI CÙNG NÀY)
# =================================================================

def test_final_encoder(u_pos, v_pos, n):
    """
    Hàm chính để kiểm tra bộ mã hóa cuối cùng.
    """
    print(f"\n===== KIỂM TRA PHIÊN BẢN CUỐI CÙNG: U = {u_pos}, V = {v_pos}, N = {n} =====")
    
    vpool = IDPool()
    clauses = []

    # Tạo và ràng buộc biến one-hot U và V
    U_vars = [vpool.id(f'U_{i}') for i in range(1, n + 1)]
    V_vars = [vpool.id(f'V_{i}') for i in range(1, n + 1)]
    
    # Cố định giá trị U = u_pos và V = v_pos
    clauses.append([U_vars[u_pos - 1]])
    clauses.append([V_vars[v_pos - 1]])
    # (Để đơn giản, không cần thêm các mệnh đề phủ định vì chỉ có 1 lời giải)

    # Sử dụng bộ mã hóa cuối cùng
    T_vars, dist_clauses = encode_abs_distance_final(U_vars, V_vars, n, vpool)
    clauses.extend(dist_clauses)
    
    # Giải và kiểm tra
    with Solver(bootstrap_with=clauses, use_timer=True) as solver:
        is_sat = solver.solve()
        time = solver.time()
        print(f"\nKết quả SAT: {is_sat} (Thời gian: {time:.4f}s)")
        
        if not is_sat:
            print("LỖI: Mô hình không thể giải được!")
            return

        model = solver.get_model()
        true_literals = set(model)
        
        actual_dist = abs(u_pos - v_pos)
        print(f"Khoảng cách thực tế: {actual_dist}")
        
        print("Trạng thái thanh ghi T:")
        correct = True
        for d in range(1, n):
            t_d_var = T_vars[d-1]
            should_be_true = (actual_dist >= d)
            is_true_in_model = (t_d_var in true_literals)
            
            status = "ĐÚNG" if should_be_true == is_true_in_model else "SAI!"
            if status == "SAI!": correct = False
            
            print(f"  T_{d} (>= {d}):\t Nên là {should_be_true},\t Thực tế là {is_true_in_model}\t -> {status}")
            
        if correct:
            print(">>> KẾT LUẬN: Bộ mã hóa hoạt động chính xác.")
        else:
            print(">>> KẾT LUẬN: Bộ mã hóa có lỗi logic.")

if __name__ == '__main__':
    N_SIZE = 10
    print("="*60)
    print("KIỂM TRA BỘ MÃ HÓA THEO TRIẾT LÝ 'ĐẶT CHỐT & LAN TRUYỀN'")
    print("="*60)
    test_final_encoder(u_pos=5, v_pos=2, n=N_SIZE) # dist = 3
    test_final_encoder(u_pos=3, v_pos=8, n=N_SIZE) # dist = 5
    test_final_encoder(u_pos=7, v_pos=7, n=N_SIZE) # dist = 0