# 2dbmp_solver_hybrid.py

from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from itertools import combinations

# Import các hàm mã hóa NSC
from nsc_encoder import (
    encode_nsc_exactly_k, 
    encode_nsc_at_most_k
)

class TwoDBMP_Solver_Hybrid:
    def __init__(self, num_vertices, grid_size, edges):
        self.n = num_vertices
        self.s = grid_size
        self.edges = edges
        self.vpool = IDPool()
        
        # Chọn phương pháp encode dựa trên kích thước bài toán
        self.encoding_method = self._choose_encoding_method()
        print(f"Sử dụng phương pháp encode: {self.encoding_method}")
        
        # Tạo biến cơ sở
        self._create_base_variables()
    
    def _choose_encoding_method(self):
        """Chọn phương pháp encode dựa trên kích thước bài toán."""
        total_vars = self.n * self.s * self.s
        total_edges = len(self.edges)
        
        # Heuristic: Nếu bài toán nhỏ, dùng direct; nếu lớn, dùng sequential
        if self.s <= 4 and total_edges <= 20:
            return "direct"
        else:
            return "sequential"
    
    def _create_base_variables(self):
        """Tạo các biến cơ sở cho cả hai phương pháp."""
        # Biến vị trí (chung cho cả hai phương pháp)
        for v in range(1, self.n + 1):
            for r in range(1, self.s + 1):
                for c in range(1, self.s + 1):
                    self._var_pos(v, r, c)
        
        # Biến phụ cho sequential encoding
        if self.encoding_method == "sequential":
            # Biến cho sequential counter tọa độ
            for v in range(1, self.n + 1):
                for i in range(1, self.s + 1):
                    self._var_coord_x(v, i)
                    self._var_coord_y(v, i)
            
            # Biến khoảng cách
            for u, v in self.edges:
                for d in range(1, self.s):
                    self._var_dist_x(u, v, d)
                    self._var_dist_y(u, v, d)
    
    def _var_pos(self, v, r, c):
        """Biến vị trí: đỉnh v có ở vị trí (r,c)."""
        return self.vpool.id(f'pos_{v}_{r}_{c}')

    def _var_coord_x(self, v, i):
        """Biến tọa độ x: đỉnh v có tọa độ x >= i."""
        return self.vpool.id(f'coord_x_{v}_{i}')
    
    def _var_coord_y(self, v, i):
        """Biến tọa độ y: đỉnh v có tọa độ y >= i."""
        return self.vpool.id(f'coord_y_{v}_{i}')
    
    def _var_dist_x(self, u, v, d):
        """Biến khoảng cách x: |xu - xv| >= d."""
        return self.vpool.id(f'dist_x_{u}_{v}_{d}')
    
    def _var_dist_y(self, u, v, d):
        """Biến khoảng cách y: |yu - yv| >= d."""
        return self.vpool.id(f'dist_y_{u}_{v}_{d}')

    def _get_structural_clauses(self):
        """Tạo ràng buộc cấu trúc sử dụng NSC encoder."""
        clauses = []
        print("Bắt đầu mã hóa ràng buộc cấu trúc với NSC encoder...")
        
        # Ràng buộc 1: Mỗi đỉnh có đúng một vị trí
        for v in range(1, self.n + 1):
            vars_for_v = []
            for r in range(1, self.s + 1):
                for c in range(1, self.s + 1):
                    vars_for_v.append(self._var_pos(v, r, c))
            
            # Exactly-one using NSC
            clauses.extend(encode_nsc_exactly_k(vars_for_v, 1, self.vpool))
        
        # Ràng buộc 2: Mỗi vị trí có nhiều nhất một đỉnh
        for r in range(1, self.s + 1):
            for c in range(1, self.s + 1):
                vars_at_pos = [self._var_pos(v, r, c) for v in range(1, self.n + 1)]
                
                # At-most-one using NSC
                clauses.extend(encode_nsc_at_most_k(vars_at_pos, 1, self.vpool))
        
        print(f"... Hoàn thành ràng buộc cấu trúc với NSC. Đã tạo {len(clauses)} mệnh đề.")
        return clauses

    def _get_bandwidth_clauses_direct(self, K):
        """Phương pháp Direct Encoding cho ràng buộc băng thông."""
        clauses = []
        print(f"Bắt đầu mã hóa băng thông K={K} (Direct method)...")
        
        for u, v in self.edges:
            for r1 in range(1, self.s + 1):
                for c1 in range(1, self.s + 1):
                    for r2 in range(1, self.s + 1):
                        for c2 in range(1, self.s + 1):
                            if (r1, c1) != (r2, c2):
                                manhattan_dist = abs(r1 - r2) + abs(c1 - c2)
                                if manhattan_dist > K:
                                    # Không thể cùng lúc u ở (r1,c1) và v ở (r2,c2)
                                    clauses.append([-self._var_pos(u, r1, c1), -self._var_pos(v, r2, c2)])
        
        print(f"... Hoàn thành băng thông (Direct). Đã tạo {len(clauses)} mệnh đề.")
        return clauses

    def _get_bandwidth_clauses_sequential(self, K):
        """Phương pháp Sequential Encoding cho ràng buộc băng thông."""
        clauses = []
        print(f"Bắt đầu mã hóa băng thông K={K} (Sequential method)...")
        
        # Bước 1: Định nghĩa biến tọa độ
        clauses.extend(self._encode_coordinate_definitions())
        
        # Bước 2: Định nghĩa biến khoảng cách
        clauses.extend(self._encode_distance_definitions())
        
        # Bước 3: Ràng buộc băng thông
        clauses.extend(self._encode_bandwidth_constraints(K))
        
        print(f"... Hoàn thành băng thông (Sequential). Đã tạo {len(clauses)} mệnh đề.")
        return clauses

    def _encode_coordinate_definitions(self):
        """Định nghĩa biến tọa độ dựa trên vị trí."""
        clauses = []
        
        for v in range(1, self.n + 1):
            # Định nghĩa coord_x[v][i]: đỉnh v có tọa độ x >= i
            for i in range(1, self.s + 1):
                vars_x_geq_i = []
                for r in range(i, self.s + 1):  # Hàng >= i
                    for c in range(1, self.s + 1):
                        vars_x_geq_i.append(self._var_pos(v, r, c))
                
                if vars_x_geq_i:
                    # coord_x[v][i] ↔ (v ở hàng >= i)
                    clauses.append([-self._var_coord_x(v, i)] + vars_x_geq_i)
                    for var in vars_x_geq_i:
                        clauses.append([-var, self._var_coord_x(v, i)])
                else:
                    # Không có hàng nào >= i
                    clauses.append([-self._var_coord_x(v, i)])
            
            # Định nghĩa coord_y[v][j]: đỉnh v có tọa độ y >= j
            for j in range(1, self.s + 1):
                vars_y_geq_j = []
                for r in range(1, self.s + 1):
                    for c in range(j, self.s + 1):  # Cột >= j
                        vars_y_geq_j.append(self._var_pos(v, r, c))
                
                if vars_y_geq_j:
                    clauses.append([-self._var_coord_y(v, j)] + vars_y_geq_j)
                    for var in vars_y_geq_j:
                        clauses.append([-var, self._var_coord_y(v, j)])
                else:
                    clauses.append([-self._var_coord_y(v, j)])
        
        return clauses

    def _encode_distance_definitions(self):
        """Định nghĩa biến khoảng cách dựa trên biến tọa độ."""
        clauses = []
        
        for u, v in self.edges:
            for d in range(1, self.s):
                # dist_x[u][v][d] ↔ |xu - xv| >= d
                # Điều này xảy ra khi:
                # (xu >= xv + d) hoặc (xv >= xu + d)
                # Tức là: (coord_x[u][xv + d] ∧ pos[v][xv][*]) hoặc (coord_x[v][xu + d] ∧ pos[u][xu][*])
                
                dist_x_conditions = []
                
                # Trường hợp 1: xu >= xv + d
                for xv in range(1, self.s + 1):
                    if xv + d <= self.s:
                        for c in range(1, self.s + 1):
                            pos_v = self._var_pos(v, xv, c)
                            coord_u = self._var_coord_x(u, xv + d)
                            # pos_v ∧ coord_u → dist_x[u][v][d]
                            clauses.append([-pos_v, -coord_u, self._var_dist_x(u, v, d)])
                            dist_x_conditions.append((pos_v, coord_u))
                
                # Trường hợp 2: xv >= xu + d
                for xu in range(1, self.s + 1):
                    if xu + d <= self.s:
                        for c in range(1, self.s + 1):
                            pos_u = self._var_pos(u, xu, c)
                            coord_v = self._var_coord_x(v, xu + d)
                            clauses.append([-pos_u, -coord_v, self._var_dist_x(u, v, d)])
                            dist_x_conditions.append((pos_u, coord_v))
                
                # Tương tự cho dist_y
                for yu in range(1, self.s + 1):
                    if yu + d <= self.s:
                        for r in range(1, self.s + 1):
                            pos_u = self._var_pos(u, r, yu)
                            coord_v = self._var_coord_y(v, yu + d)
                            clauses.append([-pos_u, -coord_v, self._var_dist_y(u, v, d)])
                
                for yv in range(1, self.s + 1):
                    if yv + d <= self.s:
                        for r in range(1, self.s + 1):
                            pos_v = self._var_pos(v, r, yv)
                            coord_u = self._var_coord_y(u, yv + d)
                            clauses.append([-pos_v, -coord_u, self._var_dist_y(u, v, d)])
        
        return clauses

    def _encode_bandwidth_constraints(self, K):
        """Encode ràng buộc băng thông: dist_x + dist_y <= K."""
        clauses = []
        
        for u, v in self.edges:
            # |xu - xv| + |yu - yv| <= K
            # Nghĩa là: ¬(dist_x[u][v][a] ∧ dist_y[u][v][b]) với a + b > K
            for a in range(1, self.s):
                for b in range(1, self.s):
                    if a + b > K:
                        clauses.append([-self._var_dist_x(u, v, a), -self._var_dist_y(u, v, b)])
        
        return clauses

    def _get_bandwidth_clauses(self, K):
        """Dispatcher cho phương pháp encode băng thông."""
        if self.encoding_method == "direct":
            return self._get_bandwidth_clauses_direct(K)
        else:
            return self._get_bandwidth_clauses_sequential(K)

    def find_optimal_bandwidth(self):
        """Tìm băng thông tối ưu."""
        structural_clauses = self._get_structural_clauses()
        
        print(f"\nThống kê encoding method: {self.encoding_method}")
        print(f"Số biến cơ sở: {self.n * self.s * self.s}")
        print(f"Số biến phụ: {self.vpool.top - self.n * self.s * self.s}")
        print(f"Tổng số biến: {self.vpool.top}")
        
        for k_to_test in range(1, self.s * 2):
            print(f"\n=== Kiểm tra với K = {k_to_test} ===")
            
            bandwidth_clauses = self._get_bandwidth_clauses(k_to_test)
            all_clauses = structural_clauses + bandwidth_clauses
            
            print(f"Tổng số mệnh đề: {len(all_clauses)}")
            
            with Solver(name='g3', bootstrap_with=all_clauses) as solver:
                is_sat = solver.solve()
                if is_sat:
                    model = solver.get_model()
                    print(f"Tìm thấy SAT với {len(model)} literals")
                    
                    if self.verify_solution(model, k_to_test):
                        print(f"Tìm thấy lời giải tối ưu với K = {k_to_test}")
                        self.print_solution(model)
                        return k_to_test, model
                    else:
                        print("Lời giải không hợp lệ")
                else:
                    print(f"UNSAT cho K = {k_to_test}")
        
        print("\nKhông tìm thấy lời giải.")
        return -1, None

    def verify_solution(self, model, k):
        """Kiểm tra tính hợp lệ của lời giải."""
        if not model:
            return False
        
        true_literals = set(model)
        vertex_pos = {}
        
        # Kiểm tra mỗi đỉnh có đúng một vị trí
        for v in range(1, self.n + 1):
            positions = []
            for r in range(1, self.s + 1):
                for c in range(1, self.s + 1):
                    if self._var_pos(v, r, c) in true_literals:
                        positions.append((r, c))
            
            if len(positions) != 1:
                print(f"Lỗi: Đỉnh {v} có {len(positions)} vị trí")
                return False
            
            vertex_pos[v] = positions[0]
        
        # Kiểm tra ràng buộc băng thông
        for u, v in self.edges:
            if u not in vertex_pos or v not in vertex_pos:
                print(f"Lỗi: Đỉnh {u} hoặc {v} không có vị trí")
                return False
            
            r1, c1 = vertex_pos[u]
            r2, c2 = vertex_pos[v]
            dist = abs(r1 - r2) + abs(c1 - c2)
            
            if dist > k:
                print(f"Lỗi: Cạnh ({u},{v}) có khoảng cách {dist} > {k}")
                return False
        
        return True

    def print_solution(self, model):
        """In lời giải."""
        if not model:
            print("Không có lời giải.")
            return
        
        true_literals = set(model)
        grid = [['.' for _ in range(self.s)] for _ in range(self.s)]
        
        for v in range(1, self.n + 1):
            for r in range(1, self.s + 1):
                for c in range(1, self.s + 1):
                    if self._var_pos(v, r, c) in true_literals:
                        grid[r-1][c-1] = str(v)
        
        print("\nLời giải tìm được:")
        for row in grid:
            print(" ".join(row))

# Test hybrid approach
if __name__ == '__main__':
    # Test case 1: Bài toán nhỏ (sử dụng direct encoding)
    print("=== TEST CASE 1: Bài toán nhỏ ===")
    NUM_VERTICES_1 = 4
    GRID_SIZE_1 = 2
    EDGES_1 = [(1, 2), (2, 3), (3, 4), (4, 1)]
    
    solver1 = TwoDBMP_Solver_Hybrid(NUM_VERTICES_1, GRID_SIZE_1, EDGES_1)
    optimal_K1, solution1 = solver1.find_optimal_bandwidth()
    
    print(f"\nKết quả test case 1: K = {optimal_K1}")
    
    # Test case 2: Bài toán lớn hơn (sử dụng sequential encoding)
    print("\n" + "="*50)
    print("=== TEST CASE 2: Bài toán lớn hơn ===")
    NUM_VERTICES_2 = 9
    GRID_SIZE_2 = 3
    EDGES_2 = [(i, i + 1) for i in range(1, NUM_VERTICES_2)] + [(NUM_VERTICES_2, 1)]
    
    solver2 = TwoDBMP_Solver_Hybrid(NUM_VERTICES_2, GRID_SIZE_2, EDGES_2)
    optimal_K2, solution2 = solver2.find_optimal_bandwidth()
    
    print(f"\nKết quả test case 2: K = {optimal_K2}")
    
    # So sánh hiệu suất
    print("\n" + "="*50)
    print("=== SO SÁNH HIỆU SUẤT ===")
    print(f"Test case 1 (Direct): Biến={solver1.vpool.top}")
    print(f"Test case 2 (Sequential): Biến={solver2.vpool.top}")

