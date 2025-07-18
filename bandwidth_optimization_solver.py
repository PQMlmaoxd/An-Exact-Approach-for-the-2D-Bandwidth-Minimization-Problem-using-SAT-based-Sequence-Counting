# bandwidth_optimization_solver.py
# Giải bài toán 2D Bandwidth Minimization với SAT theo phương pháp UB-reduction

from pysat.formula import IDPool
from pysat.solvers import Glucose4, Glucose3, Solver
import random
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from distance_encoder import encode_abs_distance_final
    from random_assignment_ub_finder import RandomAssignmentUBFinder
except ImportError:
    # Nếu không tìm thấy, tạo một implementation đơn giản
    def encode_abs_distance_final(U_vars, V_vars, n, vpool):
        # Simplified version for testing
        T_vars = [vpool.id(f'T_geq_{d}') for d in range(1, n)]
        clauses = []
        return T_vars, clauses
    
    class RandomAssignmentUBFinder:
        def __init__(self, n, edges, seed=None):
            self.n = n
            self.edges = edges
        
        def find_ub_random_search(self, max_iterations=1000, time_limit=30):
            # Fallback implementation
            return {'ub': 2 * (self.n - 1), 'assignment': None, 'iterations': 0, 'time': 0}

class BandwidthOptimizationSolver:
    def __init__(self, n, solver_type='glucose4'):
        """
        Khởi tạo solver cho bài toán 2D Bandwidth Minimization
        
        Args:
            n: Kích thước bài toán (số đỉnh)
            solver_type: Loại SAT solver ('glucose4', 'glucose41', 'glucose3')
        """
        self.n = n
        self.solver_type = solver_type
        self.vpool = IDPool()
        
        # Tạo biến cho vị trí X và Y của mỗi đỉnh
        self.X_vars = {}  # X_vars[v][pos] = biến cho đỉnh v ở vị trí pos trên trục X
        self.Y_vars = {}  # Y_vars[v][pos] = biến cho đỉnh v ở vị trí pos trên trục Y
        
        # Tạo biến khoảng cách
        self.Tx_vars = {}  # Tx_vars[edge] = biến T cho khoảng cách X của cạnh
        self.Ty_vars = {}  # Ty_vars[edge] = biến T cho khoảng cách Y của cạnh
        
        # Edges của đồ thị (sẽ được set từ bên ngoài)
        self.edges = []
        
        print(f"Initialized BandwidthOptimizationSolver with n={n}, solver={solver_type}")
    
    def set_graph_edges(self, edges):
        """
        Thiết lập danh sách cạnh của đồ thị
        
        Args:
            edges: List of tuples [(u1,v1), (u2,v2), ...] 
        """
        self.edges = edges
        print(f"Set graph with {len(edges)} edges")
    
    def create_position_variables(self):
        """
        Tạo biến vị trí cho mỗi đỉnh trên trục X và Y
        """
        for v in range(1, self.n + 1):
            self.X_vars[v] = [self.vpool.id(f'X_{v}_{pos}') for pos in range(1, self.n + 1)]
            self.Y_vars[v] = [self.vpool.id(f'Y_{v}_{pos}') for pos in range(1, self.n + 1)]
    
    def create_distance_variables(self):
        """
        Tạo biến khoảng cách T cho mỗi cạnh
        """
        for i, (u, v) in enumerate(self.edges):
            edge_id = f'edge_{u}_{v}'
            # Tạo biến Tx và Ty cho cạnh này
            self.Tx_vars[edge_id] = [self.vpool.id(f'Tx_{edge_id}_{d}') for d in range(1, self.n)]
            self.Ty_vars[edge_id] = [self.vpool.id(f'Ty_{edge_id}_{d}') for d in range(1, self.n)]
    
    def encode_position_constraints(self):
        """
        Mã hóa ràng buộc: mỗi đỉnh có đúng một vị trí trên mỗi trục
        """
        clauses = []
        
        for v in range(1, self.n + 1):
            # Mỗi đỉnh có ít nhất một vị trí X
            clauses.append(self.X_vars[v][:])
            # Mỗi đỉnh có tối đa một vị trí X
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    clauses.append([-self.X_vars[v][i], -self.X_vars[v][j]])
            
            # Tương tự cho trục Y
            clauses.append(self.Y_vars[v][:])
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    clauses.append([-self.Y_vars[v][i], -self.Y_vars[v][j]])
        
        # Mỗi vị trí có tối đa một đỉnh
        for pos in range(self.n):
            # Trục X
            for i in range(1, self.n + 1):
                for j in range(i + 1, self.n + 1):
                    clauses.append([-self.X_vars[i][pos], -self.X_vars[j][pos]])
            # Trục Y  
            for i in range(1, self.n + 1):
                for j in range(i + 1, self.n + 1):
                    clauses.append([-self.Y_vars[i][pos], -self.Y_vars[j][pos]])
        
        return clauses
    
    def encode_distance_constraints(self):
        """
        Mã hóa ràng buộc khoảng cách cho mỗi cạnh bằng distance encoder
        """
        clauses = []
        
        for edge_id, (u, v) in zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges):
            # Encode khoảng cách X
            Tx_vars, Tx_clauses = encode_abs_distance_final(
                self.X_vars[u], self.X_vars[v], self.n, self.vpool
            )
            self.Tx_vars[edge_id] = Tx_vars
            clauses.extend(Tx_clauses)
            
            # Encode khoảng cách Y
            Ty_vars, Ty_clauses = encode_abs_distance_final(
                self.Y_vars[u], self.Y_vars[v], self.n, self.vpool
            )
            self.Ty_vars[edge_id] = Ty_vars
            clauses.extend(Ty_clauses)
        
        return clauses
    
    def step1_find_ub_random(self, max_iterations=1000, time_limit=30):
        """
        Bước 1: Tìm UB bằng phép gán ngẫu nhiên
        
        Args:
            max_iterations: Số lần thử tối đa
            time_limit: Thời gian giới hạn (seconds)
            
        Returns:
            dict: Kết quả với UB và thông tin chi tiết
        """
        print(f"\n=== STEP 1: Finding Upper Bound with Random Assignment ===")
        print(f"Strategy: Random search without SAT encoding")
        print(f"Max iterations: {max_iterations}, Time limit: {time_limit}s")
        
        # Tạo UB finder
        ub_finder = RandomAssignmentUBFinder(self.n, self.edges, seed=42)
        
        # Tìm UB bằng random search
        result = ub_finder.find_ub_random_search(max_iterations, time_limit)
        
        print(f"\nRandom UB Search Results:")
        print(f"Upper Bound found: {result['ub']}")
        print(f"Iterations used: {result['iterations']}")
        print(f"Time taken: {result['time']:.2f}s")
        
        # Visualize nếu tìm được assignment
        if result['assignment'] is not None:
            print(f"\nVisualizing best assignment:")
            ub_finder.visualize_assignment(result)
        
        return result
    
    def step1_find_ub_hybrid(self, random_iterations=500, greedy_tries=10):
        """
        Bước 1: Tìm UB bằng hybrid approach (random + greedy)
        
        Args:
            random_iterations: Số lần thử random
            greedy_tries: Số lần thử greedy
            
        Returns:
            dict: Kết quả với UB tốt nhất
        """
        print(f"\n=== STEP 1: Finding Upper Bound with Hybrid Approach ===")
        print(f"Strategy: Greedy + Random + Smart Sampling")
        
        # Tạo UB finder
        ub_finder = RandomAssignmentUBFinder(self.n, self.edges, seed=42)
        
        # Thử các phương pháp khác nhau
        results = []
        
        # 1. Random search
        print(f"\n1. Random Search ({random_iterations} iterations):")
        result1 = ub_finder.find_ub_random_search(random_iterations, time_limit=15)
        results.append(("Random Search", result1))
        
        # 2. Greedy + Random
        print(f"\n2. Greedy + Random ({greedy_tries} tries):")
        result2 = ub_finder.find_ub_greedy_random(greedy_tries, random_iterations//greedy_tries)
        results.append(("Greedy + Random", result2))
        
        # 3. Smart Sampling
        print(f"\n3. Smart Sampling:")
        result3 = ub_finder.find_ub_smart_sampling(random_iterations//4)
        results.append(("Smart Sampling", result3))
        
        # Tìm kết quả tốt nhất
        best_method, best_result = min(results, key=lambda x: x[1]['ub'])
        
        print(f"\n=== HYBRID SEARCH SUMMARY ===")
        for method, result in results:
            print(f"{method:15}: UB = {result['ub']:3d}, Time = {result['time']:.2f}s")
        
        print(f"\n🏆 BEST METHOD: {best_method}")
        print(f"🎯 BEST UB: {best_result['ub']}")
        
        # Visualize best result
        print(f"\nVisualizing best assignment:")
        ub_finder.visualize_assignment(best_result)
        
        return best_result
    
    def step1_test_upper_bound(self, K):
        """
        Bước 1: Test UB với K bằng SAT solver (phương pháp cũ)
        
        Args:
            K: Upper bound để test
            
        Returns:
            bool: True nếu có solution với K, False nếu không
        """
        print(f"\n=== STEP 1: Testing Upper Bound K={K} with SAT ===")
        print(f"Strategy: SAT solving with full distance encoding")
        
        # Tạo solver mới
        if self.solver_type == 'glucose4':
            solver = Glucose4()
        elif self.solver_type == 'glucose41':
            # Glucose4.1 có thể không có trong pysat, dùng Glucose4
            solver = Glucose4()
        else:
            solver = Glucose3()
        
        try:
            # Thêm các constraint cơ bản
            base_clauses = []
            base_clauses.extend(self.encode_position_constraints())
            base_clauses.extend(self.encode_distance_constraints())
            
            # Thêm constraint bandwidth <= K
            bandwidth_clauses = self.encode_bandwidth_constraint(K)
            base_clauses.extend(bandwidth_clauses)
            
            # Add clauses to solver
            for clause in base_clauses:
                solver.add_clause(clause)
            
            # Test
            result = solver.solve()
            
            if result:
                model = solver.get_model()
                print(f"✓ K={K} is feasible (found SAT solution)")
                return True
            else:
                print(f"✗ K={K} is not feasible")
                return False
                
        finally:
            solver.delete()
    
    def step1_combined_ub_search(self, use_random=True, use_sat_verification=True):
        """
        Bước 1: Combined UB search - Random + SAT verification
        
        Args:
            use_random: Có dùng random search không
            use_sat_verification: Có verify bằng SAT không
            
        Returns:
            dict: Kết quả với UB verified
        """
        print(f"\n=== STEP 1: COMBINED UB SEARCH ===")
        print(f"Strategy: Random Assignment + SAT Verification")
        
        # Phase 1: Random search để tìm UB candidate
        if use_random:
            print(f"\nPhase 1: Random search for UB candidate")
            ub_result = self.step1_find_ub_hybrid(random_iterations=800, greedy_tries=8)
            candidate_ub = ub_result['ub']
            
            print(f"Random search found UB candidate: {candidate_ub}")
        else:
            # Fallback: theoretical upper bound
            candidate_ub = 2 * (self.n - 1)
            print(f"Using theoretical UB: {candidate_ub}")
        
        # Phase 2: SAT verification
        if use_sat_verification:
            print(f"\nPhase 2: SAT verification of UB candidate")
            
            # Test từ candidate_ub xuống để tìm UB chính xác
            verified_ub = candidate_ub
            
            for K in range(candidate_ub, 0, -1):
                print(f"Verifying K={K} with SAT...")
                if self.step1_test_upper_bound(K):
                    verified_ub = K
                    print(f"✓ K={K} is verified as feasible UB")
                    break
                else:
                    print(f"✗ K={K} is not feasible")
                    verified_ub = K + 1
                    break
            
            print(f"SAT verification result: UB = {verified_ub}")
        else:
            verified_ub = candidate_ub
        
        print(f"\n🎯 FINAL UB: {verified_ub}")
        
        return {
            'ub': verified_ub,
            'candidate_ub': candidate_ub if use_random else None,
            'verified': use_sat_verification,
            'method': 'combined'
        }
        
    def step2_test_ub_minus_1(self, UB):
        """
        Bước 2: Test với K = UB - 1 và encode đầy đủ constraints
        
        Args:
            UB: Upper bound từ bước 1
            
        Returns:
            tuple: (is_sat, solver_state) để có thể dùng cho incremental
        """
        K = UB - 1
        print(f"\n=== STEP 2: Testing K={K} (UB-1) with full encoding ===")
        
        # Tạo solver mới
        if self.solver_type == 'glucose4':
            solver = Glucose4()
        elif self.solver_type == 'glucose41':
            solver = Glucose4()
        else:
            solver = Glucose3()
        
        # Encode đầy đủ constraints
        base_clauses = []
        base_clauses.extend(self.encode_position_constraints())
        base_clauses.extend(self.encode_distance_constraints())
        
        # Encode constraint phức tạp cho K
        advanced_constraints = self.encode_advanced_bandwidth_constraint(K)
        base_clauses.extend(advanced_constraints)
        
        # Add to solver
        for clause in base_clauses:
            solver.add_clause(clause)
        
        result = solver.solve()
        
        if result:
            model = solver.get_model()
            print(f"✓ K={K} is feasible with advanced constraints")
            return True, solver
        else:
            print(f"✗ K={K} is not feasible")
            solver.delete()
            return False, None
    
    def step3_incremental_search(self, UB, method='incremental'):
        """
        Bước 3: Tìm kiếm tối ưu từ UB-2 trở xuống
        
        Args:
            UB: Upper bound ban đầu
            method: 'incremental' hoặc 'new_solver'
            
        Returns:
            int: Bandwidth tối ưu tìm được
        """
        print(f"\n=== STEP 3: Optimization search using {method} method ===")
        
        # Test UB-1 trước
        is_feasible, base_solver = self.step2_test_ub_minus_1(UB)
        if not is_feasible:
            print(f"UB-1 = {UB-1} is not feasible, optimal is {UB}")
            return UB
        
        optimal_K = UB - 1
        
        if method == 'incremental':
            return self._incremental_search(UB, base_solver)
        else:
            if base_solver:
                base_solver.delete()
            return self._new_solver_search(UB)
    
    def _incremental_search(self, UB, base_solver):
        """
        Tìm kiếm incremental - sử dụng solver cũ và thêm constraints
        """
        print("Using INCREMENTAL search method")
        optimal_K = UB - 1
        
        try:
            for K in range(UB - 2, 0, -1):
                print(f"Testing K={K} incrementally...")
                
                # Thêm constraint mới cho K
                new_constraints = self.encode_k_constraint_incremental(K)
                
                # Add new constraints to existing solver
                for clause in new_constraints:
                    base_solver.add_clause(clause)
                
                result = base_solver.solve()
                
                if result:
                    model = base_solver.get_model()
                    optimal_K = K
                    print(f"✓ K={K} is feasible")
                else:
                    print(f"✗ K={K} is not feasible")
                    break
                    
        finally:
            base_solver.delete()
            
        print(f"Incremental search found optimal K = {optimal_K}")
        return optimal_K
    
    def _new_solver_search(self, UB):
        """
        Tìm kiếm với solver mới cho mỗi K
        """
        print("Using NEW SOLVER method")
        optimal_K = UB - 1
        
        for K in range(UB - 2, 0, -1):
            print(f"Testing K={K} with new solver...")
            
            # Tạo solver hoàn toàn mới
            if self.solver_type == 'glucose4':
                solver = Glucose4()
            elif self.solver_type == 'glucose41':
                solver = Glucose4()
            else:
                solver = Glucose3()
            
            try:
                # Encode lại tất cả constraints
                all_clauses = []
                all_clauses.extend(self.encode_position_constraints())
                all_clauses.extend(self.encode_distance_constraints())
                all_clauses.extend(self.encode_advanced_bandwidth_constraint(K))
                
                # Add to solver
                for clause in all_clauses:
                    solver.add_clause(clause)
                
                result = solver.solve()
                
                if result:
                    model = solver.get_model()
                    optimal_K = K
                    print(f"✓ K={K} is feasible")
                else:
                    print(f"✗ K={K} is not feasible")
                    break
                    
            finally:
                solver.delete()
        
        print(f"New solver search found optimal K = {optimal_K}")
        return optimal_K
    
    def encode_bandwidth_constraint(self, K):
        """
        Encode constraint đơn giản: max distance <= K
        """
        clauses = []
        
        for edge_id in self.Tx_vars:
            # Tx <= K  <=> ¬Tx_{K+1}
            if K < len(self.Tx_vars[edge_id]):
                clauses.append([-self.Tx_vars[edge_id][K]])
            
            # Ty <= K  <=> ¬Ty_{K+1}  
            if K < len(self.Ty_vars[edge_id]):
                clauses.append([-self.Ty_vars[edge_id][K]])
        
        return clauses
    
    def encode_advanced_bandwidth_constraint(self, K):
        """
        Encode constraint phức tạp:
        (Tx<=K) ∧ (Ty<=K) ∧ (Tx>=1 → Ty<=K-1) ∧ (Tx>=2 → Ty<=K-2) ∧ ... ∧ (Tx=K → Ty<=0)
        """
        clauses = []
        
        for edge_id in self.Tx_vars:
            Tx = self.Tx_vars[edge_id] 
            Ty = self.Ty_vars[edge_id]
            
            # (Tx <= K) ∧ (Ty <= K)
            if K < len(Tx):
                clauses.append([-Tx[K]])  # ¬Tx_{K+1}
            if K < len(Ty):
                clauses.append([-Ty[K]])  # ¬Ty_{K+1}
            
            # (Tx >= i → Ty <= K-i) cho i = 1, 2, ..., K
            for i in range(1, min(K + 1, len(Tx) + 1)):
                # Tx >= i  <=> Tx_i = True
                # Ty <= K-i <=> ¬Ty_{K-i+1}
                
                if i-1 < len(Tx) and K-i >= 0 and K-i < len(Ty):
                    # (Tx >= i → Ty <= K-i) <=> (¬Tx_i ∨ ¬Ty_{K-i+1})
                    clauses.append([-Tx[i-1], -Ty[K-i]])
        
        return clauses
    
    def encode_k_constraint_incremental(self, K):
        """
        Encode constraint cho K mới trong incremental search
        Chỉ cần thêm constraint tighter hơn K cũ
        """
        clauses = []
        
        for edge_id in self.Tx_vars:
            # Thêm constraint nghiêm ngặt hơn cho K mới
            if K >= 0 and K < len(self.Tx_vars[edge_id]):
                clauses.append([-self.Tx_vars[edge_id][K]])
            if K >= 0 and K < len(self.Ty_vars[edge_id]):
                clauses.append([-self.Ty_vars[edge_id][K]])
        
        return clauses

def test_bandwidth_solver():
    """
    Test function cho BandwidthOptimizationSolver với random UB finder
    """
    print("=== TESTING BANDWIDTH OPTIMIZATION SOLVER WITH RANDOM UB ===")
    
    # Tạo đồ thị test
    n = 5
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]  # Cycle graph
    
    # Test với Glucose4
    solver = BandwidthOptimizationSolver(n, 'glucose4')
    solver.set_graph_edges(edges)
    solver.create_position_variables()
    solver.create_distance_variables()
    
    # Bước 1: Tìm UB bằng random assignment
    print("\n" + "="*60)
    print("TESTING DIFFERENT UB FINDING METHODS")
    print("="*60)
    
    # Method 1: Pure random search
    print("\nMethod 1: Pure Random Search")
    ub_result1 = solver.step1_find_ub_random(max_iterations=500, time_limit=10)
    
    # Method 2: Hybrid approach  
    print("\nMethod 2: Hybrid Approach")
    ub_result2 = solver.step1_find_ub_hybrid(random_iterations=300, greedy_tries=5)
    
    # Method 3: Combined (random + SAT verification)
    print("\nMethod 3: Combined Random + SAT")
    ub_result3 = solver.step1_combined_ub_search(use_random=True, use_sat_verification=True)
    
    # So sánh kết quả
    print("\n" + "="*60)
    print("UB FINDING METHODS COMPARISON")
    print("="*60)
    
    print(f"Pure Random Search:      UB = {ub_result1['ub']}")
    print(f"Hybrid Approach:         UB = {ub_result2['ub']}")
    print(f"Combined (Random + SAT): UB = {ub_result3['ub']}")
    
    # Chọn UB tốt nhất để tiếp tục
    best_ub = min(ub_result1['ub'], ub_result2['ub'], ub_result3['ub'])
    print(f"\nBest UB found: {best_ub}")
    
    # Tiếp tục với bước 2 và 3
    print("\n" + "="*60)
    print("CONTINUING WITH OPTIMIZATION")
    print("="*60)
    
    # Bước 2 & 3: Optimization với UB tìm được
    if best_ub < 2 * (n - 1):  # Nếu UB tốt hơn theoretical bound
        print(f"\nProceeding with optimization from UB = {best_ub}")
        
        # Test cả 2 phương pháp
        print("\nTesting Incremental method:")
        optimal_incremental = solver.step3_incremental_search(best_ub + 1, 'incremental')
        
        print("\nTesting New Solver method:")
        optimal_new = solver.step3_incremental_search(best_ub + 1, 'new_solver')
        
        print(f"\nFinal Results:")
        print(f"Incremental method: Optimal K = {optimal_incremental}")
        print(f"New solver method:  Optimal K = {optimal_new}")
    else:
        print(f"UB {best_ub} is not better than theoretical bound {2 * (n - 1)}")
        print("Consider using larger graph or different approach")

def test_random_ub_standalone():
    """
    Test standalone random UB finder
    """
    print("=== TESTING STANDALONE RANDOM UB FINDER ===")
    
    # Test với đồ thị lớn hơn
    n = 6
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (1, 4), (2, 5)]
    
    ub_finder = RandomAssignmentUBFinder(n, edges, seed=42)
    
    # Test random search
    result = ub_finder.find_ub_random_search(max_iterations=1000, time_limit=15)
    ub_finder.visualize_assignment(result)
    
    print(f"\nRandom search result: UB = {result['ub']}")
    
    # Test hybrid
    hybrid_result = ub_finder.find_ub_hybrid_search(num_greedy_tries=5, random_tries_per_greedy=100)
    ub_finder.visualize_assignment(hybrid_result)
    
    print(f"Hybrid search result: UB = {hybrid_result['ub']}")

if __name__ == '__main__':
    # Test main solver
    test_bandwidth_solver()
    
    # Test standalone UB finder
    print("\n" + "="*80)
    # test_random_ub_standalone()  # Uncomment to test standalone
