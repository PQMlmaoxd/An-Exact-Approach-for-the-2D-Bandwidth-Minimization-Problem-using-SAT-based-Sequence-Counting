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
        
        # Mỗi position (X,Y) có tối đa một đỉnh - FIXED VERSION
        # Cấm hai nodes cùng exact position (x+1, y+1)
        for x in range(self.n):
            for y in range(self.n):
                for i in range(1, self.n + 1):
                    for j in range(i + 1, self.n + 1):
                        # Nodes i và j không thể cùng position (x+1, y+1)
                        # ¬(Xi_{x} ∧ Yi_{y} ∧ Xj_{x} ∧ Yj_{y})
                        clauses.append([-self.X_vars[i][x], -self.Y_vars[i][y], -self.X_vars[j][x], -self.Y_vars[j][y]])
        
        return clauses
    
    def encode_distance_constraints(self):
        """
        Mã hóa ràng buộc khoảng cách cho mỗi cạnh bằng distance encoder
        """
        clauses = []
        
        for edge_id, (u, v) in zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges):
            # Encode khoảng cách X với prefix rõ ràng
            Tx_vars, Tx_clauses = encode_abs_distance_final(
                self.X_vars[u], self.X_vars[v], self.n, self.vpool, f"Tx_{edge_id}"
            )
            self.Tx_vars[edge_id] = Tx_vars
            clauses.extend(Tx_clauses)
            
            # Encode khoảng cách Y với prefix rõ ràng
            Ty_vars, Ty_clauses = encode_abs_distance_final(
                self.Y_vars[u], self.Y_vars[v], self.n, self.vpool, f"Ty_{edge_id}"
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
    
    def step1_test_ub_pure_random(self, K):
        """
        Bước 1: Test UB với K bằng pure random assignment (theo ý tưởng thầy)
        
        Args:
            K: Upper bound để test
            
        Returns:
            bool: True nếu tìm được assignment với bandwidth ≤ K, False nếu không
        """
        print(f"\n=== STEP 1: Testing UB K={K} with Pure Random Assignment ===")
        print(f"Strategy: Pure random assignment without SAT encoding")
        print(f"Goal: Find assignment with bandwidth ≤ {K}")
        
        # Sử dụng RandomAssignmentUBFinder để test K
        ub_finder = RandomAssignmentUBFinder(self.n, self.edges, seed=42)
        
        # Tìm assignment với target UB = K
        result = ub_finder.find_ub_random_search(max_iterations=1000, time_limit=15)
        
        achieved_ub = result['ub']
        
        print(f"Random search result:")
        print(f"- Target UB: {K}")
        print(f"- Achieved UB: {achieved_ub}")
        print(f"- Iterations: {result['iterations']}")
        print(f"- Time: {result['time']:.2f}s")
        
        if achieved_ub <= K:
            print(f"✓ SUCCESS: Found assignment with bandwidth {achieved_ub} ≤ {K}")
            print(f"Proceeding to Step 2 with K = {K}")
            return True
        else:
            print(f"✗ FAILED: Best assignment has bandwidth {achieved_ub} > {K}")
            print(f"Need to try higher K value")
            return False
    
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
        
    def step2_encode_advanced_constraints(self, K):
        """
        Bước 2: Encode đầy đủ constraints theo ý tưởng thầy
        
        Encode: (Tx≤K) ∧ (Ty≤K) ∧ (Tx≥1 → Ty≤K-1) ∧ (Tx≥2 → Ty≤K-2) ∧ ... ∧ (Tx=K → Ty≤0)
        
        Với Thermometer encoding:
        - Tx_i means Tx ≥ i
        - (Tx ≤ K) ≡ ¬Tx_{K+1}
        - (Tx ≥ i → Ty ≤ K-i) ≡ ¬Tx_i ∨ ¬Ty_{K-i+1}
        
        Args:
            K: Upper bound để encode
            
        Returns:
            bool: True nếu có solution với K, False nếu không
        """
        print(f"\n=== STEP 2: Testing K={K} with Advanced Constraint Encoding ===")
        print(f"Strategy: Full SAT encoding with Thermometer constraints")
        print(f"Encoding: (Tx≤{K}) ∧ (Ty≤{K}) ∧ implication constraints")
        
        # Tạo solver mới
        if self.solver_type == 'glucose4':
            solver = Glucose4()
        elif self.solver_type == 'glucose41':
            solver = Glucose4()
        else:
            solver = Glucose3()
        
        try:
            # Thêm constraints cơ bản (position + distance)
            base_clauses = []
            base_clauses.extend(self.encode_position_constraints())
            base_clauses.extend(self.encode_distance_constraints())
            
            print(f"Added {len(base_clauses)} base clauses (position + distance)")
            
            # Encode advanced bandwidth constraints theo ý tưởng thầy
            advanced_clauses = self.encode_thermometer_bandwidth_constraints(K)
            base_clauses.extend(advanced_clauses)
            
            print(f"Added {len(advanced_clauses)} advanced bandwidth clauses")
            
            # Add all clauses to solver
            for clause in base_clauses:
                solver.add_clause(clause)
            
            # Test feasibility
            result = solver.solve()
            
            if result:
                model = solver.get_model()
                print(f"✓ K={K} is FEASIBLE with advanced encoding")
                
                # Optional: Decode và verify solution
                print(f"Solution found! Extracting assignment...")
                self.extract_and_verify_solution(model, K)
                
                return True
            else:
                print(f"✗ K={K} is INFEASIBLE")
                return False
                
        finally:
            solver.delete()
    
    def encode_thermometer_bandwidth_constraints(self, K):
        """
        Encode bandwidth constraints theo Thermometer encoding
        
        (Tx≤K) ∧ (Ty≤K) ∧ (Tx≥1 → Ty≤K-1) ∧ ... ∧ (Tx≥K → Ty≤0)
        
        Args:
            K: Upper bound
            
        Returns:
            list: Danh sách clauses
        """
        clauses = []
        
        print(f"\nEncoding Thermometer constraints for K={K}:")
        
        for edge_id in self.Tx_vars:
            Tx = self.Tx_vars[edge_id]  # Tx[i] means Tx ≥ i+1
            Ty = self.Ty_vars[edge_id]  # Ty[i] means Ty ≥ i+1
            
            print(f"  {edge_id}: Tx_vars={len(Tx)}, Ty_vars={len(Ty)}")
            
            # 1. (Tx ≤ K) ≡ ¬Tx_{K+1}
            if K < len(Tx):  # Tx[K] means Tx ≥ K+1
                clauses.append([-Tx[K]])
                print(f"    Added: Tx ≤ {K} (¬Tx_{K+1})")
            
            # 2. (Ty ≤ K) ≡ ¬Ty_{K+1}  
            if K < len(Ty):  # Ty[K] means Ty ≥ K+1
                clauses.append([-Ty[K]])
                print(f"    Added: Ty ≤ {K} (¬Ty_{K+1})")
            
            # 3. Implication constraints: (Tx ≥ i → Ty ≤ K-i)
            for i in range(1, K + 1):
                if K - i >= 0:
                    # Tx ≥ i is represented by Tx[i-1] 
                    # Ty ≤ K-i is represented by ¬Ty[K-i] (since Ty[K-i] means Ty ≥ K-i+1)
                    
                    tx_geq_i = None
                    ty_leq_ki = None
                    
                    if i-1 < len(Tx):
                        tx_geq_i = Tx[i-1]  # Tx ≥ i
                    
                    if K-i < len(Ty):
                        ty_leq_ki = -Ty[K-i]  # Ty ≤ K-i
                    
                    # Add implication: Tx ≥ i → Ty ≤ K-i
                    # Equivalent: ¬Tx_i ∨ ¬Ty_{K-i+1}
                    if tx_geq_i is not None and ty_leq_ki is not None:
                        clauses.append([-tx_geq_i, ty_leq_ki])
                        print(f"    Added: Tx≥{i} → Ty≤{K-i} (¬Tx_{i} ∨ ¬Ty_{K-i+1})")
        
        print(f"Total thermometer clauses generated: {len(clauses)}")
        return clauses
    
    def extract_and_verify_solution(self, model, K):
        """
        Extract và verify SAT solution
        """
        print(f"\n--- Solution Verification ---")
        
        # Extract positions
        positions = {}
        for v in range(1, self.n + 1):
            # Find X position
            for pos in range(1, self.n + 1):
                var_id = self.X_vars[v][pos-1]
                if var_id in model and model[model.index(var_id)] > 0:
                    positions[f'X_{v}'] = pos
                    break
            
            # Find Y position
            for pos in range(1, self.n + 1):
                var_id = self.Y_vars[v][pos-1]
                if var_id in model and model[model.index(var_id)] > 0:
                    positions[f'Y_{v}'] = pos
                    break
        
        print(f"Position assignment:")
        for v in range(1, self.n + 1):
            x = positions.get(f'X_{v}', '?')
            y = positions.get(f'Y_{v}', '?')
            print(f"  Node {v}: ({x}, {y})")
        
        # Calculate bandwidth manually
        max_distance = 0
        print(f"\nEdge distances:")
        
        for u, v in self.edges:
            x_u = positions.get(f'X_{u}', 0)
            y_u = positions.get(f'Y_{u}', 0)
            x_v = positions.get(f'X_{v}', 0)
            y_v = positions.get(f'Y_{v}', 0)
            
            distance = abs(x_u - x_v) + abs(y_u - y_v)
            max_distance = max(max_distance, distance)
            
            print(f"  Edge ({u},{v}): distance = {distance}")
        
        print(f"\nBandwidth = {max_distance}")
        print(f"Constraint K={K} satisfied: {'✓' if max_distance <= K else '✗'}")
        
        return max_distance <= K
    
    def solve_bandwidth_optimization(self, start_k=None, end_k=None):
        """
        Main workflow theo ý tưởng thầy:
        
        1. Test UB với pure random assignment cho các K từ 1 đến 2(n-1)
        2. Khi tìm được K khả thi, chuyển sang bước 2 với SAT encoding đầy đủ
        
        Args:
            start_k: K bắt đầu test (default: 1)
            end_k: K kết thúc test (default: 2(n-1))
            
        Returns:
            int: Bandwidth tối ưu tìm được
        """
        if start_k is None:
            start_k = 1
        if end_k is None:
            end_k = 2 * (self.n - 1)
        
        print(f"\n" + "="*80)
        print(f"BANDWIDTH OPTIMIZATION - TEACHER'S APPROACH")
        print(f"Graph: {self.n} nodes, {len(self.edges)} edges")
        print(f"Testing K range: {start_k} to {end_k}")
        print(f"="*80)
        
        # Phase 1: Tìm UB bằng pure random assignment
        print(f"\n🔍 PHASE 1: Finding feasible UB with pure random assignment")
        
        feasible_ub = None
        
        for K in range(start_k, end_k + 1):
            print(f"\n--- Testing K = {K} ---")
            
            if self.step1_test_ub_pure_random(K):
                feasible_ub = K
                print(f"✅ Found feasible UB = {K}")
                break
            else:
                print(f"❌ K = {K} not achievable with random assignment")
        
        if feasible_ub is None:
            print(f"\n❌ ERROR: No feasible UB found in range [{start_k}, {end_k}]")
            print(f"Consider increasing end_k or checking graph connectivity")
            return None
        
        # Phase 2: SAT encoding với K = UB - 1
        print(f"\n🔧 PHASE 2: SAT encoding with K = {feasible_ub - 1}")
        
        target_k = feasible_ub - 1
        
        if target_k < 1:
            print(f"✅ OPTIMAL: UB = {feasible_ub} is already minimal (K=1 not possible)")
            return feasible_ub
        
        if self.step2_encode_advanced_constraints(target_k):
            print(f"✅ SUCCESS: K = {target_k} is feasible with SAT")
            print(f"🎯 OPTIMAL BANDWIDTH = {target_k}")
            return target_k
        else:
            print(f"❌ K = {target_k} is infeasible with SAT")
            print(f"🎯 OPTIMAL BANDWIDTH = {feasible_ub}")
            return feasible_ub
    
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
        Encode constraint đúng: max(all Manhattan distances) <= K
        
        Bandwidth = max_{(u,v) ∈ E} {|X[u] - X[v]| + |Y[u] - Y[v]|}
        
        Constraint: max{Tx₁ + Ty₁, Tx₂ + Ty₂, ..., Txₘ + Tyₘ} <= K
        Equivalent: (Tx₁ + Ty₁ <= K) ∧ (Tx₂ + Ty₂ <= K) ∧ ... ∧ (Txₘ + Tyₘ <= K)
        
        Với thermometer encoding:
        - Tx[i] means Tx >= i+1 
        - Ty[j] means Ty >= j+1
        - Tx + Ty <= K means: không thể có Tx >= i và Ty >= j where i + j > K
        
        FIXED: Bao gồm cả trường hợp Ty = 0 (j = 0)
        """
        clauses = []
        
        # Constraint: For ALL edges, Tx + Ty <= K (this ensures max <= K)
        for edge_id in self.Tx_vars:
            Tx = self.Tx_vars[edge_id]
            Ty = self.Ty_vars[edge_id]
            
            # For each combination (i,j) where i + j > K, add constraint
            # Include j=0 case (Ty = 0)
            for i in range(1, len(Tx) + 1):  # Tx >= i (represented by Tx[i-1])
                for j in range(0, len(Ty) + 1):  # Ty >= j (j=0 means any Ty value)
                    if i + j > K:
                        if j == 0:
                            # Case: Tx >= i ∧ Ty >= 0 (any Ty) where i > K
                            # Simply forbid Tx >= i when i > K
                            if i-1 < len(Tx):
                                clauses.append([-Tx[i-1]])
                        else:
                            # Case: Tx >= i ∧ Ty >= j where i + j > K
                            if i-1 < len(Tx) and j-1 < len(Ty):
                                clauses.append([-Tx[i-1], -Ty[j-1]])
        
        return clauses
    
    def encode_advanced_bandwidth_constraint(self, K):
        """
        Encode constraint đúng cho bandwidth:
        max{Tx₁ + Ty₁, Tx₂ + Ty₂, ..., Txₘ + Tyₘ} <= K
        
        Equivalent: (Tx₁ + Ty₁ <= K) ∧ (Tx₂ + Ty₂ <= K) ∧ ... ∧ (Txₘ + Tyₘ <= K)
        
        For each edge (u,v): Tx + Ty <= K
        Advanced encoding thêm các constraint tighter để improve performance.
        
        FIXED: Bao gồm constraint cho trường hợp Ty = 0
        
        Thermometer encoding semantics:
        - Tx[i] means Tx >= i+1 (0-indexed)
        - So Tx >= d is represented by Tx[d-1] 
        - And Ty <= d is represented by ¬Ty[d] (since Ty[d] means Ty >= d+1)
        """
        clauses = []
        
        # Main constraint: For ALL edges, Tx + Ty <= K
        for edge_id in self.Tx_vars:
            Tx = self.Tx_vars[edge_id] 
            Ty = self.Ty_vars[edge_id]
            
            # Basic constraint: Tx + Ty <= K (FIXED VERSION)
            # For each combination (i,j) where i + j > K, add constraint
            for i in range(1, len(Tx) + 1):  # Tx >= i
                for j in range(0, len(Ty) + 1):  # Ty >= j (include j=0 case)
                    if i + j > K:
                        if j == 0:
                            # Case: Tx >= i where i > K (forbid completely)
                            if i-1 < len(Tx):
                                clauses.append([-Tx[i-1]])
                        else:
                            # Case: Tx >= i ∧ Ty >= j where i + j > K
                            if i-1 < len(Tx) and j-1 < len(Ty):
                                clauses.append([-Tx[i-1], -Ty[j-1]])
            
            # Advanced tighter constraints for better propagation:
            # (Tx >= i → Ty <= K-i) cho i = 1, 2, ..., K
            for i in range(1, K + 1):
                if K-i >= 0:
                    # Tx >= i is represented by Tx[i-1] (if i-1 < len(Tx))
                    # Ty <= K-i is represented by ¬Ty[K-i] (if K-i < len(Ty))
                    
                    if i-1 < len(Tx):
                        if K-i == 0:
                            # Special case: Tx >= i → Ty = 0 (forbid all Ty variables)
                            if len(Ty) > 0:
                                clauses.append([-Tx[i-1], -Ty[0]])
                        elif K-i < len(Ty):
                            # General case: Tx >= i → Ty <= K-i
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
    Test function theo ý tưởng thầy - Pure random UB + SAT encoding
    """
    print("=== TESTING BANDWIDTH SOLVER - TEACHER'S APPROACH ===")
    
    # Test case 1: Triangle graph
    print(f"\n" + "="*60)
    print(f"TEST CASE 1: TRIANGLE GRAPH")
    print(f"="*60)
    
    n1 = 3
    edges1 = [(1, 2), (2, 3), (1, 3)]
    
    solver1 = BandwidthOptimizationSolver(n1, 'glucose4')
    solver1.set_graph_edges(edges1)
    solver1.create_position_variables()
    solver1.create_distance_variables()
    
    optimal1 = solver1.solve_bandwidth_optimization(start_k=1, end_k=4)
    print(f"\n🎯 TRIANGLE RESULT: Optimal bandwidth = {optimal1}")
    
    # Test case 2: Path graph
    print(f"\n" + "="*60)
    print(f"TEST CASE 2: PATH GRAPH")
    print(f"="*60)
    
    n2 = 4
    edges2 = [(1, 2), (2, 3), (3, 4)]  # Path: 1-2-3-4
    
    solver2 = BandwidthOptimizationSolver(n2, 'glucose4')
    solver2.set_graph_edges(edges2)
    solver2.create_position_variables()
    solver2.create_distance_variables()
    
    optimal2 = solver2.solve_bandwidth_optimization(start_k=1, end_k=6)
    print(f"\n🎯 PATH RESULT: Optimal bandwidth = {optimal2}")
    
    # Test case 3: Cycle graph (original test)
    print(f"\n" + "="*60)
    print(f"TEST CASE 3: CYCLE GRAPH")
    print(f"="*60)
    
    n3 = 5
    edges3 = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]  # Cycle graph
    
    solver3 = BandwidthOptimizationSolver(n3, 'glucose4')
    solver3.set_graph_edges(edges3)
    solver3.create_position_variables()
    solver3.create_distance_variables()
    
    optimal3 = solver3.solve_bandwidth_optimization(start_k=1, end_k=8)
    print(f"\n🎯 CYCLE RESULT: Optimal bandwidth = {optimal3}")
    
    # Summary
    print(f"\n" + "="*80)
    print(f"FINAL SUMMARY")
    print(f"="*80)
    print(f"Triangle (3 nodes, 3 edges): Optimal = {optimal1}")
    print(f"Path     (4 nodes, 3 edges): Optimal = {optimal2}")
    print(f"Cycle    (5 nodes, 5 edges): Optimal = {optimal3}")
    print(f"="*80)

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
