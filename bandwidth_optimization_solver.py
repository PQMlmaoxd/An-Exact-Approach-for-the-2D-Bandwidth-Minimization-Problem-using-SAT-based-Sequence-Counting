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
    from nsc_encoder import encode_nsc_exactly_k, encode_nsc_at_most_k
    print("✅ Successfully imported essential modules including NSC encoder")
except ImportError as e:
    print(f"❌ Critical import error: {e}")
    print("NSC encoder is required for O(n²) complexity optimization")
    raise ImportError("Missing required NSC encoder - cannot proceed without it")

try:
    from kissat_solver import KissatSolver
except ImportError:
    class KissatSolver:
        def __init__(self, *args, **kwargs):
            pass

# Fallback implementations if imports fail
if 'encode_abs_distance_final' not in locals():
    def encode_abs_distance_final(U_vars, V_vars, n, vpool, prefix="T"):
        T_vars = [vpool.id(f'{prefix}_geq_{d}') for d in range(1, n)]
        clauses = []
        return T_vars, clauses

if 'RandomAssignmentUBFinder' not in locals():
    class RandomAssignmentUBFinder:
        def __init__(self, n, edges, seed=None):
            self.n = n
            self.edges = edges
        
        def find_ub_random_search(self, max_iterations=1000, time_limit=30):
            return {'ub': 2 * (self.n - 1), 'assignment': None, 'iterations': 0, 'time': 0}

class BandwidthOptimizationSolver:
    def __init__(self, n, solver_type='glucose4'):
        """
        Khởi tạo solver cho bài toán 2D Bandwidth Minimization
        
        Args:
            n: Kích thước bài toán (số đỉnh)
            solver_type: Loại SAT solver ('glucose4', 'glucose41', 'glucose3', 'kissat')
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
        Mã hóa ràng buộc: mỗi đỉnh có đúng một vị trí trên mỗi trục - NSC ONLY VERSION
        Độ phức tạp: O(n²)
        """
        clauses = []
        
        for v in range(1, self.n + 1):
            # NSC Exactly-1 encoding - O(n) clauses per constraint
            # Exactly-One cho X using NSC
            nsc_x_clauses = encode_nsc_exactly_k(self.X_vars[v], 1, self.vpool)
            clauses.extend(nsc_x_clauses)
            
            # Exactly-One cho Y using NSC  
            nsc_y_clauses = encode_nsc_exactly_k(self.Y_vars[v], 1, self.vpool)
            clauses.extend(nsc_y_clauses)
        
        # Mỗi position (X,Y) có tối đa một đỉnh - NSC OPTIMIZED VERSION O(n²)
        # NSC-based position uniqueness: At most 1 node per position
        for x in range(self.n):
            for y in range(self.n):
                # Tạo indicator variables: node_at_pos[v] = (X_v_x ∧ Y_v_y)
                node_indicators = []
                for v in range(1, self.n + 1):
                    indicator = self.vpool.id(f'node_{v}_at_{x}_{y}')
                    node_indicators.append(indicator)
                    
                    # indicator ↔ (X_v_x ∧ Y_v_y)
                    clauses.append([-indicator, self.X_vars[v][x]])
                    clauses.append([-indicator, self.Y_vars[v][y]])
                    clauses.append([indicator, -self.X_vars[v][x], -self.Y_vars[v][y]])
                
                # NSC: At most 1 node at position (x,y)  
                # Use unified nsc_encoder.py implementation for consistency
                nsc_at_most_1 = encode_nsc_at_most_k(node_indicators, 1, self.vpool)
                clauses.extend(nsc_at_most_1)
        
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
        print(f"Strategy: Full SAT encoding with Thermometer constraints + NSC")
        print(f"Encoding: (Tx≤{K}) ∧ (Ty≤{K}) ∧ implication constraints")
        print(f"Position constraint encoding: NSC Sequential Counter O(n²)")
        
        # Tạo solver mới
        if self.solver_type == 'glucose4':
            solver = Glucose4()
        elif self.solver_type == 'glucose41':
            solver = Glucose4()
        elif self.solver_type == 'kissat':
            solver = KissatSolver()
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
        
        # Phase 2: SAT encoding - thử từ K = UB-1 xuống đến 1
        print(f"\n🔧 PHASE 2: SAT encoding optimization from K={feasible_ub-1} down to 1")
        
        optimal_k = feasible_ub  # Mặc định là UB
        
        # Thử từ UB-1 xuống đến 1 cho đến khi UNSAT
        for K in range(feasible_ub - 1, 0, -1):
            print(f"\n--- SAT Testing K = {K} ---")
            
            if self.step2_encode_advanced_constraints(K):
                optimal_k = K
                print(f"✅ K = {K} is feasible with SAT - continuing to test smaller K")
            else:
                print(f"❌ K = {K} is UNSAT - stopping search")
                print(f"🎯 OPTIMAL BANDWIDTH = {optimal_k}")
                return optimal_k
        
        # Nếu đến K=1 vẫn SAT thì optimal = 1
        print(f"🎯 OPTIMAL BANDWIDTH = {optimal_k} (tested down to K=1)")
        return optimal_k

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

if __name__ == '__main__':
    # Test main solver
    test_bandwidth_solver()
