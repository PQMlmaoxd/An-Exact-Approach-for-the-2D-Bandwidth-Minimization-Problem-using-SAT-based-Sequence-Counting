# random_assignment_ub_finder.py
# Tìm Upper Bound bằng phép gán ngẫu nhiên - Bước 1 trong strategy

import random
import time
from itertools import permutations
import numpy as np

class RandomAssignmentUBFinder:
    """
    Tìm Upper Bound cho 2D Bandwidth Minimization bằng phép gán ngẫu nhiên
    
    Strategy:
    1. Gán ngẫu nhiên vị trí X, Y cho các đỉnh
    2. Tính bandwidth của assignment đó
    3. Lặp lại nhiều lần để tìm best assignment
    4. Trả về bandwidth nhỏ nhất tìm được làm Upper Bound
    """
    
    def __init__(self, n, edges, seed=None):
        """
        Khởi tạo UB finder
        
        Args:
            n: Số đỉnh của đồ thị
            edges: Danh sách cạnh [(u1,v1), (u2,v2), ...]
            seed: Random seed for reproducibility
        """
        self.n = n
        self.edges = edges
        self.best_ub = float('inf')
        self.best_assignment = None
        self.search_history = []
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"Initialized UB Finder for graph with n={n}, edges={len(edges)}")
    
    def find_ub_random_search(self, max_iterations=1000, time_limit=30):
        """
        Tìm UB bằng random search
        
        Args:
            max_iterations: Số lần thử tối đa
            time_limit: Thời gian giới hạn (seconds)
            
        Returns:
            dict: Kết quả search với UB và assignment tốt nhất
        """
        print(f"\n=== RANDOM SEARCH FOR UPPER BOUND ===")
        print(f"Max iterations: {max_iterations}, Time limit: {time_limit}s")
        
        start_time = time.time()
        best_ub = float('inf')
        best_assignment = None
        
        for iteration in range(max_iterations):
            # Check time limit
            if time.time() - start_time > time_limit:
                print(f"Time limit reached after {iteration} iterations")
                break
            
            # Tạo assignment ngẫu nhiên
            x_assignment = self._random_assignment()
            y_assignment = self._random_assignment()
            
            # Tính bandwidth
            bandwidth = self._calculate_bandwidth(x_assignment, y_assignment)
            
            # Cập nhật best
            if bandwidth < best_ub:
                best_ub = bandwidth
                best_assignment = (x_assignment.copy(), y_assignment.copy())
                print(f"Iteration {iteration:4d}: New best UB = {best_ub}")
            
            # Log progress
            if iteration % 100 == 0 and iteration > 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iteration:4d}: Current best = {best_ub}, Elapsed = {elapsed:.2f}s")
        
        elapsed = time.time() - start_time
        print(f"\nRandom search completed:")
        print(f"Best UB found: {best_ub}")
        print(f"Total iterations: {iteration + 1}")
        print(f"Total time: {elapsed:.2f}s")
        
        return {
            'ub': best_ub,
            'assignment': best_assignment,
            'iterations': iteration + 1,
            'time': elapsed
        }
    

    
    def _random_assignment(self):
        """Tạo assignment hoàn toàn ngẫu nhiên"""
        assignment = list(range(1, self.n + 1))
        random.shuffle(assignment)
        return assignment
    
    def _calculate_bandwidth(self, x_assignment, y_assignment):
        """
        Tính bandwidth của assignment theo định nghĩa chuẩn 2DBMP
        
        Args:
            x_assignment: Assignment cho trục X [pos_of_vertex_1, pos_of_vertex_2, ...]
            y_assignment: Assignment cho trục Y
            
        Returns:
            int: Bandwidth (max Manhattan distance across all edges)
        """
        max_bandwidth = 0
        
        for u, v in self.edges:
            # Tính khoảng cách X và Y
            x_dist = abs(x_assignment[u - 1] - x_assignment[v - 1])
            y_dist = abs(y_assignment[u - 1] - y_assignment[v - 1])
            
            # Bandwidth = x_dist + y_dist (Manhattan distance) cho 2DBMP
            edge_bandwidth = x_dist + y_dist
            max_bandwidth = max(max_bandwidth, edge_bandwidth)
        
        return max_bandwidth
    
    def visualize_assignment(self, assignment_result):
        """
        Visualize assignment tốt nhất
        
        Args:
            assignment_result: Kết quả từ các hàm find_ub_*
        """
        if assignment_result['assignment'] is None:
            print("No assignment found!")
            return
        
        x_assignment, y_assignment = assignment_result['assignment']
        
        print(f"\n=== BEST ASSIGNMENT VISUALIZATION ===")
        print(f"Upper Bound: {assignment_result['ub']}")
        print(f"X-axis assignment: {x_assignment}")
        print(f"Y-axis assignment: {y_assignment}")
        
        # Tạo grid visualization
        print(f"\nGrid visualization:")
        grid = [[' ' for _ in range(self.n)] for _ in range(self.n)]
        
        for vertex in range(1, self.n + 1):
            x_pos = x_assignment[vertex - 1] - 1  # 0-indexed
            y_pos = y_assignment[vertex - 1] - 1  # 0-indexed
            grid[y_pos][x_pos] = str(vertex)
        
        for row in grid:
            print('|' + '|'.join(f'{cell:^3}' for cell in row) + '|')
        
        # Tính bandwidth chi tiết
        print(f"\nEdge bandwidth details:")
        for u, v in self.edges:
            x_dist = abs(x_assignment[u - 1] - x_assignment[v - 1])
            y_dist = abs(y_assignment[u - 1] - y_assignment[v - 1])
            bandwidth = x_dist + y_dist  # Manhattan distance
            print(f"Edge ({u},{v}): X-dist={x_dist}, Y-dist={y_dist}, Bandwidth={bandwidth}")

def test_random_ub_finder():
    """Test function cho RandomAssignmentUBFinder"""
    print("=== TESTING RANDOM UB FINDER ===")
    
    # Test graph: cycle of 4 vertices
    n = 4
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    
    finder = RandomAssignmentUBFinder(n, edges, seed=42)
    
    # Test random search method
    print("\nTesting Random Search:")
    result = finder.find_ub_random_search(max_iterations=1000, time_limit=15)
    finder.visualize_assignment(result)
    
    print(f"\n=== RESULT SUMMARY ===")
    print(f"Random Search UB: {result['ub']}")
    print(f"Total iterations: {result['iterations']}")
    print(f"Total time: {result['time']:.2f}s")

if __name__ == '__main__':
    test_random_ub_finder()
