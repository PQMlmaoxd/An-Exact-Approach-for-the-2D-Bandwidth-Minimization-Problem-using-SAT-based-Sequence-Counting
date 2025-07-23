#!/usr/bin/env python3
"""
Complete DRSA Implementation for 2D Bandwidth Minimization Problem
Following the exact algorithm structure from the research paper
"""

import sys
import os
import time
import random
import numpy as np
import math
from copy import deepcopy

class TwoDBMP_DRSA:
    """
    Implementation đầy đủ của DRSA cho bài toán 2D Bandwidth Minimization
    Theo đúng cấu trúc thuật toán từ research paper
    """
    
    def __init__(self, graph_vertices, graph_edges, grid_size=None, seed=42):
        """
        Khởi tạo DRSA solver
        
        Args:
            graph_vertices: Số lượng vertices (n)
            graph_edges: Danh sách edges [(u, v), ...]
            grid_size: Kích thước grid (width, height). Nếu None, sử dụng n x n
            seed: Random seed
        """
        self.n = graph_vertices
        self.edges = graph_edges
        self.m = len(graph_edges)  # Số lượng edges
        
        # Thiết lập Grid
        if grid_size is None:
            # Mặc định: sử dụng grid vuông nhỏ nhất có thể chứa n vertices
            grid_dim = math.ceil(math.sqrt(self.n))
            self.grid_size = (grid_dim, grid_dim)
        else:
            self.grid_size = grid_size
        
        # Tham số DRSA
        self.T0 = 100.0           # Nhiệt độ ban đầu
        self.alpha = 0.95         # Tốc độ làm lạnh
        self.L = 50               # Độ dài Markov chain
        self.T_final = 0.001      # Nhiệt độ cuối
        
        # Xác suất các toán tử neighbor
        self.p_rex = 0.4  # Random Exchange
        self.p_nex = 0.4  # Neighbor Exchange  
        self.p_rot = 0.2  # Rotation
        
        # Random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Thống kê
        self.stats = {
            'iterations': 0,
            'temperature_steps': 0,
            'accepted_moves': 0,
            'rejected_moves': 0
        }
    
    def generate_initial_solution(self):
        """
        Tạo solution ngẫu nhiên ban đầu
        Returns: φ (labeling) - ánh xạ từ vertex đến tọa độ (x, y)
        """
        # 1. Tạo tất cả tọa độ có thể trên grid
        all_coords = []
        for x in range(1, self.grid_size[0] + 1):
            for y in range(1, self.grid_size[1] + 1):
                all_coords.append((x, y))
        
        # 2. Xáo trộn ngẫu nhiên tọa độ
        random.shuffle(all_coords)
        
        # 3. Gán mỗi vertex cho một tọa độ duy nhất
        phi = {}
        for v in range(1, self.n + 1):
            phi[v] = all_coords[v - 1]
        
        return phi
    
    def evaluate(self, phi):
        """
        Hàm đánh giá nâng cao theo công thức Γ = β + γ/N trong paper
        
        Args:
            phi: Labeling (ánh xạ từ vertex đến coordinate)
            
        Returns:
            Gamma: Điểm đánh giá tổng hợp
        """
        # 1. Tính tất cả khoảng cách L1 (Manhattan) cho mỗi edge
        distances = []
        for u, v in self.edges:
            coord_u = phi[u]
            coord_v = phi[v]
            l1_dist = abs(coord_u[0] - coord_v[0]) + abs(coord_u[1] - coord_v[1])
            distances.append(l1_dist)
        
        # 2. Tính β: 2D bandwidth (khoảng cách tối đa)
        beta = max(distances) if distances else 0
        
        # 3. Xây dựng counting vector C
        # C[d] = số lượng edges có khoảng cách L1 = d
        max_dist = self.grid_size[0] + self.grid_size[1] - 2
        C = [0] * (max_dist + 1)
        for dist in distances:
            if dist <= max_dist:
                C[dist] += 1
        
        # 4. Tính discriminating component γ (gamma)
        # γ giúp phân biệt các solution có cùng β
        # Công thức: Σ C[d] * (m+1)^(d-1) for d=1 to max_dist
        gamma = 0
        for d in range(1, max_dist + 1):
            if C[d] > 0:
                gamma += C[d] * ((self.m + 1) ** (d - 1))
        
        # 5. Tính evaluation cuối cùng Γ (Gamma)
        # N = (m+1)^β đảm bảo γ là phần thập phân
        if beta > 0:
            N = (self.m + 1) ** beta
            Gamma = beta + gamma / N
        else:
            Gamma = gamma
        
        return Gamma
    
    def generate_neighbor(self, phi):
        """
        Tạo neighbor solution sử dụng các toán tử DRSA: REX, NEX, ROT
        
        Args:
            phi: Labeling hiện tại
            
        Returns:
            phi_new: Labeling neighbor mới
        """
        phi_new = deepcopy(phi)
        
        # Chọn toán tử dựa trên xác suất
        rand = random.random()
        if rand < self.p_rex:
            op = 'REX'
        elif rand < self.p_rex + self.p_nex:
            op = 'NEX'
        else:
            op = 'ROT'
        
        vertices = list(range(1, self.n + 1))
        
        if op == 'REX':
            # Random Exchange: Hoán đổi vị trí của hai vertex ngẫu nhiên
            v1, v2 = random.sample(vertices, 2)
            phi_new[v1], phi_new[v2] = phi_new[v2], phi_new[v1]
            
        elif op == 'NEX':
            # Neighbor Exchange: Hoán đổi vị trí của hai vertex kề nhau
            if self.edges:
                # Chọn edge ngẫu nhiên và hoán đổi hai endpoint
                u, v = random.choice(self.edges)
                phi_new[u], phi_new[v] = phi_new[v], phi_new[u]
            else:
                # Fallback về REX nếu không có edges
                v1, v2 = random.sample(vertices, 2)
                phi_new[v1], phi_new[v2] = phi_new[v2], phi_new[v1]
                
        elif op == 'ROT':
            # Rotation: Xoay vị trí của ba vertex ngẫu nhiên
            # v1 -> v2, v2 -> v3, v3 -> v1
            if self.n >= 3:
                v1, v2, v3 = random.sample(vertices, 3)
                temp_pos = phi_new[v1]
                phi_new[v1] = phi_new[v2]
                phi_new[v2] = phi_new[v3]
                phi_new[v3] = temp_pos
            else:
                # Fallback về REX nếu quá ít vertices
                v1, v2 = random.sample(vertices, 2)
                phi_new[v1], phi_new[v2] = phi_new[v2], phi_new[v1]
        
        return phi_new
    
    def TwoDBMP_DRSA(self, verbose=False):
        """
        Thuật toán DRSA chính theo cấu trúc trong paper
        
        Args:
            verbose: In thông tin tiến trình
            
        Returns:
            phi_best: Labeling tốt nhất tìm được
            cost_best: Cost tốt nhất (giá trị Gamma)
            stats: Thống kê thuật toán
        """
        start_time = time.time()
        
        # Khởi tạo
        phi_current = self.generate_initial_solution()
        cost_current = self.evaluate(phi_current)
        phi_best = deepcopy(phi_current)
        cost_best = cost_current
        T = self.T0
        
        if verbose:
            print(f"DRSA Algorithm Started")
            print(f"   Initial cost: {cost_current:.6f}")
            print(f"   Initial temperature: {T}")
            print(f"   Cooling rate: {self.alpha}")
            print(f"   Markov chain length: {self.L}")
            print()
        
        # Main Simulated Annealing Loop
        while T > self.T_final:
            if verbose:
                print(f"Temperature: {T:.6f} | Best cost: {cost_best:.6f}")
            
            # Markov Chain tại nhiệt độ hiện tại
            for i in range(self.L):
                # Tạo neighbor solution
                phi_candidate = self.generate_neighbor(phi_current)
                
                # Đánh giá solution mới
                cost_candidate = self.evaluate(phi_candidate)
                
                # Quyết định chấp nhận
                delta_cost = cost_candidate - cost_current
                
                if delta_cost < 0:
                    # Chấp nhận solution tốt hơn
                    phi_current = phi_candidate
                    cost_current = cost_candidate
                    self.stats['accepted_moves'] += 1
                else:
                    # Chấp nhận solution xấu hơn với xác suất
                    acceptance_probability = math.exp(-delta_cost / T)
                    if random.random() < acceptance_probability:
                        phi_current = phi_candidate
                        cost_current = cost_candidate
                        self.stats['accepted_moves'] += 1
                    else:
                        self.stats['rejected_moves'] += 1
                
                # Update best solution
                if cost_current < cost_best:
                    phi_best = deepcopy(phi_current)
                    cost_best = cost_current
                    if verbose:
                        print(f"   New best: {cost_best:.6f}")
                
                self.stats['iterations'] += 1
            
            # Cooling
            T = T * self.alpha
            self.stats['temperature_steps'] += 1
        
        # Calculate final statistics
        total_time = time.time() - start_time
        self.stats['total_time'] = total_time
        self.stats['final_temperature'] = T
        
        if verbose:
            print(f"\nDRSA Algorithm Completed")
            print(f"   Best cost: {cost_best:.6f}")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Iterations: {self.stats['iterations']}")
            print(f"   Accepted moves: {self.stats['accepted_moves']}")
            print(f"   Rejected moves: {self.stats['rejected_moves']}")
            print(f"   Acceptance rate: {self.stats['accepted_moves']/(self.stats['accepted_moves']+self.stats['rejected_moves'])*100:.1f}%")
        
        return phi_best, cost_best, self.stats
    
    def extract_bandwidth(self, phi):
        """
        Trích xuất bandwidth đơn giản (β) từ labeling
        
        Args:
            phi: Labeling
            
        Returns:
            beta: Khoảng cách Manhattan tối đa
        """
        if not self.edges:
            return 0
        
        max_dist = 0
        for u, v in self.edges:
            coord_u = phi[u]
            coord_v = phi[v]
            dist = abs(coord_u[0] - coord_v[0]) + abs(coord_u[1] - coord_v[1])
            max_dist = max(max_dist, dist)
        
        return max_dist
    
    def print_solution(self, phi):
        """
        In ra solution dưới dạng dễ đọc
        
        Args:
            phi: Labeling để in
        """
        print("Solution Layout:")
        
        # Tạo grid visualization
        grid = {}
        for v in range(1, self.n + 1):
            x, y = phi[v]
            grid[(x, y)] = v
        
        print("   ", end="")
        for x in range(1, self.grid_size[0] + 1):
            print(f"{x:3}", end="")
        print()
        
        for y in range(1, self.grid_size[1] + 1):
            print(f"{y:2}:", end="")
            for x in range(1, self.grid_size[0] + 1):
                if (x, y) in grid:
                    print(f"{grid[(x, y)]:3}", end="")
                else:
                    print("  .", end="")
            print()
        
        # In bandwidth
        bandwidth = self.extract_bandwidth(phi)
        gamma = self.evaluate(phi)
        print(f"\nBandwidth (β): {bandwidth}")
        print(f"Gamma (Γ): {gamma:.6f}")

# Test function
def test_drsa_paper_implementation():
    """
    Kiểm tra implementation DRSA với các ví dụ đơn giản
    """
    print("Testing Complete DRSA Implementation")
    print("="*50)
    
    test_cases = [
        {
            'name': 'Triangle',
            'n': 3,
            'edges': [(1, 2), (2, 3), (3, 1)],
            'description': 'Complete triangle'
        },
        {
            'name': 'Path P4',
            'n': 4,
            'edges': [(1, 2), (2, 3), (3, 4)],
            'description': 'Linear path with 4 nodes'
        },
        {
            'name': 'Star S4',
            'n': 4,
            'edges': [(1, 2), (1, 3), (1, 4)],
            'description': 'Star with center node 1'
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print(f"   {test_case['description']}")
        print(f"   Vertices: {test_case['n']}, Edges: {len(test_case['edges'])}")
        
        # Tạo DRSA solver
        drsa = TwoDBMP_DRSA(
            graph_vertices=test_case['n'],
            graph_edges=test_case['edges'],
            seed=42
        )
        
        # Chạy thuật toán
        phi_best, cost_best, stats = drsa.TwoDBMP_DRSA(verbose=False)
        
        # Trích xuất bandwidth
        bandwidth = drsa.extract_bandwidth(phi_best)
        
        print(f"   Results:")
        print(f"      Bandwidth: {bandwidth}")
        print(f"      Gamma: {cost_best:.6f}")
        print(f"      Time: {stats['total_time']:.3f}s")
        print(f"      Iterations: {stats['iterations']}")
        
        # In solution layout cho các trường hợp nhỏ
        if test_case['n'] <= 4:
            drsa.print_solution(phi_best)

if __name__ == "__main__":
    test_drsa_paper_implementation()
