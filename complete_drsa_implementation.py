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
    Complete implementation of DRSA for 2D Bandwidth Minimization Problem
    Following the exact algorithm structure from research paper
    """
    
    def __init__(self, graph_vertices, graph_edges, grid_size=None, seed=42):
        """
        Initialize DRSA solver
        
        Args:
            graph_vertices: Number of vertices (n)
            graph_edges: List of edges [(u, v), ...]
            grid_size: Grid dimensions (width, height). If None, uses n x n
            seed: Random seed
        """
        self.n = graph_vertices
        self.edges = graph_edges
        self.m = len(graph_edges)  # Number of edges
        
        # Grid setup
        if grid_size is None:
            # Default: use minimum square grid that can fit n vertices
            grid_dim = math.ceil(math.sqrt(self.n))
            self.grid_size = (grid_dim, grid_dim)
        else:
            self.grid_size = grid_size
        
        # DRSA Parameters
        self.T0 = 100.0           # Initial temperature
        self.alpha = 0.95         # Cooling rate
        self.L = 50               # Markov chain length
        self.T_final = 0.001      # Final temperature
        
        # Neighbor operator probabilities
        self.p_rex = 0.4  # Random Exchange
        self.p_nex = 0.4  # Neighbor Exchange  
        self.p_rot = 0.2  # Rotation
        
        # Random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Statistics
        self.stats = {
            'iterations': 0,
            'temperature_steps': 0,
            'accepted_moves': 0,
            'rejected_moves': 0
        }
    
    def generate_initial_solution(self):
        """
        Generate initial random solution
        Returns: φ (labeling) - mapping from vertex to (x, y) coordinate
        """
        # 1. Create all possible coordinates on grid
        all_coords = []
        for x in range(1, self.grid_size[0] + 1):
            for y in range(1, self.grid_size[1] + 1):
                all_coords.append((x, y))
        
        # 2. Randomly shuffle coordinates
        random.shuffle(all_coords)
        
        # 3. Assign each vertex to a unique coordinate
        phi = {}
        for v in range(1, self.n + 1):
            phi[v] = all_coords[v - 1]
        
        return phi
    
    def evaluate(self, phi):
        """
        Advanced evaluation function following paper's Γ = β + γ/N formula
        
        Args:
            phi: Labeling (vertex -> coordinate mapping)
            
        Returns:
            Gamma: Combined evaluation score
        """
        # 1. Calculate all L1 (Manhattan) distances for each edge
        distances = []
        for u, v in self.edges:
            coord_u = phi[u]
            coord_v = phi[v]
            l1_dist = abs(coord_u[0] - coord_v[0]) + abs(coord_u[1] - coord_v[1])
            distances.append(l1_dist)
        
        # 2. Calculate β: 2D bandwidth (maximum distance)
        beta = max(distances) if distances else 0
        
        # 3. Build counting vector C
        # C[d] = number of edges with L1 distance = d
        max_dist = self.grid_size[0] + self.grid_size[1] - 2
        C = [0] * (max_dist + 1)
        for dist in distances:
            if dist <= max_dist:
                C[dist] += 1
        
        # 4. Calculate discriminating component γ (gamma)
        # γ helps differentiate solutions with same β
        # Formula: Σ C[d] * (m+1)^(d-1) for d=1 to max_dist
        gamma = 0
        for d in range(1, max_dist + 1):
            if C[d] > 0:
                gamma += C[d] * ((self.m + 1) ** (d - 1))
        
        # 5. Calculate final evaluation Γ (Gamma)
        # N = (m+1)^β ensures γ is fractional part
        if beta > 0:
            N = (self.m + 1) ** beta
            Gamma = beta + gamma / N
        else:
            Gamma = gamma
        
        return Gamma
    
    def generate_neighbor(self, phi):
        """
        Generate neighbor solution using DRSA operators: REX, NEX, ROT
        
        Args:
            phi: Current labeling
            
        Returns:
            phi_new: New neighboring labeling
        """
        phi_new = deepcopy(phi)
        
        # Choose operator based on probabilities
        rand = random.random()
        if rand < self.p_rex:
            op = 'REX'
        elif rand < self.p_rex + self.p_nex:
            op = 'NEX'
        else:
            op = 'ROT'
        
        vertices = list(range(1, self.n + 1))
        
        if op == 'REX':
            # Random Exchange: Swap positions of two random vertices
            v1, v2 = random.sample(vertices, 2)
            phi_new[v1], phi_new[v2] = phi_new[v2], phi_new[v1]
            
        elif op == 'NEX':
            # Neighbor Exchange: Swap positions of two adjacent vertices
            if self.edges:
                # Choose random edge and swap its endpoints
                u, v = random.choice(self.edges)
                phi_new[u], phi_new[v] = phi_new[v], phi_new[u]
            else:
                # Fallback to REX if no edges
                v1, v2 = random.sample(vertices, 2)
                phi_new[v1], phi_new[v2] = phi_new[v2], phi_new[v1]
                
        elif op == 'ROT':
            # Rotation: Cycle positions of three random vertices
            # v1 -> v2, v2 -> v3, v3 -> v1
            if self.n >= 3:
                v1, v2, v3 = random.sample(vertices, 3)
                temp_pos = phi_new[v1]
                phi_new[v1] = phi_new[v2]
                phi_new[v2] = phi_new[v3]
                phi_new[v3] = temp_pos
            else:
                # Fallback to REX if too few vertices
                v1, v2 = random.sample(vertices, 2)
                phi_new[v1], phi_new[v2] = phi_new[v2], phi_new[v1]
        
        return phi_new
    
    def TwoDBMP_DRSA(self, verbose=False):
        """
        Main DRSA algorithm following paper's structure
        
        Args:
            verbose: Print progress information
            
        Returns:
            phi_best: Best labeling found
            cost_best: Best cost (Gamma value)
            stats: Algorithm statistics
        """
        start_time = time.time()
        
        # Step 1: Initialization
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
        
        # Step 2: Main Simulated Annealing Loop
        while T > self.T_final:
            if verbose:
                print(f"Temperature: {T:.6f} | Best cost: {cost_best:.6f}")
            
            # Markov Chain at current temperature
            for i in range(self.L):
                # 2.1. Generate neighbor solution
                phi_candidate = self.generate_neighbor(phi_current)
                
                # 2.2. Evaluate new solution
                cost_candidate = self.evaluate(phi_candidate)
                
                # 2.3. Acceptance decision
                delta_cost = cost_candidate - cost_current
                
                if delta_cost < 0:
                    # Accept better solution
                    phi_current = phi_candidate
                    cost_current = cost_candidate
                    self.stats['accepted_moves'] += 1
                else:
                    # Accept worse solution with probability
                    acceptance_probability = math.exp(-delta_cost / T)
                    if random.random() < acceptance_probability:
                        phi_current = phi_candidate
                        cost_current = cost_candidate
                        self.stats['accepted_moves'] += 1
                    else:
                        self.stats['rejected_moves'] += 1
                
                # 2.4. Update best solution
                if cost_current < cost_best:
                    phi_best = deepcopy(phi_current)
                    cost_best = cost_current
                    if verbose:
                        print(f"   New best: {cost_best:.6f}")
                
                self.stats['iterations'] += 1
            
            # Step 3: Cooling
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
        Extract simple bandwidth (β) from labeling
        
        Args:
            phi: Labeling
            
        Returns:
            beta: Maximum Manhattan distance
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
        Print solution in readable format
        
        Args:
            phi: Labeling to print
        """
        print("Solution Layout:")
        
        # Create grid visualization
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
        
        # Print bandwidth
        bandwidth = self.extract_bandwidth(phi)
        gamma = self.evaluate(phi)
        print(f"\nBandwidth (β): {bandwidth}")
        print(f"Gamma (Γ): {gamma:.6f}")

# Test function
def test_drsa_paper_implementation():
    """
    Test DRSA implementation with simple examples
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
        
        # Create DRSA solver
        drsa = TwoDBMP_DRSA(
            graph_vertices=test_case['n'],
            graph_edges=test_case['edges'],
            seed=42
        )
        
        # Run algorithm
        phi_best, cost_best, stats = drsa.TwoDBMP_DRSA(verbose=False)
        
        # Extract bandwidth
        bandwidth = drsa.extract_bandwidth(phi_best)
        
        print(f"   Results:")
        print(f"      Bandwidth: {bandwidth}")
        print(f"      Gamma: {cost_best:.6f}")
        print(f"      Time: {stats['total_time']:.3f}s")
        print(f"      Iterations: {stats['iterations']}")
        
        # Print solution layout for small cases
        if test_case['n'] <= 4:
            drsa.print_solution(phi_best)

if __name__ == "__main__":
    test_drsa_paper_implementation()
