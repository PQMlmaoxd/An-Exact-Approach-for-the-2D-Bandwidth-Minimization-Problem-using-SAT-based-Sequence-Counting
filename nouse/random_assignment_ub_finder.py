# random_assignment_ub_finder.py
# Find Upper Bound using random assignment - Step 1 in strategy

import random
import time
from itertools import permutations
import numpy as np

class RandomAssignmentUBFinder:
    """
    Find Upper Bound for 2D Bandwidth Minimization using valid random assignment
    
    Strategy:
    1. Generate valid random assignments that satisfy position constraints:
       - Each vertex gets exactly one X position and one Y position
       - Each position gets exactly one vertex (bijection/permutation)
    2. Calculate bandwidth of that valid assignment
    3. Repeat multiple times to find best valid assignment
    4. Return smallest bandwidth found as Upper Bound
    
    Position Constraints Enforced:
    - X assignment: permutation of [1,2,...,n] 
    - Y assignment: permutation of [1,2,...,n]
    - No two vertices share the same (x,y) coordinate
    """
    
    def __init__(self, n, edges, seed=None):
        """
        Initialize UB finder
        
        Args:
            n: Number of graph vertices
            edges: List of edges [(u1,v1), (u2,v2), ...]
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
        Find UB using random search
        
        Args:
            max_iterations: Maximum number of attempts
            time_limit: Time limit (seconds)
            
        Returns:
            dict: Search results with best UB and assignment
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
            
            # Create random assignment
            x_assignment = self._random_assignment()
            y_assignment = self._random_assignment()
            
            # Calculate bandwidth
            bandwidth = self._calculate_bandwidth(x_assignment, y_assignment)
            
            # Update best
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
        """
        Create valid random assignment with position constraints
        
        Ensures each vertex gets exactly one position AND
        each position gets exactly one vertex (bijection)
        
        Returns:
            list: Permutation of positions [1, 2, ..., n]
        """
        # Create valid permutation: each position appears exactly once
        positions = list(range(1, self.n + 1))
        random.shuffle(positions)
        return positions
    
    def _calculate_bandwidth(self, x_assignment, y_assignment):
        """
        Calculate bandwidth of assignment according to standard 2DBMP definition
        
        Args:
            x_assignment: Valid assignment for X axis [pos_of_vertex_1, pos_of_vertex_2, ...]
                         where each position 1..n appears exactly once
            y_assignment: Valid assignment for Y axis (same constraint)
            
        Returns:
            int: Bandwidth (max Manhattan distance across all edges)
        """
        # Validate assignments first
        if not self._validate_assignment(x_assignment):
            raise ValueError("Invalid X assignment: positions not unique")
        if not self._validate_assignment(y_assignment):
            raise ValueError("Invalid Y assignment: positions not unique")
        
        max_bandwidth = 0
        
        for u, v in self.edges:
            # Calculate X and Y distances
            x_dist = abs(x_assignment[u - 1] - x_assignment[v - 1])
            y_dist = abs(y_assignment[u - 1] - y_assignment[v - 1])
            
            # Bandwidth = x_dist + y_dist (Manhattan distance) for 2DBMP
            edge_bandwidth = x_dist + y_dist
            max_bandwidth = max(max_bandwidth, edge_bandwidth)
        
        return max_bandwidth
    
    def _validate_assignment(self, assignment):
        """
        Validate that assignment satisfies position constraints
        
        Args:
            assignment: List of positions for vertices
            
        Returns:
            bool: True if valid (each position 1..n appears exactly once)
        """
        if len(assignment) != self.n:
            return False
        
        # Check that all positions 1..n appear exactly once
        sorted_assignment = sorted(assignment)
        expected = list(range(1, self.n + 1))
        
        return sorted_assignment == expected
    
    def visualize_assignment(self, assignment_result):
        """
        Visualize best assignment
        
        Args:
            assignment_result: Results from find_ub_* functions
        """
        if assignment_result['assignment'] is None:
            print("No assignment found!")
            return
        
        x_assignment, y_assignment = assignment_result['assignment']
        
        print(f"\n=== BEST ASSIGNMENT VISUALIZATION ===")
        print(f"Upper Bound: {assignment_result['ub']}")
        print(f"X-axis assignment: {x_assignment}")
        print(f"Y-axis assignment: {y_assignment}")
        
        # Create grid visualization
        print(f"\nGrid visualization:")
        grid = [[' ' for _ in range(self.n)] for _ in range(self.n)]
        
        for vertex in range(1, self.n + 1):
            x_pos = x_assignment[vertex - 1] - 1  # 0-indexed
            y_pos = y_assignment[vertex - 1] - 1  # 0-indexed
            grid[y_pos][x_pos] = str(vertex)
        
        for row in grid:
            print('|' + '|'.join(f'{cell:^3}' for cell in row) + '|')
        
        # Calculate detailed bandwidth
        print(f"\nEdge bandwidth details:")
        for u, v in self.edges:
            x_dist = abs(x_assignment[u - 1] - x_assignment[v - 1])
            y_dist = abs(y_assignment[u - 1] - y_assignment[v - 1])
            bandwidth = x_dist + y_dist  # Manhattan distance
            print(f"Edge ({u},{v}): X-dist={x_dist}, Y-dist={y_dist}, Bandwidth={bandwidth}")

def test_random_ub_finder():
    """Test function for RandomAssignmentUBFinder"""
    print("=== TESTING RANDOM UB FINDER ===")
    
    # Test graph: cycle of 4 vertices
    n = 4
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    
    finder = RandomAssignmentUBFinder(n, edges, seed=42)
    
    # Test position constraint validation
    print("\nTesting Position Constraints Validation:")
    valid_assignment = [1, 3, 2, 4]
    invalid_assignment1 = [1, 1, 2, 3]  # Position 1 appears twice
    invalid_assignment2 = [1, 2, 3]     # Wrong length
    invalid_assignment3 = [0, 2, 3, 4]  # Position 0 invalid
    
    print(f"Valid assignment {valid_assignment}: {finder._validate_assignment(valid_assignment)}")
    print(f"Invalid assignment {invalid_assignment1}: {finder._validate_assignment(invalid_assignment1)}")
    print(f"Invalid assignment {invalid_assignment2}: {finder._validate_assignment(invalid_assignment2)}")
    print(f"Invalid assignment {invalid_assignment3}: {finder._validate_assignment(invalid_assignment3)}")
    
    # Test random assignment generation
    print(f"\nTesting Random Assignment Generation:")
    for i in range(5):
        x_assign = finder._random_assignment()
        y_assign = finder._random_assignment()
        print(f"Trial {i+1}: X={x_assign}, Y={y_assign}")
        print(f"  X valid: {finder._validate_assignment(x_assign)}")
        print(f"  Y valid: {finder._validate_assignment(y_assign)}")
        
        # Calculate bandwidth for this valid assignment
        bandwidth = finder._calculate_bandwidth(x_assign, y_assign)
        print(f"  Bandwidth: {bandwidth}")
    
    # Test random search method
    print("\nTesting Random Search:")
    result = finder.find_ub_random_search(max_iterations=1000, time_limit=15)
    finder.visualize_assignment(result)
    
    print(f"\n=== RESULT SUMMARY ===")
    print(f"Random Search UB: {result['ub']}")
    print(f"Total iterations: {result['iterations']}")
    print(f"Total time: {result['time']:.2f}s")
    
    # Verify final assignment is valid
    if result['assignment'] is not None:
        x_final, y_final = result['assignment']
        print(f"Final assignment validation:")
        print(f"  X assignment valid: {finder._validate_assignment(x_final)}")
        print(f"  Y assignment valid: {finder._validate_assignment(y_final)}")
    else:
        print("No assignment found!")

if __name__ == '__main__':
    test_random_ub_finder()
