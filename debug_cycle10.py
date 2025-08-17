#!/usr/bin/env python3
"""
Debug script to compare validator vs incremental solver results
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mtx_bandwidth_validator_simple import CustomKBandwidthValidator
from incremental_bandwidth_solver import IncrementalBandwidthSolver

def test_cycle10_with_symmetry():
    """Test cycle10 with symmetry breaking to match incremental solver"""
    
    print("="*80)
    print("DEBUGGING CYCLE10 VALIDATOR VS INCREMENTAL SOLVER")
    print("="*80)
    
    # Test with validator (no symmetry breaking)
    print("\n1. VALIDATOR (no symmetry breaking)")
    print("-"*50)
    
    validator = CustomKBandwidthValidator("mtx/regular/cycle10.mtx")
    is_sat_1, result_1 = validator.test_bandwidth_k(1, 'cadical195')
    
    print(f"Validator result K=1: {'SAT' if is_sat_1 else 'UNSAT'}")
    if is_sat_1 and result_1.get('solution_info'):
        solution = result_1['solution_info']
        print(f"Actual bandwidth: {solution.get('actual_bandwidth')}")
        positions = solution.get('positions', {})
        if 1 in positions and 2 in positions:
            print(f"Vertex 1 position: {positions[1]}")
            print(f"Vertex 2 position: {positions[2]}")
    
    # Test with modified validator (add symmetry breaking manually)
    print("\n2. VALIDATOR WITH SYMMETRY BREAKING")
    print("-"*50)
    
    # Create a modified solver with symmetry breaking
    solver = validator._create_modified_solver_with_symmetry(1)
    is_sat_2 = solver.solve()
    
    print(f"Validator with symmetry K=1: {'SAT' if is_sat_2 else 'UNSAT'}")
    
    # Test incremental solver for comparison
    print("\n3. INCREMENTAL SOLVER")
    print("-"*50)
    
    inc_solver = IncrementalBandwidthSolver(10, 'cadical195')
    inc_solver.set_graph_edges([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 1)])
    inc_solver.create_position_variables()
    inc_solver.create_distance_variables()
    
    optimal = inc_solver.solve_bandwidth_optimization(start_k=1, end_k=4)
    print(f"Incremental solver optimal: {optimal}")

if __name__ == "__main__":
    test_cycle10_with_symmetry()
