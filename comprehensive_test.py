#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bandwidth_optimization_solver import BandwidthOptimizationSolver

def comprehensive_test():
    """Test comprehensive để verify SAT solver đã fix"""
    
    print("🔍 === COMPREHENSIVE SAT SOLVER TEST ===")
    print("Testing if SAT solver can now achieve optimal solutions like DRSA")
    print()
    
    test_cases = [
        {
            'name': 'Single Edge',
            'n': 2,
            'edges': [(1, 2)],
            'expected_optimal': 1,
            'description': 'Two nodes connected by one edge'
        },
        {
            'name': 'Path P3',
            'n': 3, 
            'edges': [(1, 2), (2, 3)],
            'expected_optimal': 1,
            'description': 'Path with 3 nodes'
        },
        {
            'name': 'Cycle C4',
            'n': 4,
            'edges': [(1, 2), (2, 3), (3, 4), (4, 1)],
            'expected_optimal': 1,
            'description': 'Cycle with 4 nodes'
        },
        {
            'name': 'Path P4',
            'n': 4,
            'edges': [(1, 2), (2, 3), (3, 4)],
            'expected_optimal': 1,
            'description': 'Path with 4 nodes'
        },
        {
            'name': 'Star S4',
            'n': 4,
            'edges': [(1, 2), (1, 3), (1, 4)],
            'expected_optimal': 1,
            'description': 'Star with center node 1'
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['name']}")
        print(f"   Description: {test_case['description']}")
        print(f"   Nodes: {test_case['n']}, Edges: {test_case['edges']}")
        print(f"   Expected optimal: {test_case['expected_optimal']}")
        
        try:
            # Create solver
            solver = BandwidthOptimizationSolver(test_case['n'], 'glucose3')
            solver.set_graph_edges(test_case['edges'])
            solver.create_position_variables()
            solver.create_distance_variables()
            
            # Test từ K=1 lên để tìm minimum feasible K
            optimal_found = None
            
            for K in range(1, test_case['n'] + 1):
                result = solver.step1_test_upper_bound(K)
                if result:
                    optimal_found = K
                    print(f"   ✅ K={K} is feasible")
                    break
                else:
                    print(f"   ❌ K={K} is not feasible")
            
            if optimal_found is not None:
                is_optimal = (optimal_found == test_case['expected_optimal'])
                status = "✅ OPTIMAL" if is_optimal else f"❌ SUBOPTIMAL (expected {test_case['expected_optimal']})"
                print(f"   🎯 SAT found: K={optimal_found} {status}")
                
                results.append({
                    'name': test_case['name'],
                    'expected': test_case['expected_optimal'],
                    'found': optimal_found,
                    'optimal': is_optimal
                })
            else:
                print(f"   ❌ ERROR: No feasible solution found")
                results.append({
                    'name': test_case['name'],
                    'expected': test_case['expected_optimal'],
                    'found': None,
                    'optimal': False
                })
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({
                'name': test_case['name'],
                'expected': test_case['expected_optimal'],
                'found': None,
                'optimal': False
            })
    
    # Summary
    print(f"\n" + "="*60)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*60)
    
    optimal_count = sum(1 for r in results if r['optimal'])
    total_count = len(results)
    
    print(f"Test Cases: {total_count}")
    print(f"Optimal Solutions: {optimal_count}")
    print(f"Success Rate: {optimal_count}/{total_count} ({100*optimal_count/total_count:.1f}%)")
    print()
    
    print("Detailed Results:")
    for result in results:
        status = "✅" if result['optimal'] else "❌"
        found_str = str(result['found']) if result['found'] is not None else "FAIL"
        print(f"  {status} {result['name']:12} Expected: {result['expected']:2d}, Found: {found_str:4}")
    
    if optimal_count == total_count:
        print("\n🎉 SUCCESS! SAT solver is now competitive with DRSA!")
        print("✅ All test cases achieved optimal solutions")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: {optimal_count}/{total_count} optimal solutions")
        print("❌ Some test cases need further debugging")
    
    return results

if __name__ == "__main__":
    comprehensive_test()
