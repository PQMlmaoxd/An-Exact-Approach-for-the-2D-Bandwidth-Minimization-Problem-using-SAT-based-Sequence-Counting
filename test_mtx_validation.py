# test_mtx_validation.py
# Test SAT method với các sample MTX datasets để validate correctness

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bandwidth_optimization_solver import BandwidthOptimizationSolver
from mtx_parser import parse_mtx_file

def test_sample_mtx_files():
    """
    Test SAT solver với tất cả sample MTX files để validate correctness
    """
    print("="*80)
    print("MTX VALIDATION TEST - SAT METHOD VALIDATION")
    print("="*80)
    
    # Danh sách sample files để test
    sample_files = [
        ("sample_mtx_datasets/path_p6.mtx", "Path Graph P6"),
        ("sample_mtx_datasets/cycle_c5.mtx", "Cycle Graph C5"), 
        ("sample_mtx_datasets/complete_k4.mtx", "Complete Graph K4"),
        ("sample_mtx_datasets/star_s6.mtx", "Star Graph S6"),
        ("sample_mtx_datasets/random_sparse.mtx", "Random Sparse Graph"),
        ("mtx/1138_bus.mtx", "1138 Bus Graph (Large)")
    ]
    
    results = []
    
    for mtx_file, description in sample_files:
        print(f"\n" + "="*60)
        print(f"TESTING: {description}")
        print(f"File: {mtx_file}")
        print(f"="*60)
        
        try:
            # Parse MTX file
            print(f"📁 Parsing MTX file...")
            start_time = time.time()
            n, edges = parse_mtx_file(mtx_file)
            parse_time = time.time() - start_time
            
            print(f"Graph info: n={n}, edges={len(edges)}")
            print(f"Parse time: {parse_time:.3f}s")
            
            # Skip if graph is too large
            if n > 12:  # Limit for reasonable test time
                print(f"⚠️  SKIPPED: Graph too large (n={n}), would take too long")
                results.append({
                    'file': mtx_file,
                    'description': description,
                    'n': n,
                    'edges': len(edges),
                    'status': 'SKIPPED',
                    'reason': 'Too large'
                })
                continue
            
            # Create and test solver
            print(f"🔧 Setting up SAT solver...")
            solver = BandwidthOptimizationSolver(n, 'glucose4')
            solver.set_graph_edges(edges)
            solver.create_position_variables()
            solver.create_distance_variables()
            
            # Solve with reasonable limits
            print(f"🚀 Starting bandwidth optimization...")
            start_time = time.time()
            
            # Set reasonable K range based on graph size
            max_k = min(2 * (n - 1), 8)  # Cap at 8 for testing
            optimal_bandwidth = solver.solve_bandwidth_optimization(start_k=1, end_k=max_k)
            
            solve_time = time.time() - start_time
            
            if optimal_bandwidth is not None:
                print(f"✅ SUCCESS: Optimal bandwidth = {optimal_bandwidth}")
                print(f"Solve time: {solve_time:.3f}s")
                
                results.append({
                    'file': mtx_file,
                    'description': description,
                    'n': n,
                    'edges': len(edges),
                    'optimal_bandwidth': optimal_bandwidth,
                    'solve_time': solve_time,
                    'status': 'SUCCESS'
                })
            else:
                print(f"❌ FAILED: No solution found")
                results.append({
                    'file': mtx_file,
                    'description': description,
                    'n': n,
                    'edges': len(edges),
                    'status': 'FAILED',
                    'reason': 'No solution found'
                })
                
        except Exception as e:
            print(f"💥 ERROR: {e}")
            results.append({
                'file': mtx_file,
                'description': description,
                'status': 'ERROR',
                'reason': str(e)
            })
    
    # Print summary
    print_test_summary(results)
    return results

def test_small_known_graphs():
    """
    Test với các graphs nhỏ có known optimal bandwidth để verify correctness
    """
    print(f"\n" + "="*60)
    print(f"KNOWN OPTIMAL BANDWIDTH VALIDATION")
    print(f"="*60)
    
    known_tests = [
        {
            'name': 'Triangle K3',
            'n': 3,
            'edges': [(1, 2), (2, 3), (1, 3)],
            'expected_optimal': 2,
            'explanation': 'Triangle: optimal placement forms L-shape with bandwidth 2'
        },
        {
            'name': 'Path P4',
            'n': 4,
            'edges': [(1, 2), (2, 3), (3, 4)],
            'expected_optimal': 1,
            'explanation': 'Path: can be placed in straight line with bandwidth 1'
        },
        {
            'name': 'Square C4',
            'n': 4,
            'edges': [(1, 2), (2, 3), (3, 4), (4, 1)],
            'expected_optimal': 2,
            'explanation': 'Square: optimal placement in 2x2 grid with bandwidth 2'
        },
        {
            'name': 'Star S4',
            'n': 4,
            'edges': [(1, 2), (1, 3), (1, 4)],
            'expected_optimal': 1,
            'explanation': 'Star: center at (1,1), leaves at adjacent positions with bandwidth 1'
        }
    ]
    
    validation_results = []
    
    for test in known_tests:
        print(f"\n--- Testing {test['name']} ---")
        print(f"Expected optimal: {test['expected_optimal']}")
        print(f"Explanation: {test['explanation']}")
        
        try:
            solver = BandwidthOptimizationSolver(test['n'], 'glucose4')
            solver.set_graph_edges(test['edges'])
            solver.create_position_variables()
            solver.create_distance_variables()
            
            start_time = time.time()
            optimal = solver.solve_bandwidth_optimization(start_k=1, end_k=4)
            solve_time = time.time() - start_time
            
            if optimal == test['expected_optimal']:
                print(f"✅ VALIDATION SUCCESS: Found optimal = {optimal} (expected {test['expected_optimal']})")
                status = 'CORRECT'
            elif optimal is not None:
                print(f"❌ VALIDATION FAILED: Found optimal = {optimal}, expected {test['expected_optimal']}")
                status = 'INCORRECT'
            else:
                print(f"💥 SOLVER FAILED: No solution found")
                status = 'FAILED'
            
            validation_results.append({
                'name': test['name'],
                'expected': test['expected_optimal'],
                'found': optimal,
                'solve_time': solve_time,
                'status': status
            })
            
        except Exception as e:
            print(f"💥 ERROR: {e}")
            validation_results.append({
                'name': test['name'],
                'expected': test['expected_optimal'],
                'status': 'ERROR',
                'error': str(e)
            })
    
    return validation_results

def print_test_summary(results):
    """
    In summary của test results
    """
    print(f"\n" + "="*80)
    print(f"TEST SUMMARY")
    print(f"="*80)
    
    success_count = sum(1 for r in results if r.get('status') == 'SUCCESS')
    total_count = len(results)
    
    print(f"Total tests: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed/Error: {total_count - success_count}")
    print(f"Success rate: {success_count/total_count*100:.1f}%")
    
    print(f"\nDetailed Results:")
    print(f"{'File':<25} {'Nodes':<6} {'Edges':<6} {'Bandwidth':<10} {'Time':<8} {'Status'}")
    print(f"-" * 80)
    
    for r in results:
        file_short = r['file'].split('/')[-1] if 'file' in r else r.get('name', 'Unknown')
        n = r.get('n', '?')
        edges = r.get('edges', '?')
        bandwidth = r.get('optimal_bandwidth', r.get('found', '?'))
        time_str = f"{r.get('solve_time', 0):.2f}s" if 'solve_time' in r else 'N/A'
        status = r.get('status', 'UNKNOWN')
        
        print(f"{file_short:<25} {n:<6} {edges:<6} {bandwidth:<10} {time_str:<8} {status}")

def main():
    """
    Main test function
    """
    print("🧪 Starting MTX Validation Tests...")
    
    # Test 1: Known optimal graphs for correctness validation
    print(f"\n{'='*20} PHASE 1: CORRECTNESS VALIDATION {'='*20}")
    validation_results = test_small_known_graphs()
    
    # Test 2: Sample MTX files for broader testing
    print(f"\n{'='*20} PHASE 2: MTX FILES TESTING {'='*20}")
    mtx_results = test_sample_mtx_files()
    
    # Overall summary
    print(f"\n" + "="*80)
    print(f"OVERALL VALIDATION SUMMARY")
    print(f"="*80)
    
    correct_validations = sum(1 for r in validation_results if r.get('status') == 'CORRECT')
    total_validations = len(validation_results)
    
    successful_mtx = sum(1 for r in mtx_results if r.get('status') == 'SUCCESS')
    total_mtx = len(mtx_results)
    
    print(f"Correctness validation: {correct_validations}/{total_validations} correct")
    print(f"MTX file testing: {successful_mtx}/{total_mtx} successful")
    
    if correct_validations == total_validations:
        print(f"✅ ALL CORRECTNESS TESTS PASSED - SAT solver is working correctly!")
    else:
        print(f"❌ Some correctness tests failed - need to investigate solver implementation")
    
    if successful_mtx > 0:
        print(f"✅ SAT solver can handle various graph types successfully")
    else:
        print(f"❌ SAT solver failed on all MTX files - may need optimization")

if __name__ == '__main__':
    main()
