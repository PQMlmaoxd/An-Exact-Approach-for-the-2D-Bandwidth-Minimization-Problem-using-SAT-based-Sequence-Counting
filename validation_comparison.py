# validation_comparison.py
"""
So sánh validation methods giữa Complete DRSA và SAT-based approaches
cho bài toán 2D Bandwidth Minimization Problem
"""

import sys
import time
import traceback
from typing import Dict, List, Any

# Import both implementations
sys.path.append('.')
try:
    from complete_drsa_implementation import TwoDBMP_DRSA
    DRSA_AVAILABLE = True
except ImportError:
    print("Complete DRSA implementation not available")
    DRSA_AVAILABLE = False

try:
    from bandwidth_optimization_solver import BandwidthOptimizationSolver
    SAT_AVAILABLE = True
except ImportError:
    print("SAT bandwidth solver not available")
    SAT_AVAILABLE = False

class ValidationComparison:
    """
    Lớp so sánh validation methods giữa DRSA và SAT approaches
    """
    
    def __init__(self):
        self.test_cases = [
            {
                'name': 'Single Edge',
                'n': 2,
                'edges': [(1, 2)],
                'expected_optimal': 1,
                'description': 'Trivial case - minimum graph'
            },
            {
                'name': 'Triangle',
                'n': 3,
                'edges': [(1, 2), (2, 3), (1, 3)],
                'expected_optimal': 2,
                'description': 'Complete triangle K3'
            },
            {
                'name': 'Path P4',
                'n': 4,
                'edges': [(1, 2), (2, 3), (3, 4)],
                'expected_optimal': 1,
                'description': 'Linear path graph'
            },
            {
                'name': 'Square C4',
                'n': 4,
                'edges': [(1, 2), (2, 3), (3, 4), (4, 1)],
                'expected_optimal': 1,
                'description': '4-cycle graph'
            }
        ]
        
        self.drsa_results = []
        self.sat_results = []
    
    def validate_drsa_approach(self, test_case: Dict) -> Dict[str, Any]:
        """
        Validate Complete DRSA approach
        
        Returns:
            dict: Validation results với metrics chi tiết
        """
        if not DRSA_AVAILABLE:
            return {'status': 'unavailable', 'error': 'DRSA implementation not found'}
        
        result = {
            'method': 'Complete DRSA',
            'test_case': test_case['name'],
            'n': test_case['n'],
            'edges_count': len(test_case['edges']),
            'expected_optimal': test_case['expected_optimal']
        }
        
        try:
            print(f"\nTesting DRSA: {test_case['name']}")
            
            # Initialize DRSA
            drsa = TwoDBMP_DRSA(
                graph_vertices=test_case['n'],
                graph_edges=test_case['edges'],
                seed=42
            )
            
            # Set parameters based on problem size
            if test_case['n'] <= 3:
                drsa.L = 40
                drsa.T_final = 0.001
                drsa.T0 = 100.0
                max_time = 5  # seconds
            else:
                drsa.L = 60
                drsa.T_final = 0.001
                drsa.T0 = 150.0
                max_time = 10  # seconds
            
            # Run DRSA
            start_time = time.time()
            phi_best, gamma_best, stats = drsa.TwoDBMP_DRSA(verbose=False)
            solve_time = time.time() - start_time
            
            # Extract results
            bandwidth_found = drsa.extract_bandwidth(phi_best)
            
            # Validation metrics
            result.update({
                'status': 'success',
                'bandwidth_found': bandwidth_found,
                'gamma_value': gamma_best,
                'solve_time': solve_time,
                'iterations': stats.get('iterations', 0),
                'acceptance_rate': stats.get('accepted_moves', 0) / max(stats.get('accepted_moves', 0) + stats.get('rejected_moves', 0), 1) * 100,
                'optimality_gap': abs(bandwidth_found - test_case['expected_optimal']),
                'is_optimal': bandwidth_found == test_case['expected_optimal'],
                'solution_quality': 'OPTIMAL' if bandwidth_found == test_case['expected_optimal'] else 'SUBOPTIMAL',
                'validation_method': 'Stochastic search with γ-discrimination',
                'solution_representation': 'Permutation φ with bandwidth calculation',
                'convergence_criterion': 'Annealing schedule completion'
            })
            
            print(f"   DRSA Result: bandwidth={bandwidth_found}, γ={gamma_best:.6f}, time={solve_time:.2f}s")
            
        except Exception as e:
            result.update({
                'status': 'error',
                'error': str(e),
                'solve_time': 0,
                'bandwidth_found': None
            })
            print(f"   DRSA Error: {e}")
        
        return result
    
    def validate_sat_approach(self, test_case: Dict) -> Dict[str, Any]:
        """
        Validate SAT-based approach
        
        Returns:
            dict: Validation results với metrics chi tiết
        """
        if not SAT_AVAILABLE:
            return {'status': 'unavailable', 'error': 'SAT implementation not found'}
        
        result = {
            'method': 'SAT-based',
            'test_case': test_case['name'],
            'n': test_case['n'],
            'edges_count': len(test_case['edges']),
            'expected_optimal': test_case['expected_optimal']
        }
        
        try:
            print(f"\nTesting SAT: {test_case['name']}")
            
            # Initialize SAT solver
            solver = BandwidthOptimizationSolver(test_case['n'], 'glucose4')
            solver.set_graph_edges(test_case['edges'])
            solver.create_position_variables()
            solver.create_distance_variables()
            
            # Run optimization
            start_time = time.time()
            optimal_bandwidth = solver.solve_bandwidth_optimization(
                start_k=1, 
                end_k=min(6, 2 * test_case['n'])  # Reasonable upper bound
            )
            solve_time = time.time() - start_time
            
            # Validation metrics
            if optimal_bandwidth is not None:
                result.update({
                    'status': 'success',
                    'bandwidth_found': optimal_bandwidth,
                    'solve_time': solve_time,
                    'optimality_gap': abs(optimal_bandwidth - test_case['expected_optimal']),
                    'is_optimal': optimal_bandwidth == test_case['expected_optimal'],
                    'solution_quality': 'OPTIMAL' if optimal_bandwidth == test_case['expected_optimal'] else 'SUBOPTIMAL',
                    'validation_method': 'SAT solving with constraint encoding',
                    'solution_representation': 'Variable assignment with constraint satisfaction',
                    'convergence_criterion': 'UNSAT proof or SAT model found',
                    'proof_type': 'Mathematical proof of optimality',
                    'search_completeness': 'Complete (guaranteed to find optimal if exists)'
                })
                
                print(f"   SAT Result: bandwidth={optimal_bandwidth}, time={solve_time:.2f}s")
            else:
                result.update({
                    'status': 'timeout_or_error',
                    'bandwidth_found': None,
                    'solve_time': solve_time,
                    'error': 'No solution found within bounds'
                })
                print(f"   SAT: No solution found")
                
        except Exception as e:
            result.update({
                'status': 'error',
                'error': str(e),
                'solve_time': 0,
                'bandwidth_found': None
            })
            print(f"   SAT Error: {e}")
            traceback.print_exc()
        
        return result
    
    def run_validation_comparison(self):
        """
        Chạy so sánh validation toàn diện
        """
        print("=== VALIDATION METHODS COMPARISON ===")
        print("Comparing Complete DRSA vs SAT-based approaches")
        print("="*70)
        
        for test_case in self.test_cases:
            print(f"\nTest Case: {test_case['name']} (n={test_case['n']}, E={len(test_case['edges'])})")
            print(f"   Expected optimal: {test_case['expected_optimal']}")
            print(f"   Description: {test_case['description']}")
            
            # Validate DRSA
            drsa_result = self.validate_drsa_approach(test_case)
            self.drsa_results.append(drsa_result)
            
            # Validate SAT
            sat_result = self.validate_sat_approach(test_case)
            self.sat_results.append(sat_result)
            
            # Compare results
            self.compare_single_case(drsa_result, sat_result)
        
        # Print comprehensive comparison
        self.print_comprehensive_comparison()
    
    def compare_single_case(self, drsa_result: Dict, sat_result: Dict):
        """
        So sánh kết quả của một test case
        """
        print(f"\n   Comparison for {drsa_result['test_case']}:")
        
        # Accuracy comparison
        drsa_optimal = drsa_result.get('is_optimal', False)
        sat_optimal = sat_result.get('is_optimal', False)
        
        print(f"      Optimality: DRSA={'Yes' if drsa_optimal else 'No'}, SAT={'Yes' if sat_optimal else 'No'}")
        
        # Time comparison
        drsa_time = drsa_result.get('solve_time', float('inf'))
        sat_time = sat_result.get('solve_time', float('inf'))
        
        if drsa_time < float('inf') and sat_time < float('inf'):
            faster = 'DRSA' if drsa_time < sat_time else 'SAT'
            print(f"      Speed: DRSA={drsa_time:.2f}s, SAT={sat_time:.2f}s (Winner: {faster})")
        
        # Solution quality
        drsa_bw = drsa_result.get('bandwidth_found', 'N/A')
        sat_bw = sat_result.get('bandwidth_found', 'N/A')
        print(f"      Bandwidth: DRSA={drsa_bw}, SAT={sat_bw}")
    
    def print_comprehensive_comparison(self):
        """
        In bảng so sánh toàn diện
        """
        print(f"\n" + "="*100)
        print(f"=== COMPREHENSIVE VALIDATION COMPARISON ===")
        print(f"="*100)
        
        # Summary table
        print(f"{'Test Case':<12} {'Method':<8} {'Status':<10} {'BW Found':<8} {'Expected':<8} {'Optimal':<7} {'Time(s)':<8} {'Validation Type':<20}")
        print("-"*100)
        
        for i, test_case in enumerate(self.test_cases):
            drsa_res = self.drsa_results[i] if i < len(self.drsa_results) else {}
            sat_res = self.sat_results[i] if i < len(self.sat_results) else {}
            
            # DRSA row
            drsa_status = drsa_res.get('status', 'N/A')
            drsa_bw = drsa_res.get('bandwidth_found', 'N/A')
            drsa_optimal = 'Yes' if drsa_res.get('is_optimal', False) else 'No'
            drsa_time = f"{drsa_res.get('solve_time', 0):.2f}"
            
            print(f"{test_case['name']:<12} {'DRSA':<8} {drsa_status:<10} {str(drsa_bw):<8} {test_case['expected_optimal']:<8} {drsa_optimal:<7} {drsa_time:<8} {'Stochastic':<20}")
            
            # SAT row
            sat_status = sat_res.get('status', 'N/A')
            sat_bw = sat_res.get('bandwidth_found', 'N/A')
            sat_optimal = 'Yes' if sat_res.get('is_optimal', False) else 'No'
            sat_time = f"{sat_res.get('solve_time', 0):.2f}"
            
            print(f"{'':12} {'SAT':<8} {sat_status:<10} {str(sat_bw):<8} {test_case['expected_optimal']:<8} {sat_optimal:<7} {sat_time:<8} {'Exact/Complete':<20}")
            print()
        
        # Analysis summary
        self.print_validation_analysis()
    
    def print_validation_analysis(self):
        """
        In phân tích validation methods
        """
        print("\n=== VALIDATION METHODS ANALYSIS ===")
        print("="*60)
        
        # Count successful cases
        drsa_success = sum(1 for r in self.drsa_results if r.get('status') == 'success')
        sat_success = sum(1 for r in self.sat_results if r.get('status') == 'success')
        
        drsa_optimal_count = sum(1 for r in self.drsa_results if r.get('is_optimal', False))
        sat_optimal_count = sum(1 for r in self.sat_results if r.get('is_optimal', False))
        
        print(f"Success Rate:")
        print(f"  DRSA: {drsa_success}/{len(self.test_cases)} ({drsa_success/len(self.test_cases)*100:.1f}%)")
        print(f"  SAT:  {sat_success}/{len(self.test_cases)} ({sat_success/len(self.test_cases)*100:.1f}%)")
        
        print(f"\nOptimality Rate:")
        print(f"  DRSA: {drsa_optimal_count}/{len(self.test_cases)} ({drsa_optimal_count/len(self.test_cases)*100:.1f}%)")
        print(f"  SAT:  {sat_optimal_count}/{len(self.test_cases)} ({sat_optimal_count/len(self.test_cases)*100:.1f}%)")
        
        # Validation characteristics
        print(f"\nVALIDATION CHARACTERISTICS:")
        print(f"\nComplete DRSA:")
        print(f"  Strengths:")
        print(f"     - Fast heuristic search với γ-discrimination")
        print(f"     - Good for large instances")
        print(f"     - Detailed convergence statistics")
        print(f"     - Academic-compliant with DRSA methodology")
        print(f"  Limitations:")
        print(f"     - Stochastic - không guarantee optimal")
        print(f"     - Solution quality depends on parameters")
        print(f"     - Validation based on multiple runs")
        
        print(f"\nSAT-based Approach:")
        print(f"  Strengths:")
        print(f"     - Mathematical proof of optimality")
        print(f"     - Complete search guarantee")
        print(f"     - Exact solution with verification")
        print(f"     - NSC encoding for efficiency")
        print(f"  Limitations:")
        print(f"     - Can be slow for large instances")
        print(f"     - Memory intensive")
        print(f"     - Limited to smaller problem sizes")
        
        print(f"\nRECOMMENDATION:")
        print(f"  - Use SAT for small instances requiring proven optimality")
        print(f"  - Use DRSA for large instances or when good solutions suffice")
        print(f"  - Hybrid approach: DRSA for UB, SAT for verification")

def main():
    """
    Main validation comparison
    """
    print("Starting Validation Methods Comparison")
    print("Comparing Complete DRSA vs SAT-based validation approaches")
    print("="*70)
    
    validator = ValidationComparison()
    validator.run_validation_comparison()
    
    print(f"\nValidation comparison completed!")
    print(f"Results saved in validator.drsa_results and validator.sat_results")

if __name__ == '__main__':
    main()
