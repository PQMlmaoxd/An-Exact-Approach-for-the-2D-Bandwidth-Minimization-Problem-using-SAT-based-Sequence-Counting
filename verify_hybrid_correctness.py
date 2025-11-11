#!/usr/bin/env python3
# verify_hybrid_correctness.py - Verify that all encoding methods give same SAT/UNSAT results

"""
Verification script to ensure all encoding methods are semantically equivalent.

This script tests the same (graph, K) pair with all encoding methods and verifies:
1. All methods agree on SAT/UNSAT result
2. If SAT, all methods find valid solutions with actual bandwidth ≤ K
3. Solution extraction and validation works correctly

Usage:
    python verify_hybrid_correctness.py <mtx_file> <K> [--solver=glucose42]
"""

import sys
import os
from custom_k_bandwidth_solver import CustomKBandwidthSolver

def verify_encoding_equivalence(mtx_file: str, K: int, solver_type: str = 'glucose42'):
    """
    Verify that all encoding methods give equivalent results
    
    Args:
        mtx_file: Path to MTX file
        K: Target bandwidth
        solver_type: SAT solver to use
    """
    print("=" * 80)
    print("HYBRID ENCODING CORRECTNESS VERIFICATION")
    print("=" * 80)
    print(f"File: {mtx_file}")
    print(f"Target K: {K}")
    print(f"Solver: {solver_type.upper()}")
    print("=" * 80)
    
    # Get graph info
    temp_solver = CustomKBandwidthSolver(mtx_file, 'standard')
    n = temp_solver.n
    edges = temp_solver.edges
    
    from distance_encoder_hybrid import calculate_theoretical_upper_bound
    ub = calculate_theoretical_upper_bound(n)
    
    print(f"\nGraph Info:")
    print(f"  Vertices: {n}")
    print(f"  Edges: {len(edges)}")
    print(f"  Theoretical UB: {ub}")
    print()
    
    # Test configurations
    configs = [
        ('standard', 0, "Standard"),
        ('cutoff', 0, "Cutoff"),
        ('hybrid', 1, "Hybrid (1 replacement)"),
        ('hybrid', n - 1 - ub, "Hybrid (full replacement)"),
    ]
    
    results = []
    
    print(f"Testing {len(configs)} configurations...")
    print()
    
    # Test each configuration
    for method, num_repl, desc in configs:
        print(f"{'='*60}")
        print(f"Testing: {desc}")
        print(f"{'='*60}")
        
        try:
            solver = CustomKBandwidthSolver(mtx_file, method, num_repl)
            is_sat, result_info = solver.test_bandwidth_k(K, solver_type)
            
            # Extract key info
            solve_time = result_info.get('solve_time', 0)
            solution_info = result_info.get('solution_info')
            
            actual_bw = None
            is_valid = None
            if solution_info:
                actual_bw = solution_info.get('actual_bandwidth')
                is_valid = solution_info.get('is_valid')
            
            results.append({
                'method': method,
                'num_replacements': num_repl,
                'description': desc,
                'is_sat': is_sat,
                'solve_time': solve_time,
                'actual_bandwidth': actual_bw,
                'is_valid': is_valid
            })
            
            print(f"Result: {'SAT' if is_sat else 'UNSAT'}")
            print(f"Solve time: {solve_time:.3f}s")
            if is_sat and actual_bw is not None:
                print(f"Actual bandwidth: {actual_bw}")
                print(f"Valid solution: {'Yes' if is_valid else 'No'}")
            print()
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'method': method,
                'num_replacements': num_repl,
                'description': desc,
                'error': str(e)
            })
            print()
    
    # Verification
    print(f"\n{'='*80}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*80}")
    
    # Check 1: All methods agree on SAT/UNSAT
    sat_results = [r.get('is_sat') for r in results if 'error' not in r]
    
    if len(set(sat_results)) == 1:
        consensus = sat_results[0]
        print(f"✓ CHECK 1: All methods agree on result: {'SAT' if consensus else 'UNSAT'}")
        check1_pass = True
    else:
        print(f"✗ CHECK 1: Methods DISAGREE on SAT/UNSAT!")
        for r in results:
            if 'error' not in r:
                print(f"  {r['description']}: {'SAT' if r['is_sat'] else 'UNSAT'}")
        check1_pass = False
    
    # Check 2: If SAT, all solutions are valid
    if sat_results and sat_results[0]:  # If SAT
        valid_solutions = [r.get('is_valid') for r in results if 'error' not in r and r.get('is_valid') is not None]
        
        if all(valid_solutions):
            print(f"✓ CHECK 2: All SAT solutions are valid")
            check2_pass = True
        else:
            print(f"✗ CHECK 2: Some SAT solutions are INVALID!")
            for r in results:
                if 'error' not in r and r.get('is_valid') is not None:
                    print(f"  {r['description']}: {'Valid' if r['is_valid'] else 'INVALID'}")
            check2_pass = False
        
        # Check 3: All solutions have actual bandwidth ≤ K
        actual_bws = [r.get('actual_bandwidth') for r in results if 'error' not in r and r.get('actual_bandwidth') is not None]
        
        if all(bw <= K for bw in actual_bws):
            print(f"✓ CHECK 3: All solutions satisfy bandwidth ≤ {K}")
            max_bw = max(actual_bws) if actual_bws else -1
            min_bw = min(actual_bws) if actual_bws else -1
            if max_bw == min_bw:
                print(f"  All methods found actual bandwidth = {max_bw}")
            else:
                print(f"  Actual bandwidths range: {min_bw} to {max_bw}")
                for r in results:
                    if 'error' not in r and r.get('actual_bandwidth') is not None:
                        print(f"    {r['description']}: {r['actual_bandwidth']}")
            check3_pass = True
        else:
            print(f"✗ CHECK 3: Some solutions VIOLATE bandwidth ≤ {K}!")
            for r in results:
                if 'error' not in r and r.get('actual_bandwidth') is not None:
                    bw = r['actual_bandwidth']
                    status = '✓' if bw <= K else '✗'
                    print(f"  {r['description']}: {bw} {status}")
            check3_pass = False
    else:
        check2_pass = True
        check3_pass = True
        print(f"  CHECK 2 & 3: N/A (result is UNSAT)")
    
    # Check 4: No errors
    errors = [r for r in results if 'error' in r]
    if not errors:
        print(f"✓ CHECK 4: No errors in any method")
        check4_pass = True
    else:
        print(f"✗ CHECK 4: {len(errors)} method(s) encountered errors:")
        for r in errors:
            print(f"  {r['description']}: {r['error']}")
        check4_pass = False
    
    # Overall verdict
    print(f"\n{'='*80}")
    all_pass = check1_pass and check2_pass and check3_pass and check4_pass
    
    if all_pass:
        print(f"✓✓✓ VERIFICATION PASSED ✓✓✓")
        print(f"All encoding methods are semantically equivalent for this instance.")
    else:
        print(f"✗✗✗ VERIFICATION FAILED ✗✗✗")
        print(f"Some encoding methods produced different or invalid results.")
    
    print(f"{'='*80}")
    
    return all_pass


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("=" * 80)
        print("HYBRID ENCODING CORRECTNESS VERIFICATION")
        print("=" * 80)
        print("Usage: python verify_hybrid_correctness.py <mtx_file> <K> [--solver=glucose42|cadical195]")
        print()
        print("This script verifies that all encoding methods (standard, cutoff, hybrid)")
        print("produce semantically equivalent results for the same (graph, K) pair.")
        print()
        print("Checks performed:")
        print("  1. All methods agree on SAT/UNSAT result")
        print("  2. All SAT solutions are valid")
        print("  3. All solutions satisfy bandwidth ≤ K")
        print("  4. No errors in any method")
        print()
        print("Examples:")
        print("  python verify_hybrid_correctness.py bcsstk01.mtx 4")
        print("  python verify_hybrid_correctness.py jgl009.mtx 10 --solver=cadical195")
        print("  python verify_hybrid_correctness.py ash85.mtx 25 --solver=cadical195")
        sys.exit(1)
    
    mtx_file = sys.argv[1]
    
    try:
        K = int(sys.argv[2])
    except ValueError:
        print(f"Error: K must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)
    
    # Parse solver (optional)
    solver_type = 'glucose42'
    for arg in sys.argv[3:]:
        if arg.startswith('--solver='):
            solver_type = arg.split('=')[1]
            if solver_type not in ['glucose42', 'cadical195']:
                print(f"Error: --solver must be 'glucose42' or 'cadical195', got '{solver_type}'")
                sys.exit(1)
    
    # Run verification
    success = verify_encoding_equivalence(mtx_file, K, solver_type)
    
    sys.exit(0 if success else 1)
