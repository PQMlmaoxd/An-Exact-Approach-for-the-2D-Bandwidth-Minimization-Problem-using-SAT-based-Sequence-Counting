#!/usr/bin/env python3
# test_hybrid_performance.py - Performance comparison between encoding methods

"""
Test script to compare performance between standard, cutoff, and hybrid encodings.

This script allows you to test the same graph with the same K value using different
encoding methods to compare:
- Number of variables
- Number of clauses  
- Encoding time
- Solving time
- Total time

Usage:
    python test_hybrid_performance.py <mtx_file> <K> [--solver=glucose42]
    
Examples:
    # Test with default solver (glucose42)
    python test_hybrid_performance.py bcsstk01.mtx 4
    
    # Test with cadical195
    python test_hybrid_performance.py jgl009.mtx 10 --solver=cadical195
    
    # Test larger instance
    python test_hybrid_performance.py ash85.mtx 25 --solver=cadical195
"""

import sys
import os
import time
from custom_k_bandwidth_solver import CustomKBandwidthSolver, calculate_theoretical_upper_bound

def test_encoding_performance(mtx_file: str, K: int, solver_type: str = 'glucose42'):
    """
    Compare performance across all encoding methods
    
    Args:
        mtx_file: Path to MTX file
        K: Target bandwidth
        solver_type: SAT solver to use
    """
    print("=" * 80)
    print("HYBRID ENCODING PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"File: {mtx_file}")
    print(f"Target K: {K}")
    print(f"Solver: {solver_type.upper()}")
    print("=" * 80)
    
    # Test configurations
    configs = [
        ('standard', 0, "Standard (full T variables)"),
        ('cutoff', 0, "Cutoff (UB-optimized)"),
    ]
    
    # First, get theoretical UB to determine hybrid configurations
    print(f"\nCalculating theoretical UB...")
    temp_solver = CustomKBandwidthSolver(mtx_file, 'standard')
    n = temp_solver.n
    ub = calculate_theoretical_upper_bound(n)
    print(f"Graph size: n={n}, Theoretical UB={ub}")
    
    # Add hybrid configurations
    max_replacements = n - 1 - ub
    if max_replacements > 0:
        # Test different replacement levels
        replacement_levels = [
            1,  # Replace only T_UB
            min(5, max_replacements),  # Replace 5 levels (or max if less)
            max_replacements,  # Full replacement (equivalent to cutoff)
        ]
        
        for num_repl in replacement_levels:
            if num_repl <= max_replacements:
                configs.append((
                    'hybrid', 
                    num_repl, 
                    f"Hybrid (replace {num_repl} level{'s' if num_repl > 1 else ''})"
                ))
    
    print(f"\nTesting {len(configs)} encoding configurations...")
    print()
    
    results = []
    
    # Test each configuration
    for i, (method, num_repl, desc) in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"Configuration {i}/{len(configs)}: {desc}")
        print(f"{'='*60}")
        print(f"Method: {method}")
        if method == 'hybrid':
            print(f"Replacements: {num_repl}")
        
        try:
            # Create solver with specific encoding
            solver = CustomKBandwidthSolver(mtx_file, method, num_repl)
            
            # Test bandwidth K
            is_sat, result_info = solver.test_bandwidth_k(K, solver_type)
            
            # Extract metrics
            solve_time = result_info.get('solve_time', 0)
            total_clauses = result_info.get('total_clauses', 0)
            total_variables = result_info.get('total_variables', 0)
            
            results.append({
                'method': method,
                'num_replacements': num_repl,
                'description': desc,
                'is_sat': is_sat,
                'solve_time': solve_time,
                'total_clauses': total_clauses,
                'total_variables': total_variables
            })
            
            print(f"\nResult: {'SAT' if is_sat else 'UNSAT'}")
            print(f"Solve time: {solve_time:.3f}s")
            print(f"Variables: {total_variables}")
            print(f"Clauses: {total_clauses}")
            
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
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print(f"PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"File: {mtx_file}")
    print(f"K: {K}")
    print(f"Graph size: n={n}")
    print(f"Theoretical UB: {ub}")
    print()
    
    # Print table header
    print(f"{'Method':<20} {'Repl':<6} {'Result':<8} {'Time (s)':<10} {'Variables':<12} {'Clauses':<12}")
    print(f"{'-'*80}")
    
    # Print results
    baseline_time = None
    baseline_vars = None
    baseline_clauses = None
    
    for r in results:
        if 'error' in r:
            print(f"{r['description']:<20} {r['num_replacements']:<6} {'ERROR':<8} {'-':<10} {'-':<12} {'-':<12}")
            continue
        
        method = r['method']
        num_repl = r['num_replacements']
        is_sat = r['is_sat']
        solve_time = r['solve_time']
        variables = r['total_variables']
        clauses = r['total_clauses']
        
        result_str = 'SAT' if is_sat else 'UNSAT'
        
        # Set baseline (first successful result)
        if baseline_time is None:
            baseline_time = solve_time
            baseline_vars = variables
            baseline_clauses = clauses
        
        # Calculate relative performance
        time_ratio = solve_time / baseline_time if baseline_time > 0 else 1.0
        vars_ratio = variables / baseline_vars if baseline_vars > 0 else 1.0
        clauses_ratio = clauses / baseline_clauses if baseline_clauses > 0 else 1.0
        
        print(f"{r['description']:<20} {num_repl:<6} {result_str:<8} {solve_time:>6.3f} {time_ratio:>5.2f}x {variables:>7} {vars_ratio:>5.2f}x {clauses:>7} {clauses_ratio:>5.2f}x")
    
    print(f"{'-'*80}")
    
    # Analysis
    print(f"\nKEY OBSERVATIONS:")
    
    # Find best performing method
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        fastest = min(successful_results, key=lambda r: r['solve_time'])
        fewest_vars = min(successful_results, key=lambda r: r['total_variables'])
        fewest_clauses = min(successful_results, key=lambda r: r['total_clauses'])
        
        print(f"1. Fastest encoding: {fastest['description']} ({fastest['solve_time']:.3f}s)")
        print(f"2. Fewest variables: {fewest_vars['description']} ({fewest_vars['total_variables']} vars)")
        print(f"3. Fewest clauses: {fewest_clauses['description']} ({fewest_clauses['total_clauses']} clauses)")
        
        # Check if all methods agree on SAT/UNSAT
        all_sat = [r['is_sat'] for r in successful_results]
        if len(set(all_sat)) == 1:
            print(f"4. All methods agree: {'SAT' if all_sat[0] else 'UNSAT'} ✓")
        else:
            print(f"4. WARNING: Methods disagree on SAT/UNSAT! ✗")
        
        # Hybrid vs Cutoff comparison (if full replacement was tested)
        hybrid_full = [r for r in successful_results if r['method'] == 'hybrid' and r['num_replacements'] == max_replacements]
        cutoff_result = [r for r in successful_results if r['method'] == 'cutoff']
        
        if hybrid_full and cutoff_result:
            h = hybrid_full[0]
            c = cutoff_result[0]
            print(f"\n5. EQUIVALENCE CHECK (Hybrid full replacement vs Cutoff):")
            print(f"   Hybrid: {h['total_variables']} vars, {h['total_clauses']} clauses, {h['solve_time']:.3f}s")
            print(f"   Cutoff: {c['total_variables']} vars, {c['total_clauses']} clauses, {c['solve_time']:.3f}s")
            
            vars_match = abs(h['total_variables'] - c['total_variables']) < 10
            clauses_similar = abs(h['total_clauses'] - c['total_clauses']) < h['total_clauses'] * 0.1
            
            if vars_match and clauses_similar:
                print(f"   ✓ Hybrid full replacement ≈ Cutoff encoding (as expected)")
            else:
                print(f"   ⚠ Some differences detected (may be due to implementation)")
    
    print(f"\n{'='*80}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("=" * 80)
        print("HYBRID ENCODING PERFORMANCE COMPARISON")
        print("=" * 80)
        print("Usage: python test_hybrid_performance.py <mtx_file> <K> [--solver=glucose42|cadical195]")
        print()
        print("Arguments:")
        print("  mtx_file: Name of MTX file")
        print("  K:        Target bandwidth to test")
        print("  --solver: SAT solver (optional, default=glucose42)")
        print()
        print("Examples:")
        print("  # Test with default solver")
        print("  python test_hybrid_performance.py bcsstk01.mtx 4")
        print()
        print("  # Test with cadical195")
        print("  python test_hybrid_performance.py jgl009.mtx 10 --solver=cadical195")
        print()
        print("  # Test larger instance")
        print("  python test_hybrid_performance.py ash85.mtx 25 --solver=cadical195")
        print()
        print("This script will test:")
        print("  1. Standard encoding (full T variables)")
        print("  2. Cutoff encoding (UB-optimized)")
        print("  3. Hybrid encoding with 1 replacement")
        print("  4. Hybrid encoding with 5 replacements (if possible)")
        print("  5. Hybrid encoding with full replacement (equivalent to cutoff)")
        print()
        print("Output:")
        print("  - Performance metrics for each encoding method")
        print("  - Comparison table with relative performance")
        print("  - Key observations and best performing method")
        sys.exit(1)
    
    mtx_file = sys.argv[1]
    
    try:
        K = int(sys.argv[2])
    except ValueError:
        print(f"Error: K must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)
    
    # Parse solver (optional)
    solver_type = 'glucose42'  # default
    for arg in sys.argv[3:]:
        if arg.startswith('--solver='):
            solver_type = arg.split('=')[1]
            if solver_type not in ['glucose42', 'cadical195']:
                print(f"Error: --solver must be 'glucose42' or 'cadical195', got '{solver_type}'")
                sys.exit(1)
    
    # Run performance test
    test_encoding_performance(mtx_file, K, solver_type)
