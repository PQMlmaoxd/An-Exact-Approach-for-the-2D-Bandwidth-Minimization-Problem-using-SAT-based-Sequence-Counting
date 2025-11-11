#!/usr/bin/env python3
"""
Comprehensive analysis: Test ALL key d_cutoff values to find the true optimal.

This script tests:
- d_cutoff = K+1 = 4 (minimal, just enough for bandwidth)
- d_cutoff = UB = 11 (theoretical upper bound)
- d_cutoff = UB+1 = 12 (current best observed)
- d_cutoff = UB+2 = 13 (worst observed)
- d_cutoff = 20, 30, 40 (larger values to see monotonic trend)

No external dependencies required (no pandas, matplotlib).
"""

import subprocess
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    """Results from a single d_cutoff experiment."""
    d_cutoff: int
    description: str
    solve_time: Optional[float] = None
    binary_pct: Optional[float] = None
    ternary_pct: Optional[float] = None
    total_clauses: Optional[int] = None
    unary_clauses: Optional[int] = None
    binary_clauses: Optional[int] = None
    ternary_clauses: Optional[int] = None
    error: Optional[str] = None


def run_experiment(d_cutoff: int, description: str) -> ExperimentResult:
    """
    Run solver with given d_cutoff and capture results.
    
    Args:
        d_cutoff: Distance cutoff value to test
        description: Human-readable description of this test point
        
    Returns:
        ExperimentResult with parsed metrics or error information
    """
    result = ExperimentResult(d_cutoff=d_cutoff, description=description)
    
    cmd = [
        sys.executable,
        'solver_analyze_by_distance.py',
        'bfw62a.mtx',
        '--k=3',
        f'--d-cutoff={d_cutoff}'
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: d_cutoff = {d_cutoff} ({description})")
    print(f"{'='*80}")
    
    try:
        proc = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=600  # 10 minute timeout per experiment
        )
        
        if proc.returncode != 0:
            result.error = f"Process failed with return code {proc.returncode}"
            print(f"✗ FAILED: {result.error}")
            return result
        
        # Parse output
        output = proc.stdout
        
        for line in output.split('\n'):
            line = line.strip()
            
            if 'Solve time:' in line:
                try:
                    time_str = line.split(':')[1].strip().rstrip('s')
                    result.solve_time = float(time_str)
                except (ValueError, IndexError):
                    pass
            
            elif 'Binary clause ratio:' in line:
                try:
                    pct_str = line.split(':')[1].strip().rstrip('%')
                    result.binary_pct = float(pct_str)
                except (ValueError, IndexError):
                    pass
            
            elif 'Total clauses:' in line and 'Unary' not in line:
                try:
                    count_str = line.split(':')[1].strip().replace(',', '')
                    result.total_clauses = int(count_str)
                except (ValueError, IndexError):
                    pass
            
            elif 'Unary clauses:' in line:
                try:
                    parts = line.split(':')[1].strip().split()
                    count_str = parts[0].replace(',', '')
                    result.unary_clauses = int(count_str)
                except (ValueError, IndexError):
                    pass
            
            elif 'Binary clauses:' in line:
                try:
                    parts = line.split(':')[1].strip().split()
                    count_str = parts[0].replace(',', '')
                    result.binary_clauses = int(count_str)
                except (ValueError, IndexError):
                    pass
            
            elif 'Ternary clauses:' in line:
                try:
                    parts = line.split(':')[1].strip().split()
                    count_str = parts[0].replace(',', '')
                    result.ternary_clauses = int(count_str)
                except (ValueError, IndexError):
                    pass
        
        # Calculate ternary percentage
        if result.total_clauses and result.ternary_clauses:
            result.ternary_pct = 100.0 * result.ternary_clauses / result.total_clauses
        
        if result.solve_time is None:
            result.error = "Failed to parse solve time from output"
            print(f"✗ FAILED: {result.error}")
        else:
            print(f"✓ SUCCESS: {result.solve_time:.2f}s")
        
    except subprocess.TimeoutExpired:
        result.error = "Experiment timed out (>10 minutes)"
        print(f"✗ TIMEOUT: {result.error}")
    
    except Exception as e:
        result.error = f"Unexpected error: {str(e)}"
        print(f"✗ ERROR: {result.error}")
    
    return result


def calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = (var_x * var_y) ** 0.5
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def print_results_table(results: List[ExperimentResult]) -> None:
    """Print results in a formatted table."""
    print(f"\n{'='*140}")
    print("RESULTS TABLE")
    print(f"{'='*140}")
    
    # Header
    header = (
        f"{'d_cutoff':<10} "
        f"{'Description':<20} "
        f"{'Time(s)':<12} "
        f"{'Binary%':<10} "
        f"{'Ternary%':<10} "
        f"{'Total Clauses':<15} "
        f"{'Status':<20}"
    )
    print(header)
    print('─' * 140)
    
    # Rows
    for r in results:
        if r.error:
            status = f"✗ {r.error[:15]}"
            time_str = "N/A"
            binary_str = "N/A"
            ternary_str = "N/A"
            clauses_str = "N/A"
        else:
            status = "✓ Success"
            time_str = f"{r.solve_time:.2f}" if r.solve_time else "N/A"
            binary_str = f"{r.binary_pct:.2f}" if r.binary_pct else "N/A"
            ternary_str = f"{r.ternary_pct:.2f}" if r.ternary_pct else "N/A"
            clauses_str = f"{r.total_clauses:,}" if r.total_clauses else "N/A"
        
        row = (
            f"{r.d_cutoff:<10} "
            f"{r.description:<20} "
            f"{time_str:<12} "
            f"{binary_str:<10} "
            f"{ternary_str:<10} "
            f"{clauses_str:<15} "
            f"{status:<20}"
        )
        print(row)
    
    print(f"{'='*140}")


def analyze_results(results: List[ExperimentResult], K: int, UB: int) -> None:
    """Perform comprehensive analysis of results."""
    
    # Filter successful results
    successful = [r for r in results if r.solve_time is not None]
    
    if not successful:
        print("\n✗ No successful experiments to analyze!")
        return
    
    print(f"\n{'='*120}")
    print("KEY FINDINGS")
    print(f"{'='*120}")
    
    # 1. Best and worst performance
    best = min(successful, key=lambda r: r.solve_time)
    worst = max(successful, key=lambda r: r.solve_time)
    
    print(f"\n1. BEST PERFORMANCE:")
    print(f"   d_cutoff = {best.d_cutoff} ({best.description})")
    print(f"   Solve time: {best.solve_time:.2f}s")
    print(f"   Binary ratio: {best.binary_pct:.2f}%")
    print(f"   Ternary ratio: {best.ternary_pct:.2f}%")
    
    print(f"\n2. WORST PERFORMANCE:")
    print(f"   d_cutoff = {worst.d_cutoff} ({worst.description})")
    print(f"   Solve time: {worst.solve_time:.2f}s")
    print(f"   Binary ratio: {worst.binary_pct:.2f}%")
    print(f"   Ternary ratio: {worst.ternary_pct:.2f}%")
    print(f"   Slowdown: {worst.solve_time / best.solve_time:.2f}x")
    
    # 2. Correlation analysis
    binary_pcts = [r.binary_pct for r in successful if r.binary_pct is not None]
    solve_times = [r.solve_time for r in successful if r.binary_pct is not None]
    
    if len(binary_pcts) >= 2:
        correlation = calculate_correlation(binary_pcts, solve_times)
        
        print(f"\n3. CORRELATION ANALYSIS:")
        print(f"   Binary% vs Solve Time: {correlation:.3f}")
        
        if correlation < -0.5:
            print(f"   → STRONG NEGATIVE: Higher binary ratio = Faster solving")
        elif correlation > 0.5:
            print(f"   → STRONG POSITIVE: Higher binary ratio = Slower solving (!)")
        else:
            print(f"   → WEAK: Binary ratio doesn't fully explain performance")
    
    # 3. Pattern analysis
    print(f"\n4. PATTERN ANALYSIS:")
    
    # Check if K+1 is best
    k_plus_1_results = [r for r in successful if r.d_cutoff == K + 1]
    if k_plus_1_results and k_plus_1_results[0].solve_time == best.solve_time:
        print(f"   ✓ K+1 = {K+1} IS OPTIMAL!")
        print(f"     → Minimal d_cutoff strategy is best")
    else:
        if k_plus_1_results:
            k1_time = k_plus_1_results[0].solve_time
            print(f"   ✗ K+1 = {K+1} is NOT optimal ({k1_time:.2f}s vs best {best.solve_time:.2f}s)")
        else:
            print(f"   ? K+1 = {K+1} was not tested")
    
    # Check UB performance
    ub_results = [r for r in successful if r.d_cutoff == UB]
    if ub_results:
        ub_time = ub_results[0].solve_time
        speedup_vs_ub = ub_time / best.solve_time
        if ub_time == best.solve_time:
            print(f"   ✓ UB = {UB} is OPTIMAL!")
        else:
            print(f"   ✗ UB = {UB} is NOT optimal")
            print(f"     → {speedup_vs_ub:.2f}x slower than best")
    
    # Check monotonicity for large values
    large_results = [r for r in successful if r.d_cutoff >= 20]
    large_results.sort(key=lambda r: r.d_cutoff)
    
    if len(large_results) >= 2:
        times = [r.solve_time for r in large_results]
        is_monotonic = all(times[i] <= times[i+1] for i in range(len(times)-1))
        
        if is_monotonic:
            print(f"   ✓ Large d_cutoff values show MONOTONIC degradation")
        else:
            print(f"   ✗ Non-monotonic even for large values")
    
    # 4. Speedup analysis
    print(f"\n5. SPEEDUP ANALYSIS (vs d_cutoff={best.d_cutoff}):")
    for r in successful:
        speedup = r.solve_time / best.solve_time
        status = "✓" if speedup < 1.1 else "⚠️" if speedup < 2.0 else "✗"
        print(f"   {status} d_cutoff={r.d_cutoff:2d}: {r.solve_time:7.2f}s ({speedup:5.2f}x)")
    
    # 5. Final recommendation
    print(f"\n{'='*120}")
    print("FINAL RECOMMENDATION")
    print(f"{'='*120}")
    
    print(f"""
Based on comprehensive testing of {len(successful)} d_cutoff values:

OPTIMAL STRATEGY: Use d_cutoff = {best.d_cutoff} ({best.description})
├─ Solve time: {best.solve_time:.2f}s
├─ Binary ratio: {best.binary_pct:.2f}%
└─ Ternary ratio: {best.ternary_pct:.2f}%

DANGER ZONE: Avoid d_cutoff in range [{worst.d_cutoff-1}, {worst.d_cutoff+1}]
├─ These values cause up to {worst.solve_time / best.solve_time:.2f}x slowdown
└─ Likely due to solver heuristic interference

KEY INSIGHTS:
1. For bfw62a.mtx (n=62, K=3): d_cutoff = {best.d_cutoff} is optimal
2. UB={UB} is {'OPTIMAL' if ub_results and ub_results[0].solve_time == best.solve_time else 'NOT optimal'}
3. K+1={K+1} is {'OPTIMAL' if k_plus_1_results and k_plus_1_results[0].solve_time == best.solve_time else 'NOT optimal'}

GENERAL STRATEGY:
- ALWAYS test multiple d_cutoff values: {{K+1, UB, UB+1, UB+2}}
- DO NOT assume UB is optimal (it's graph-specific!)
- Optimal d_cutoff depends on:
  ├─ Clause structure (binary/ternary ratio)
  ├─ Solver heuristics (VSIDS, restart policy)
  └─ Problem-specific search space geometry

IMPLEMENTATION:
```python
def choose_optimal_d_cutoff_empirical(n: int, K: int) -> int:
    '''
    Choose optimal d_cutoff via empirical testing.
    
    Returns the d_cutoff with fastest solve time among
    candidate values {{K+1, UB, UB+1}}.
    '''
    ub = calculate_theoretical_upper_bound(n)
    candidates = [K + 1, ub, ub + 1]
    
    # Test each candidate and pick the fastest
    # (In practice, use a small timeout and parallel testing)
    best_d_cutoff = ub  # Default fallback
    
    # ... run experiments and select best ...
    
    return best_d_cutoff
```

For production use with bfw62a.mtx: d_cutoff = {best.d_cutoff}
    """)


def main() -> None:
    """Run comprehensive d_cutoff analysis."""
    
    # Configuration
    K = 3
    UB = 11
    
    # Test points
    test_points: List[Tuple[int, str]] = [
        (K + 1, "K+1 (minimal)"),
        (UB, "UB (theoretical)"),
        (UB + 1, "UB+1"),
        (UB + 2, "UB+2"),
        (UB + 3, "UB+3"),
        (UB + 4, "UB+4"),
        (20, "Large gap"),
        (30, "Very large gap"),
    ]
    
    print(f"{'#'*120}")
    print("# COMPREHENSIVE d_cutoff ANALYSIS")
    print(f"{'#'*120}")
    print(f"\nGraph: bfw62a.mtx")
    print(f"Bandwidth K: {K}")
    print(f"Theoretical UB: {UB}")
    print(f"\nTesting {len(test_points)} d_cutoff values...")
    
    # Run experiments
    results: List[ExperimentResult] = []
    
    for d_cutoff, description in test_points:
        result = run_experiment(d_cutoff, description)
        results.append(result)
    
    # Display results
    print_results_table(results)
    
    # Analyze
    analyze_results(results, K, UB)
    
    # Save results to CSV (simple format)
    csv_file = "d_cutoff_results.csv"
    try:
        with open(csv_file, 'w') as f:
            f.write("d_cutoff,description,solve_time,binary_pct,ternary_pct,total_clauses,status\n")
            for r in results:
                status = "success" if r.solve_time else "failed"
                f.write(f"{r.d_cutoff},"
                       f"{r.description},"
                       f"{r.solve_time or 'N/A'},"
                       f"{r.binary_pct or 'N/A'},"
                       f"{r.ternary_pct or 'N/A'},"
                       f"{r.total_clauses or 'N/A'},"
                       f"{status}\n")
        print(f"\n✓ Results saved to: {csv_file}")
    except Exception as e:
        print(f"\n⚠️  Failed to save CSV: {e}")
    
    print(f"\n{'#'*120}")
    print("# ANALYSIS COMPLETE")
    print(f"{'#'*120}\n")


if __name__ == '__main__':
    main()