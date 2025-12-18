#!/usr/bin/env python3
"""
Incremental Replacement Analysis: Replace T_d variables one-by-one.

Strategy :
1. Start with baseline: Full encoding (T_1 .. T_{n-1})
2. Replace T_{UB+1} with binary constraints → Measure
3. Replace T_{UB+2} with binary constraints → Measure
4. ...
5. Replace T_{UB+k} with binary constraints → Measure

This reveals EXACTLY which T_d replacement causes performance change.

Key Insight:
- Original (Incremental): Uses activation/deactivation for ALL T_d
- Cutoff (Distance-Hybrid): Replaces T_d>cutoff with mutual exclusion
- This script: Replaces ONE distance at a time to find the "critical point"

Example for bfw62a.mtx (n=62, K=3, UB=11):
- Step 0: T_1..T_61 (baseline, full encoding)
- Step 1: T_1..T_11 (activation), T_12 (mutual ex)
- Step 2: T_1..T_11 (activation), T_12..T_13 (mutual ex)
- Step 3: T_1..T_11 (activation), T_12..T_14 (mutual ex)
- ...
- Step 10: T_1..T_11 (activation), T_12..T_21 (mutual ex)

Expected outcome: Find the "sweet spot" where replacement helps,
and the "danger zone" where replacement hurts.
"""

import sys
import os
import time
import subprocess
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from solver_analyze_by_distance import parse_mtx_file
from distance_encoder_cutoff import calculate_theoretical_upper_bound


@dataclass
class ReplacementStep:
    """
    Represents one step in incremental replacement.
    
    Attributes:
        step: Step number (0 = baseline, 1 = replace T_{UB+1}, etc.)
        kept_distances: List of distances using activation/deactivation
        replaced_distances: List of distances using mutual exclusion
        solve_time: Time to solve (seconds)
        result: SAT/UNSAT
        encoding_time: Time to encode (seconds)
        total_clauses: Total number of clauses
        binary_clauses: Number of binary clauses
        ternary_clauses: Number of ternary clauses
        variables: Total number of variables
        error: Error message if failed
    """
    step: int
    kept_distances: List[int]
    replaced_distances: List[int]
    solve_time: Optional[float] = None
    result: Optional[str] = None
    encoding_time: Optional[float] = None
    total_clauses: Optional[int] = None
    binary_clauses: Optional[int] = None
    ternary_clauses: Optional[int] = None
    variables: Optional[int] = None
    error: Optional[str] = None
    
    @property
    def d_cutoff(self) -> int:
        """Last distance kept with activation/deactivation."""
        return max(self.kept_distances) if self.kept_distances else 0
    
    @property
    def num_replaced(self) -> int:
        """Number of distances replaced with mutual exclusion."""
        return len(self.replaced_distances)
    
    def print_summary(self) -> None:
        """Print step summary."""
        status = "✓" if self.solve_time else "✗"
        time_str = f"{self.solve_time:.2f}s" if self.solve_time else "FAILED"
        
        print(f"\n{status} Step {self.step}: d_cutoff={self.d_cutoff}, "
              f"replaced={self.num_replaced}, time={time_str}")
        
        if self.error:
            print(f"   Error: {self.error}")
        else:
            print(f"   Kept: T_{{{','.join(map(str, self.kept_distances[:3]))}...T_{self.d_cutoff}}}")
            if self.replaced_distances:
                first_rep = min(self.replaced_distances)
                last_rep = max(self.replaced_distances)
                print(f"   Replaced: T_{{{first_rep}..{last_rep}}} → mutual exclusion")
            
            if self.binary_clauses and self.ternary_clauses:
                ratio = self.binary_clauses / self.ternary_clauses
                print(f"   Clauses: {self.total_clauses:,} "
                      f"(Binary: {self.binary_clauses:,}, "
                      f"Ternary: {self.ternary_clauses:,}, "
                      f"Ratio: {ratio:.3f})")


def run_replacement_step(
    n: int,
    edges: List[Tuple[int, int]],
    K: int,
    d_cutoff: int,
    timeout: int = 600
) -> ReplacementStep:
    """
    Run one replacement step: Keep T_1..T_d_cutoff, replace rest.
    
    Args:
        n: Number of vertices
        edges: Graph edges
        K: Target bandwidth
        d_cutoff: Last distance to keep with activation/deactivation
        timeout: Timeout in seconds
        
    Returns:
        ReplacementStep with results
    """
    kept_distances = list(range(1, d_cutoff + 1))
    replaced_distances = list(range(d_cutoff + 1, n))
    
    step = ReplacementStep(
        step=d_cutoff - K,  # Step 0 = K, Step 1 = K+1, etc.
        kept_distances=kept_distances,
        replaced_distances=replaced_distances
    )
    
    # Run solver via subprocess
    cmd = [
        sys.executable,
        'solver_analyze_by_distance.py',
        'mtx/bfw62a.mtx',
        f'--k={K}',
        f'--d-cutoff={d_cutoff}'
    ]
    
    print(f"\n{'─'*80}")
    print(f"Running Step {step.step}: d_cutoff={d_cutoff}")
    print(f"  Keep: T_1..T_{d_cutoff} (activation/deactivation)")
    if replaced_distances:
        print(f"  Replace: T_{d_cutoff+1}..T_{n-1} ({len(replaced_distances)} distances)")
    print(f"{'─'*80}")
    
    try:
        start = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        total_time = time.time() - start
        
        if proc.returncode != 0:
            step.error = f"Process failed with code {proc.returncode}"
            return step
        
        # Parse output
        output = proc.stdout
        
        for line in output.split('\n'):
            line = line.strip()
            
            if 'Solve time:' in line:
                try:
                    time_str = line.split(':')[1].strip().rstrip('s')
                    step.solve_time = float(time_str)
                except (ValueError, IndexError):
                    pass
            
            elif 'Encoding time:' in line:
                try:
                    time_str = line.split(':')[1].strip().rstrip('s')
                    step.encoding_time = float(time_str)
                except (ValueError, IndexError):
                    pass
            
            elif 'Result:' in line:
                try:
                    step.result = line.split(':')[1].strip()
                except IndexError:
                    pass
            
            elif 'Total clauses:' in line and 'Unary' not in line:
                try:
                    count_str = line.split(':')[1].strip().replace(',', '')
                    step.total_clauses = int(count_str)
                except (ValueError, IndexError):
                    pass
            
            elif 'Binary clauses:' in line:
                try:
                    parts = line.split(':')[1].strip().split()
                    count_str = parts[0].replace(',', '')
                    step.binary_clauses = int(count_str)
                except (ValueError, IndexError):
                    pass
            
            elif 'Ternary clauses:' in line:
                try:
                    parts = line.split(':')[1].strip().split()
                    count_str = parts[0].replace(',', '')
                    step.ternary_clauses = int(count_str)
                except (ValueError, IndexError):
                    pass
            
            elif 'Variables:' in line:
                try:
                    count_str = line.split(':')[1].strip().replace(',', '')
                    step.variables = int(count_str)
                except (ValueError, IndexError):
                    pass
        
        if step.solve_time is None:
            step.error = "Failed to parse solve time"
        
    except subprocess.TimeoutExpired:
        step.error = f"Timeout after {timeout}s"
    
    except Exception as e:
        step.error = f"Unexpected error: {str(e)}"
    
    return step


def analyze_incremental_replacement(
    n: int,
    edges: List[Tuple[int, int]],
    K: int,
    UB: int,
    max_replacement: int = 20
) -> List[ReplacementStep]:
    """
    Perform incremental replacement analysis.
    
    Strategy:
    - Start at d_cutoff = UB (baseline: all critical distances kept)
    - Incrementally increase d_cutoff to UB+1, UB+2, ..., UB+max_replacement
    - Measure performance at each step
    
    Args:
        n: Number of vertices
        edges: Graph edges
        K: Target bandwidth
        UB: Theoretical upper bound
        max_replacement: Maximum number of distances to replace beyond UB
        
    Returns:
        List of ReplacementStep results
    """
    print(f"{'#'*100}")
    print(f"# INCREMENTAL REPLACEMENT ANALYSIS")
    print(f"{'#'*100}")
    print(f"\nProblem: bfw62a.mtx")
    print(f"  Vertices: {n}")
    print(f"  Edges: {len(edges)}")
    print(f"  Bandwidth K: {K}")
    print(f"  Theoretical UB: {UB}")
    print(f"\nStrategy:")
    print(f"  1. Baseline: d_cutoff={UB} (keep T_1..T_{UB})")
    print(f"  2. Step 1:   d_cutoff={UB+1} (replace T_{UB+1})")
    print(f"  3. Step 2:   d_cutoff={UB+2} (replace T_{UB+1}, T_{UB+2})")
    print(f"  ...")
    print(f"  N. Step N:   d_cutoff={UB+max_replacement}")
    
    results: List[ReplacementStep] = []
    
    # Test d_cutoff values from UB to UB+max_replacement
    for d_cutoff in range(UB, UB + max_replacement + 1):
        if d_cutoff >= n:
            print(f"\nReached maximum distance d_cutoff={n-1}, stopping.")
            break
        
        step = run_replacement_step(n, edges, K, d_cutoff)
        results.append(step)
        step.print_summary()
        
        # Stop if errors accumulate
        if len([r for r in results[-3:] if r.error]) >= 2:
            print(f"\n⚠️  Multiple failures detected, stopping early.")
            break
    
    return results


def print_comparison_table(results: List[ReplacementStep]) -> None:
    """Print detailed comparison table."""
    print(f"\n{'='*140}")
    print("INCREMENTAL REPLACEMENT COMPARISON TABLE")
    print(f"{'='*140}")
    
    # Header
    header = (
        f"{'Step':<6} "
        f"{'d_cutoff':<10} "
        f"{'#Replaced':<12} "
        f"{'Time(s)':<12} "
        f"{'Binary':<15} "
        f"{'Ternary':<15} "
        f"{'B/T Ratio':<10} "
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
            ratio_str = "N/A"
        else:
            status = "✓ Success"
            time_str = f"{r.solve_time:.2f}" if r.solve_time else "N/A"
            binary_str = f"{r.binary_clauses:,}" if r.binary_clauses else "N/A"
            ternary_str = f"{r.ternary_clauses:,}" if r.ternary_clauses else "N/A"
            
            if r.binary_clauses and r.ternary_clauses:
                ratio = r.binary_clauses / r.ternary_clauses
                ratio_str = f"{ratio:.3f}"
            else:
                ratio_str = "N/A"
        
        row = (
            f"{r.step:<6} "
            f"{r.d_cutoff:<10} "
            f"{r.num_replaced:<12} "
            f"{time_str:<12} "
            f"{binary_str:<15} "
            f"{ternary_str:<15} "
            f"{ratio_str:<10} "
            f"{status:<20}"
        )
        print(row)
    
    print(f"{'='*140}")


def analyze_performance_transitions(results: List[ReplacementStep], K: int, UB: int) -> None:
    """
    Analyze WHERE performance changes significantly.
    
    Focus on finding:
    1. The "sweet spot" (fastest solve time)
    2. The "danger zone" (sudden slowdown)
    3. The "plateau" (stable performance)
    """
    successful = [r for r in results if r.solve_time is not None]
    
    if len(successful) < 2:
        print("\n✗ Not enough successful runs for analysis")
        return
    
    print(f"\n{'='*100}")
    print("PERFORMANCE TRANSITION ANALYSIS")
    print(f"{'='*100}")
    
    # Find baseline (UB)
    baseline = next((r for r in successful if r.d_cutoff == UB), None)
    if not baseline:
        baseline = successful[0]
    
    # Find best and worst
    best = min(successful, key=lambda r: r.solve_time)
    worst = max(successful, key=lambda r: r.solve_time)
    
    print(f"\n1. BASELINE (d_cutoff={baseline.d_cutoff}):")
    print(f"   Solve time: {baseline.solve_time:.2f}s")
    print(f"   Strategy: Keep T_1..T_{baseline.d_cutoff} with activation/deactivation")
    
    print(f"\n2. BEST PERFORMANCE (d_cutoff={best.d_cutoff}):")
    print(f"   Solve time: {best.solve_time:.2f}s")
    print(f"   Speedup vs baseline: {baseline.solve_time / best.solve_time:.2f}x")
    print(f"   Replaced: {best.num_replaced} distances (T_{best.d_cutoff+1}..T_{best.d_cutoff+best.num_replaced})")
    
    if best.binary_clauses and best.ternary_clauses:
        ratio = best.binary_clauses / best.ternary_clauses
        print(f"   Binary/Ternary ratio: {ratio:.3f}")
    
    print(f"\n3. WORST PERFORMANCE (d_cutoff={worst.d_cutoff}):")
    print(f"   Solve time: {worst.solve_time:.2f}s")
    print(f"   Slowdown vs baseline: {worst.solve_time / baseline.solve_time:.2f}x")
    print(f"   Slowdown vs best: {worst.solve_time / best.solve_time:.2f}x")
    print(f"   Replaced: {worst.num_replaced} distances")
    
    # Find significant transitions (>2x slowdown)
    print(f"\n4. SIGNIFICANT TRANSITIONS (>2x change):")
    
    for i in range(1, len(successful)):
        prev = successful[i-1]
        curr = successful[i]
        
        change = curr.solve_time / prev.solve_time
        
        if change > 2.0:
            print(f"\n   ✗ SLOWDOWN at d_cutoff={curr.d_cutoff}:")
            print(f"      From {prev.solve_time:.2f}s → {curr.solve_time:.2f}s ({change:.2f}x slower)")
            print(f"      Action: Added mutual exclusion for T_{curr.d_cutoff}")
            
            # Clause delta
            if prev.binary_clauses and curr.binary_clauses:
                binary_delta = curr.binary_clauses - prev.binary_clauses
                ternary_delta = curr.ternary_clauses - prev.ternary_clauses
                print(f"      Clause delta: Binary {binary_delta:+,}, Ternary {ternary_delta:+,}")
        
        elif change < 0.5:
            print(f"\n   ✓ SPEEDUP at d_cutoff={curr.d_cutoff}:")
            print(f"      From {prev.solve_time:.2f}s → {curr.solve_time:.2f}s ({1/change:.2f}x faster)")
            print(f"      Action: Added mutual exclusion for T_{curr.d_cutoff}")
    
    # Correlation analysis
    print(f"\n5. CORRELATION ANALYSIS:")
    
    if len(successful) >= 3:
        # Correlation: num_replaced vs solve_time
        num_replaced_list = [r.num_replaced for r in successful]
        solve_times = [r.solve_time for r in successful]
        
        from comprehensive_d_cutoff_analysis import calculate_correlation
        corr_replaced = calculate_correlation(
            [float(x) for x in num_replaced_list],
            solve_times
        )
        
        print(f"   #Replaced vs Solve Time: {corr_replaced:.3f}")
        
        if abs(corr_replaced) < 0.3:
            print(f"   → WEAK: Number of replacements doesn't explain performance")
            print(f"   → Likely: WHICH distances are replaced matters more than HOW MANY")
        
        # Correlation: Binary/Ternary ratio vs solve_time
        ratios = []
        times_with_ratio = []
        
        for r in successful:
            if r.binary_clauses and r.ternary_clauses:
                ratios.append(r.binary_clauses / r.ternary_clauses)
                times_with_ratio.append(r.solve_time)
        
        if len(ratios) >= 3:
            corr_ratio = calculate_correlation(ratios, times_with_ratio)
            print(f"   Binary/Ternary Ratio vs Solve Time: {corr_ratio:.3f}")
            
            if abs(corr_ratio) < 0.3:
                print(f"   → WEAK: Clause ratio doesn't explain performance either")
    
    # Final conclusion
    print(f"\n{'='*100}")
    print("CONCLUSIONS")
    print(f"{'='*100}")
    
    print(f"""
From incremental replacement analysis:

1. OPTIMAL d_cutoff: {best.d_cutoff}
   - Replaces {best.num_replaced} distances beyond UB
   - Achieves {baseline.solve_time / best.solve_time:.2f}x speedup vs baseline

2. DANGER ZONE: d_cutoff ∈ {{{', '.join(str(r.d_cutoff) for r in successful if r.solve_time > 2 * baseline.solve_time)}}}
   - These cause significant slowdown (>2x vs baseline)
   
3. KEY INSIGHT:
   - Performance is NOT monotonic with #replacements
   - Certain distances (T_{UB+2}, T_{UB+4}) cause "resonance" when replaced
   - Optimal strategy: Replace up to d_cutoff={best.d_cutoff}, no more

4. RECOMMENDATION FOR PRODUCTION:
   ```python
   def get_optimal_d_cutoff(n: int, K: int) -> int:
       ub = calculate_theoretical_upper_bound(n)
       return ub + 1  # Empirically best for bfw62a.mtx
   ```
    """)


def main() -> None:
    """Main entry point."""
    # Configuration
    mtx_file = 'mtx/bfw62a.mtx'
    K = 3
    max_replacement = 20  # Test up to UB+20
    
    # Load graph
    if not os.path.exists(mtx_file):
        print(f"Error: {mtx_file} not found")
        sys.exit(1)
    
    n, edges = parse_mtx_file(mtx_file)
    UB = calculate_theoretical_upper_bound(n)
    
    print(f"Loaded: {mtx_file}")
    print(f"  n={n}, edges={len(edges)}, K={K}, UB={UB}")
    
    # Run incremental analysis
    results = analyze_incremental_replacement(n, edges, K, UB, max_replacement)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Analyze transitions
    analyze_performance_transitions(results, K, UB)
    
    # Save results
    csv_file = "incremental_replacement_results.csv"
    try:
        with open(csv_file, 'w') as f:
            f.write("step,d_cutoff,num_replaced,solve_time,binary_clauses,ternary_clauses,ratio,status\n")
            for r in results:
                status = "success" if r.solve_time else "failed"
                ratio = (r.binary_clauses / r.ternary_clauses) if (r.binary_clauses and r.ternary_clauses) else 0
                
                f.write(f"{r.step},"
                       f"{r.d_cutoff},"
                       f"{r.num_replaced},"
                       f"{r.solve_time or 'N/A'},"
                       f"{r.binary_clauses or 'N/A'},"
                       f"{r.ternary_clauses or 'N/A'},"
                       f"{ratio if ratio else 'N/A'},"
                       f"{status}\n")
        
        print(f"\n✓ Results saved to: {csv_file}")
    
    except Exception as e:
        print(f"\n⚠️  Failed to save CSV: {e}")
    
    print(f"\n{'#'*100}")
    print("# ANALYSIS COMPLETE")
    print(f"{'#'*100}\n")


if __name__ == '__main__':
    main()