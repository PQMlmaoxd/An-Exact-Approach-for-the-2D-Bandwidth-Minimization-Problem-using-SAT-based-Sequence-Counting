#!/usr/bin/env python3
"""
debug_runner.py - Comprehensive debug system for bandwidth solver
Runs both solvers from scratch with detailed instrumentation
"""

import sys
import time
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json

# Import both solvers
from incremental_bandwidth_solver import IncrementalBandwidthSolver
from incremental_bandwidth_solver_cutoff import IncrementalBandwidthSolverCutoff


@dataclass
class ConstraintStats:
    """Statistics for constraint generation"""
    activation_clauses: int = 0
    deactivation_clauses: int = 0
    monotonic_clauses: int = 0
    base_clauses: int = 0
    mutual_exclusion_clauses: int = 0  # Cutoff only
    total_clauses: int = 0
    t_variables: int = 0
    
    def to_dict(self):
        return {
            'activation': self.activation_clauses,
            'deactivation': self.deactivation_clauses,
            'monotonic': self.monotonic_clauses,
            'base': self.base_clauses,
            'mutual_exclusion': self.mutual_exclusion_clauses,
            'total': self.total_clauses,
            't_variables': self.t_variables
        }


@dataclass
class SolveStats:
    """Statistics for a single K solve"""
    k: int
    result: str  # 'SAT' or 'UNSAT'
    solve_time: float
    actual_bandwidth: Optional[int]
    solver_vars: int = 0
    solver_clauses: int = 0
    bandwidth_clauses_added: int = 0
    
    def to_dict(self):
        return {
            'k': self.k,
            'result': self.result,
            'solve_time': self.solve_time,
            'actual_bandwidth': self.actual_bandwidth,
            'solver_vars': self.solver_vars,
            'solver_clauses': self.solver_clauses,
            'bandwidth_clauses_added': self.bandwidth_clauses_added
        }


@dataclass
class DebugReport:
    """Complete debug report for a solver run"""
    solver_name: str
    n: int
    edges: int
    ub: int
    optimal: Optional[int] = None
    
    # Encoding stats
    position_stats: Dict = field(default_factory=dict)
    distance_stats: ConstraintStats = field(default_factory=ConstraintStats)
    
    # Per-K stats
    k_stats: List[SolveStats] = field(default_factory=list)
    
    # Timing
    total_encoding_time: float = 0.0
    total_solve_time: float = 0.0
    
    def to_dict(self):
        return {
            'solver_name': self.solver_name,
            'n': self.n,
            'edges': self.edges,
            'ub': self.ub,
            'optimal': self.optimal,
            'position_stats': self.position_stats,
            'distance_stats': self.distance_stats.to_dict(),
            'k_stats': [s.to_dict() for s in self.k_stats],
            'total_encoding_time': self.total_encoding_time,
            'total_solve_time': self.total_solve_time
        }
    
    def save_json(self, filename: str):
        """Save report to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class InstrumentedOriginalSolver(IncrementalBandwidthSolver):
    """Original solver with debug instrumentation"""
    
    def __init__(self, n, solver_type='cadical195'):
        super().__init__(n, solver_type)
        self.debug_stats = DebugReport('Original', n, 0, 0)
        self.constraint_counts = {}
    
    def encode_distance_constraints(self):
        """Instrumented version - count constraint types"""
        print("\n[DEBUG] Original: Encoding distance constraints...")
        
        t0 = time.time()
        stats = ConstraintStats()
        
        for edge_id, (u, v) in zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges):
            # Count clauses per edge
            edge_activation = 0
            edge_deactivation = 0
            edge_monotonic = 0
            edge_base = 0
            
            # X distance
            from distance_encoder import encode_abs_distance_final
            Tx_vars, Tx_clauses = encode_abs_distance_final(
                self.X_vars[u], self.X_vars[v], self.n, self.vpool, f"Tx_{edge_id}"
            )
            self.Tx_vars[edge_id] = Tx_vars
            stats.t_variables += len(Tx_vars)
            
            # Count clause types for X
            for clause in Tx_clauses:
                self.persistent_solver.add_clause(clause)
                if len(clause) == 3:
                    if clause[2] > 0:  # Activation: ... → T_d
                        edge_activation += 1
                    else:  # Deactivation: ... → ¬T_d
                        edge_deactivation += 1
                elif len(clause) == 2:
                    edge_monotonic += 1
            
            Tx_clauses.clear()
            del Tx_clauses
            
            # Y distance
            Ty_vars, Ty_clauses = encode_abs_distance_final(
                self.Y_vars[u], self.Y_vars[v], self.n, self.vpool, f"Ty_{edge_id}"
            )
            self.Ty_vars[edge_id] = Ty_vars
            stats.t_variables += len(Ty_vars)
            
            # Count clause types for Y
            for clause in Ty_clauses:
                self.persistent_solver.add_clause(clause)
                if len(clause) == 3:
                    if clause[2] > 0:
                        edge_activation += 1
                    else:
                        edge_deactivation += 1
                elif len(clause) == 2:
                    edge_monotonic += 1
            
            Ty_clauses.clear()
            del Ty_clauses
            
            # Accumulate stats
            stats.activation_clauses += edge_activation
            stats.deactivation_clauses += edge_deactivation
            stats.monotonic_clauses += edge_monotonic
            stats.base_clauses += edge_base
        
        enc_time = time.time() - t0
        
        # Get final solver stats
        vars_after = self.persistent_solver.nof_vars()
        cls_after = self.persistent_solver.nof_clauses()
        
        stats.total_clauses = cls_after - self.debug_stats.position_stats.get('clauses', 0)
        
        # Print detailed stats
        print(f"[DEBUG] Original Distance Encoding:")
        print(f"  T variables created: {stats.t_variables}")
        print(f"  Activation clauses: {stats.activation_clauses}")
        print(f"  Deactivation clauses: {stats.deactivation_clauses}")
        print(f"  Monotonic clauses: {stats.monotonic_clauses}")
        print(f"  Base clauses: {stats.base_clauses}")
        print(f"  Total clauses: {stats.total_clauses}")
        print(f"  Per edge (avg): {stats.total_clauses / (len(self.edges) * 2):.1f}")
        print(f"  Encoding time: {enc_time:.3f}s")
        
        self.debug_stats.distance_stats = stats
        self.debug_stats.total_encoding_time += enc_time
        
        print(f"Distance encoding time: {enc_time:.3f}s")
        print(f"Solver stats after distance: vars={vars_after}, clauses={cls_after}")
        
        return enc_time
    
    def encode_bandwidth_constraints_for_k(self, K):
        """Instrumented version - detailed logging"""
        if K in self.current_k_constraints:
            return []
        
        print(f"\n[DEBUG] Original: Encoding bandwidth constraints K={K}")
        
        new_clauses = []
        constraint_breakdown = {
            'unary': 0,
            'binary': 0
        }
        
        t0 = time.time()
        
        for edge_id in self.Tx_vars:
            Tx = self.Tx_vars[edge_id]
            Ty = self.Ty_vars[edge_id]
            
            # Unary constraints
            if K < len(Tx):
                new_clauses.append([-Tx[K]])
                constraint_breakdown['unary'] += 1
            
            if K < len(Ty):
                new_clauses.append([-Ty[K]])
                constraint_breakdown['unary'] += 1
            
            # Binary constraints (both directions)
            for i in range(1, K + 1):
                remaining = K - i
                if remaining >= 0:
                    # Direction 1
                    if i-1 < len(Tx) and remaining < len(Ty):
                        new_clauses.append([-Tx[i-1], -Ty[remaining]])
                        constraint_breakdown['binary'] += 1
                    
                    # Direction 2
                    if i-1 < len(Ty) and remaining < len(Tx):
                        new_clauses.append([-Ty[i-1], -Tx[remaining]])
                        constraint_breakdown['binary'] += 1
        
        encode_time = time.time() - t0
        
        self.current_k_constraints.add(K)
        
        print(f"  Unary constraints: {constraint_breakdown['unary']}")
        print(f"  Binary constraints: {constraint_breakdown['binary']}")
        print(f"  Total: {len(new_clauses)}")
        print(f"  Expected: {len(self.edges) * (2 + 4*K)}")
        print(f"  Encoding time: {encode_time:.6f}s")
        
        if len(new_clauses) != len(self.edges) * (2 + 4*K):
            print(f"  ⚠️  WARNING: Mismatch in clause count!")
        
        return new_clauses
    
    def solve_with_instrumentation(self, upper_bound):
        """Instrumented solve with detailed stats"""
        print(f"\n{'='*80}")
        print(f"ORIGINAL SOLVER - INSTRUMENTED RUN")
        print(f"{'='*80}")
        
        self._initialize_persistent_solver()
        
        from incremental_bandwidth_solver import calculate_theoretical_upper_bound
        theoretical_ub = calculate_theoretical_upper_bound(self.n)
        current_k = min(upper_bound, theoretical_ub)
        
        self.debug_stats.ub = theoretical_ub
        
        optimal_k = None
        
        while current_k >= 1:
            print(f"\n{'─'*80}")
            print(f"Testing K = {current_k}")
            print(f"{'─'*80}")
            
            # Add constraints
            bandwidth_clauses = self.encode_bandwidth_constraints_for_k(current_k)
            
            for clause in bandwidth_clauses:
                self.persistent_solver.add_clause(clause)
            
            clause_count = len(bandwidth_clauses)
            bandwidth_clauses.clear()
            del bandwidth_clauses
            
            # Solve
            print(f"  Solving...")
            solve_start = time.time()
            result = self.persistent_solver.solve()
            solve_time = time.time() - solve_start
            
            # Record stats
            solve_stat = SolveStats(
                k=current_k,
                result='SAT' if result else 'UNSAT',
                solve_time=solve_time,
                actual_bandwidth=None,
                solver_vars=self.persistent_solver.nof_vars(),
                solver_clauses=self.persistent_solver.nof_clauses(),
                bandwidth_clauses_added=clause_count
            )
            
            if result:
                print(f"  Result: SATISFIABLE")
                print(f"  Solve time: {solve_time:.3f}s")
                
                model = self.persistent_solver.get_model()
                self.last_model = model
                actual_bandwidth = self.extract_actual_bandwidth(model)
                solve_stat.actual_bandwidth = actual_bandwidth
                
                print(f"  Actual bandwidth: {actual_bandwidth}")
                
                if actual_bandwidth < current_k:
                    print(f"  Smart jump: {current_k} → {actual_bandwidth}")
                    optimal_k = actual_bandwidth
                    current_k = actual_bandwidth - 1
                else:
                    optimal_k = current_k
                    current_k -= 1
            else:
                print(f"  Result: UNSATISFIABLE")
                print(f"  Solve time: {solve_time:.3f}s")
                break
            
            self.debug_stats.k_stats.append(solve_stat)
            self.debug_stats.total_solve_time += solve_time
        
        self.debug_stats.optimal = optimal_k
        
        print(f"\n{'='*80}")
        print(f"ORIGINAL SOLVER COMPLETE")
        print(f"Optimal: {optimal_k}")
        print(f"Total solve time: {self.debug_stats.total_solve_time:.2f}s")
        print(f"{'='*80}")
        
        return optimal_k


class InstrumentedCutoffSolver(IncrementalBandwidthSolverCutoff):
    """Cutoff solver with debug instrumentation"""
    
    def __init__(self, n, solver_type='cadical195'):
        super().__init__(n, solver_type)
        self.debug_stats = DebugReport('Cutoff', n, 0, self.theoretical_ub)
        self.constraint_counts = {}
    
    def encode_distance_constraints(self):
        """Instrumented version - count constraint types"""
        print("\n[DEBUG] Cutoff: Encoding distance constraints...")
        
        t0 = time.time()
        stats = ConstraintStats()
        
        for edge_id, (u, v) in zip([f'edge_{u}_{v}' for u, v in self.edges], self.edges):
            edge_activation = 0
            edge_deactivation = 0
            edge_monotonic = 0
            edge_base = 0
            edge_mutual = 0
            
            # X distance
            tx_prefix = self.Tx_vars[edge_id]['prefix']
            from distance_encoder_cutoff import encode_abs_distance_cutoff
            tx_clauses, tx_vars = encode_abs_distance_cutoff(
                self.X_vars[u], self.X_vars[v],
                self.theoretical_ub, self.vpool, tx_prefix
            )
            self.Tx_vars[edge_id]['vars'] = tx_vars
            stats.t_variables += len(tx_vars)
            
            # Count clause types for X
            for clause in tx_clauses:
                self.persistent_solver.add_clause(clause)
                clen = len(clause)
                if clen == 2:
                    # Could be mutual exclusion or monotonic
                    # Mutual exclusion: both literals negative
                    if all(lit < 0 for lit in clause):
                        # Check if involves position variables
                        edge_mutual += 1
                    else:
                        edge_monotonic += 1
                elif clen == 3:
                    if clause[2] > 0:
                        edge_activation += 1
                    else:
                        edge_deactivation += 1
            
            tx_clauses.clear()
            del tx_clauses
            
            # Y distance
            ty_prefix = self.Ty_vars[edge_id]['prefix']
            ty_clauses, ty_vars = encode_abs_distance_cutoff(
                self.Y_vars[u], self.Y_vars[v],
                self.theoretical_ub, self.vpool, ty_prefix
            )
            self.Ty_vars[edge_id]['vars'] = ty_vars
            stats.t_variables += len(ty_vars)
            
            # Count clause types for Y
            for clause in ty_clauses:
                self.persistent_solver.add_clause(clause)
                clen = len(clause)
                if clen == 2:
                    if all(lit < 0 for lit in clause):
                        edge_mutual += 1
                    else:
                        edge_monotonic += 1
                elif clen == 3:
                    if clause[2] > 0:
                        edge_activation += 1
                    else:
                        edge_deactivation += 1
            
            ty_clauses.clear()
            del ty_clauses
            
            # Accumulate stats
            stats.activation_clauses += edge_activation
            stats.deactivation_clauses += edge_deactivation
            stats.monotonic_clauses += edge_monotonic
            stats.base_clauses += edge_base
            stats.mutual_exclusion_clauses += edge_mutual
        
        enc_time = time.time() - t0
        
        # Get final solver stats
        vars_after = self.persistent_solver.nof_vars()
        cls_after = self.persistent_solver.nof_clauses()
        
        stats.total_clauses = cls_after - self.debug_stats.position_stats.get('clauses', 0)
        
        # Print detailed stats
        print(f"[DEBUG] Cutoff Distance Encoding:")
        print(f"  T variables created: {stats.t_variables}")
        print(f"  Activation clauses: {stats.activation_clauses}")
        print(f"  Deactivation clauses: {stats.deactivation_clauses}")
        print(f"  Monotonic clauses: {stats.monotonic_clauses}")
        print(f"  Base clauses: {stats.base_clauses}")
        print(f"  Mutual exclusion clauses: {stats.mutual_exclusion_clauses}")
        print(f"  Total clauses: {stats.total_clauses}")
        print(f"  Per edge (avg): {stats.total_clauses / (len(self.edges) * 2):.1f}")
        print(f"  Encoding time: {enc_time:.3f}s")
        
        self.debug_stats.distance_stats = stats
        self.debug_stats.total_encoding_time += enc_time
        
        print(f"Distance encoding time: {enc_time:.3f}s")
        print(f"Solver stats after distance: vars={vars_after}, clauses={cls_after}")
        
        return enc_time
    
    def encode_bandwidth_constraints_for_k(self, K):
        """Instrumented version - detailed logging"""
        if K in self.current_k_constraints:
            return []
        
        print(f"\n[DEBUG] Cutoff: Encoding bandwidth constraints K={K}")
        
        effective_k = min(K, self.theoretical_ub)
        
        new_clauses = []
        constraint_breakdown = {
            'unary': 0,
            'binary': 0,
            'missing_unary': 0,
            'missing_binary': 0
        }
        
        t0 = time.time()
        
        for edge_id in self.Tx_vars:
            tx_vars = self.Tx_vars[edge_id]['vars']
            ty_vars = self.Ty_vars[edge_id]['vars']
            
            # Unary constraints
            if (effective_k + 1) in tx_vars:
                new_clauses.append([-tx_vars[effective_k + 1]])
                constraint_breakdown['unary'] += 1
            else:
                constraint_breakdown['missing_unary'] += 1
            
            if (effective_k + 1) in ty_vars:
                new_clauses.append([-ty_vars[effective_k + 1]])
                constraint_breakdown['unary'] += 1
            else:
                constraint_breakdown['missing_unary'] += 1
            
            # Binary constraints (both directions)
            for i in range(1, effective_k + 1):
                remaining = effective_k - i
                if remaining >= 0:
                    # Direction 1
                    if i in tx_vars and (remaining + 1) in ty_vars:
                        new_clauses.append([-tx_vars[i], -ty_vars[remaining + 1]])
                        constraint_breakdown['binary'] += 1
                    else:
                        constraint_breakdown['missing_binary'] += 1
                    
                    # Direction 2
                    if i in ty_vars and (remaining + 1) in tx_vars:
                        new_clauses.append([-ty_vars[i], -tx_vars[remaining + 1]])
                        constraint_breakdown['binary'] += 1
                    else:
                        constraint_breakdown['missing_binary'] += 1
        
        encode_time = time.time() - t0
        
        self.current_k_constraints.add(K)
        
        print(f"  Effective K: {effective_k}")
        print(f"  Unary constraints: {constraint_breakdown['unary']}")
        print(f"  Binary constraints: {constraint_breakdown['binary']}")
        print(f"  Missing unary: {constraint_breakdown['missing_unary']}")
        print(f"  Missing binary: {constraint_breakdown['missing_binary']}")
        print(f"  Total: {len(new_clauses)}")
        print(f"  Expected: {len(self.edges) * (2 + 4*effective_k)}")
        print(f"  Encoding time: {encode_time:.6f}s")
        
        if constraint_breakdown['missing_unary'] > 0 or constraint_breakdown['missing_binary'] > 0:
            print(f"  ⚠️  WARNING: Missing constraints detected!")
        
        if len(new_clauses) != len(self.edges) * (2 + 4*effective_k):
            print(f"  ⚠️  WARNING: Mismatch in clause count!")
        
        return new_clauses
    
    def solve_with_instrumentation(self, upper_bound):
        """Instrumented solve with detailed stats"""
        print(f"\n{'='*80}")
        print(f"CUTOFF SOLVER - INSTRUMENTED RUN")
        print(f"{'='*80}")
        
        self._initialize_persistent_solver()
        
        current_k = min(upper_bound, self.theoretical_ub)
        optimal_k = None
        
        while current_k >= 1:
            print(f"\n{'─'*80}")
            print(f"Testing K = {current_k}")
            print(f"{'─'*80}")
            
            # Add constraints
            bandwidth_clauses = self.encode_bandwidth_constraints_for_k(current_k)
            
            for clause in bandwidth_clauses:
                self.persistent_solver.add_clause(clause)
            
            clause_count = len(bandwidth_clauses)
            bandwidth_clauses.clear()
            del bandwidth_clauses
            
            # Solve
            print(f"  Solving...")
            solve_start = time.time()
            result = self.persistent_solver.solve()
            solve_time = time.time() - solve_start
            
            # Record stats
            solve_stat = SolveStats(
                k=current_k,
                result='SAT' if result else 'UNSAT',
                solve_time=solve_time,
                actual_bandwidth=None,
                solver_vars=self.persistent_solver.nof_vars(),
                solver_clauses=self.persistent_solver.nof_clauses(),
                bandwidth_clauses_added=clause_count
            )
            
            if result:
                print(f"  Result: SATISFIABLE")
                print(f"  Solve time: {solve_time:.3f}s")
                
                model = self.persistent_solver.get_model()
                self.last_model = model
                actual_bandwidth = self.extract_actual_bandwidth(model)
                solve_stat.actual_bandwidth = actual_bandwidth
                
                print(f"  Actual bandwidth: {actual_bandwidth}")
                
                if actual_bandwidth < current_k:
                    print(f"  Smart jump: {current_k} → {actual_bandwidth}")
                    optimal_k = actual_bandwidth
                    current_k = actual_bandwidth - 1
                else:
                    optimal_k = current_k
                    current_k -= 1
            else:
                print(f"  Result: UNSATISFIABLE")
                print(f"  Solve time: {solve_time:.3f}s")
                break
            
            self.debug_stats.k_stats.append(solve_stat)
            self.debug_stats.total_solve_time += solve_time
        
        self.debug_stats.optimal = optimal_k
        
        print(f"\n{'='*80}")
        print(f"CUTOFF SOLVER COMPLETE")
        print(f"Optimal: {optimal_k}")
        print(f"Total solve time: {self.debug_stats.total_solve_time:.2f}s")
        print(f"{'='*80}")
        
        return optimal_k


def parse_mtx_file(filename: str) -> Tuple[int, List[Tuple[int, int]]]:
    """Parse MTX file"""
    print(f"Reading MTX file: {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    header_found = False
    edges_set = set()
    n = 0
    
    for line in lines:
        line = line.strip()
        
        if not line or line.startswith('%'):
            continue
        
        if not header_found:
            parts = line.split()
            if len(parts) >= 2:
                n = int(parts[0])
                header_found = True
                continue
        
        parts = line.split()
        if len(parts) >= 2:
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                edges_set.add((min(u, v), max(u, v)))
    
    edges = list(edges_set)
    print(f"Loaded: n={n}, |E|={len(edges)}")
    return n, edges


def compare_reports(orig: DebugReport, cut: DebugReport):
    """Generate comprehensive comparison"""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nGraph: n={orig.n}, |E|={orig.edges}, UB={orig.ub}")
    print(f"Optimal: {orig.optimal} (ratio: {orig.optimal/orig.ub:.3f})")
    
    # Distance encoding comparison
    print(f"\n{'─'*80}")
    print(f"DISTANCE ENCODING COMPARISON")
    print(f"{'─'*80}")
    
    print(f"\n{'Metric':<30} {'Original':<20} {'Cutoff':<20} {'Diff %':<10}")
    print(f"{'-'*80}")
    
    orig_t = orig.distance_stats.t_variables
    cut_t = cut.distance_stats.t_variables
    diff_t = ((cut_t - orig_t) / orig_t * 100) if orig_t > 0 else 0
    print(f"{'T variables':<30} {orig_t:<20} {cut_t:<20} {diff_t:>+8.1f}%")
    
    orig_act = orig.distance_stats.activation_clauses
    cut_act = cut.distance_stats.activation_clauses
    diff_act = ((cut_act - orig_act) / orig_act * 100) if orig_act > 0 else 0
    print(f"{'Activation clauses':<30} {orig_act:<20} {cut_act:<20} {diff_act:>+8.1f}%")
    
    orig_deact = orig.distance_stats.deactivation_clauses
    cut_deact = cut.distance_stats.deactivation_clauses
    diff_deact = ((cut_deact - orig_deact) / orig_deact * 100) if orig_deact > 0 else 0
    print(f"{'Deactivation clauses':<30} {orig_deact:<20} {cut_deact:<20} {diff_deact:>+8.1f}%")
    
    orig_mono = orig.distance_stats.monotonic_clauses
    cut_mono = cut.distance_stats.monotonic_clauses
    diff_mono = ((cut_mono - orig_mono) / orig_mono * 100) if orig_mono > 0 else 0
    print(f"{'Monotonic clauses':<30} {orig_mono:<20} {cut_mono:<20} {diff_mono:>+8.1f}%")
    
    cut_mut = cut.distance_stats.mutual_exclusion_clauses
    print(f"{'Mutual exclusion clauses':<30} {'N/A':<20} {cut_mut:<20} {'N/A':<10}")
    
    orig_tot = orig.distance_stats.total_clauses
    cut_tot = cut.distance_stats.total_clauses
    diff_tot = ((cut_tot - orig_tot) / orig_tot * 100) if orig_tot > 0 else 0
    print(f"{'Total distance clauses':<30} {orig_tot:<20} {cut_tot:<20} {diff_tot:>+8.1f}%")
    
    # Per-K comparison
    print(f"\n{'─'*80}")
    print(f"PER-K SOLVE TIME COMPARISON")
    print(f"{'─'*80}")
    
    print(f"\n{'K':<8} {'Original (s)':<15} {'Cutoff (s)':<15} {'Ratio':<10} {'Winner':<15}")
    print(f"{'-'*70}")
    
    problem_k = []
    
    # Match K values
    orig_dict = {s.k: s for s in orig.k_stats}
    cut_dict = {s.k: s for s in cut.k_stats}
    
    all_k = sorted(set(orig_dict.keys()) | set(cut_dict.keys()), reverse=True)
    
    for k in all_k:
        if k in orig_dict and k in cut_dict:
            orig_time = orig_dict[k].solve_time
            cut_time = cut_dict[k].solve_time
            ratio = cut_time / orig_time if orig_time > 0 else float('inf')
            
            winner = "Cutoff" if ratio < 1.0 else "Original"
            
            if ratio > 2.0:
                winner += " ⚠️"
                problem_k.append((k, ratio, orig_time, cut_time))
            
            print(f"{k:<8} {orig_time:<15.3f} {cut_time:<15.3f} {ratio:<10.2f}x {winner:<15}")
    
    print(f"\n{'Total':<8} {orig.total_solve_time:<15.2f} {cut.total_solve_time:<15.2f} "
          f"{cut.total_solve_time/orig.total_solve_time:<10.2f}x")
    
    # Identify problems
    if problem_k:
        print(f"\n{'='*80}")
        print(f"⚠️  CRITICAL PERFORMANCE ISSUES")
        print(f"{'='*80}")
        
        print(f"\nCutoff solver significantly slower at:")
        for k, ratio, orig_t, cut_t in problem_k:
            print(f"  K={k}: {ratio:.1f}x slower ({orig_t:.3f}s → {cut_t:.3f}s)")
        
        # Analyze why
        print(f"\n{'Root Cause Analysis':-^80}")
        
        # Check if optimal is near UB
        if orig.optimal and orig.optimal >= 0.7 * orig.ub:
            print(f"✓ Optimal/UB ratio high ({orig.optimal}/{orig.ub} = {orig.optimal/orig.ub:.2f})")
            print(f"  → Boundary effect as expected")
        else:
            print(f"✗ Optimal/UB ratio LOW ({orig.optimal}/{orig.ub} = {orig.optimal/orig.ub:.2f})")
            print(f"  → Should favor Cutoff, but doesn't!")
            print(f"  → Likely encoding weakness")
        
        # Check mutual exclusion vs propagation
        mutual_ratio = cut.distance_stats.mutual_exclusion_clauses / cut.distance_stats.total_clauses
        print(f"\nMutual exclusion ratio: {mutual_ratio:.1%}")
        
        if mutual_ratio > 0.5:
            print(f"  ⚠️  Over 50% of clauses are 2-literal mutual exclusions")
            print(f"  → Weak propagation power")
            print(f"  → Solver relies on search rather than propagation")


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_runner.py <mtx_file>")
        print("Example: python debug_runner.py bfw62a.mtx")
        sys.exit(1)
    
    mtx_file = sys.argv[1]
    
    # Find file
    search_paths = [
        mtx_file,
        f"mtx/{mtx_file}",
        f"mtx/oc2/{mtx_file}",
        f"mtx/cutoff2/{mtx_file}",
    ]
    
    found_file = None
    for path in search_paths:
        if os.path.exists(path):
            found_file = path
            break
    
    if not found_file:
        print(f"Error: File '{mtx_file}' not found")
        sys.exit(1)
    
    # Parse graph
    n, edges = parse_mtx_file(found_file)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE DEBUG RUN")
    print(f"{'='*80}")
    print(f"File: {found_file}")
    print(f"Graph: n={n}, |E|={len(edges)}")
    
    # Run Original solver with instrumentation
    print(f"\n{'#'*80}")
    print(f"# PHASE 1: ORIGINAL SOLVER")
    print(f"{'#'*80}")
    
    orig_solver = InstrumentedOriginalSolver(n, 'cadical195')
    orig_solver.set_graph_edges(edges)
    orig_solver.create_position_variables()
    orig_solver.create_distance_variables()
    orig_solver.debug_stats.edges = len(edges)
    
    orig_optimal = orig_solver.solve_bandwidth_optimization()
    
    # Save original report
    orig_solver.debug_stats.save_json(f"{mtx_file}_original_debug.json")
    
    # Run Cutoff solver with instrumentation
    print(f"\n{'#'*80}")
    print(f"# PHASE 2: CUTOFF SOLVER")
    print(f"{'#'*80}")
    
    cut_solver = InstrumentedCutoffSolver(n, 'cadical195')
    cut_solver.set_graph_edges(edges)
    cut_solver.create_position_variables()
    cut_solver.create_distance_variables()
    cut_solver.debug_stats.edges = len(edges)
    
    cut_optimal = cut_solver.solve_bandwidth_optimization()
    
    # Save cutoff report
    cut_solver.debug_stats.save_json(f"{mtx_file}_cutoff_debug.json")
    
    # Compare
    compare_reports(orig_solver.debug_stats, cut_solver.debug_stats)
    
    print(f"\n{'='*80}")
    print(f"DEBUG COMPLETE")
    print(f"{'='*80}")
    print(f"Reports saved:")
    print(f"  - {mtx_file}_original_debug.json")
    print(f"  - {mtx_file}_cutoff_debug.json")


if __name__ == '__main__':
    main()