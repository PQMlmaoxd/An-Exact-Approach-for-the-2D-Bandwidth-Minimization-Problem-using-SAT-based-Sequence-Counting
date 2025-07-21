# run_mtx_with_kissat.py
# Main script to run MTX benchmark files with Kissat SAT solver

import sys
import time
import os

# Import our modules
from mtx_parser import MTXParser
from kissat_solver import KissatSolver
from bandwidth_optimization_solver import BandwidthOptimizationSolver

class MTXBandwidthSolver:
    """
    Main class for solving bandwidth optimization on MTX files using Kissat
    """
    
    def __init__(self, mtx_file, kissat_path="./bin/kissat"):
        """
        Initialize MTX bandwidth solver
        
        Args:
            mtx_file: Path to MTX file
            kissat_path: Path to kissat executable
        """
        self.mtx_file = mtx_file
        self.kissat_path = kissat_path
        self.parser = None
        self.graph_data = None
        self.solver = None
        
    def load_graph(self):
        """
        Load graph from MTX file
        """
        print(f"=== LOADING GRAPH FROM MTX FILE ===")
        print(f"File: {self.mtx_file}")
        
        if not os.path.exists(self.mtx_file):
            raise FileNotFoundError(f"MTX file not found: {self.mtx_file}")
        
        # Parse MTX file
        self.parser = MTXParser(self.mtx_file)
        self.graph_data = self.parser.parse_mtx_file()
        
        # Display graph statistics
        stats = self.parser.get_graph_statistics()
        print(f"\n📊 GRAPH STATISTICS:")
        print(f"  Nodes: {stats['num_nodes']}")
        print(f"  Edges: {stats['num_edges']}")
        print(f"  Min degree: {stats['min_degree']}")
        print(f"  Max degree: {stats['max_degree']}")
        print(f"  Avg degree: {stats['avg_degree']:.2f}")
        print(f"  Density: {stats['density']:.6f}")
        
        return self.graph_data
    
    def estimate_difficulty(self):
        """
        Estimate problem difficulty and suggest solving strategy
        """
        if not self.graph_data:
            return None
        
        n = self.graph_data['num_nodes']
        m = self.graph_data['num_edges']
        stats = self.parser.get_graph_statistics()
        
        # Rough complexity estimates
        theoretical_ub = 2 * (n - 1)
        estimated_vars = n * n * 2 + m * n * 2  # Position vars + distance vars
        estimated_clauses = n * n + m * n  # Rough estimate
        
        print(f"\n🧠 COMPLEXITY ESTIMATION:")
        print(f"  Theoretical UB: {theoretical_ub}")
        print(f"  Estimated variables: ~{estimated_vars:,}")
        print(f"  Estimated clauses: ~{estimated_clauses:,}")
        
        # Difficulty assessment
        if n <= 50:
            difficulty = "EASY"
        elif n <= 200:
            difficulty = "MEDIUM"
        elif n <= 500:
            difficulty = "HARD"
        else:
            difficulty = "VERY HARD"
        
        print(f"  Difficulty: {difficulty}")
        
        # Recommend strategy
        if n <= 100:
            strategy = "Full optimization (K=1 to UB)"
            time_limit = 300  # 5 minutes
        elif n <= 300:
            strategy = "Limited search (K=UB-10 to UB)"
            time_limit = 600  # 10 minutes
        else:
            strategy = "UB verification only"
            time_limit = 1200  # 20 minutes
        
        print(f"  Recommended strategy: {strategy}")
        print(f"  Suggested time limit: {time_limit}s")
        
        return {
            'difficulty': difficulty,
            'strategy': strategy,
            'time_limit': time_limit,
            'theoretical_ub': theoretical_ub,
            'estimated_vars': estimated_vars,
            'estimated_clauses': estimated_clauses
        }
    
    def solve_with_kissat(self, max_k=None, min_k=1, time_limit=300):
        """
        Solve bandwidth optimization using Kissat
        
        Args:
            max_k: Maximum K to test (default: use UB from random search)
            min_k: Minimum K to test
            time_limit: Time limit per SAT call
        """
        if not self.graph_data:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")
        
        n = self.graph_data['num_nodes']
        edges = self.graph_data['edges']
        
        print(f"\n=== BANDWIDTH OPTIMIZATION WITH KISSAT ===")
        print(f"Solver: Kissat SAT")
        print(f"Graph: {n} nodes, {len(edges)} edges")
        
        # Create bandwidth solver
        self.solver = BandwidthOptimizationSolver(n, solver_type='kissat')
        self.solver.set_graph_edges(edges)
        self.solver.create_position_variables()
        self.solver.create_distance_variables()
        
        # Step 1: Find UB with random assignment
        print(f"\n🔍 STEP 1: Finding Upper Bound")
        ub_result = self.solver.step1_find_ub_hybrid(random_iterations=500, greedy_tries=5)
        ub = ub_result['ub']
        
        if max_k is None:
            max_k = ub
        
        print(f"\n🎯 UB found: {ub}")
        print(f"Testing K range: [{min_k}, {max_k}]")
        
        # Step 2: SAT-based optimization with Kissat
        print(f"\n🔧 STEP 2: SAT Optimization with Kissat")
        
        optimal_k = max_k
        sat_calls = 0
        total_sat_time = 0
        
        # Test from max_k down to min_k
        for K in range(max_k, min_k - 1, -1):
            print(f"\n--- Testing K = {K} ---")
            
            # Test with Kissat
            start_time = time.time()
            result = self.test_k_with_kissat(K, time_limit)
            solve_time = time.time() - start_time
            
            sat_calls += 1
            total_sat_time += solve_time
            
            if result:
                optimal_k = K
                print(f"✅ K = {K} is FEASIBLE (time: {solve_time:.2f}s)")
                # Continue to find smaller K
            else:
                print(f"❌ K = {K} is INFEASIBLE (time: {solve_time:.2f}s)")
                # Stop search - optimal found
                print(f"🎯 OPTIMAL BANDWIDTH = {optimal_k}")
                break
        
        # Results summary
        print(f"\n" + "="*60)
        print(f"BANDWIDTH OPTIMIZATION RESULTS")
        print(f"="*60)
        print(f"Graph file: {self.mtx_file}")
        print(f"Nodes: {n}, Edges: {len(edges)}")
        print(f"Upper bound (random): {ub}")
        print(f"Optimal bandwidth: {optimal_k}")
        print(f"Improvement: {ub - optimal_k} ({((ub - optimal_k) / ub * 100):.1f}%)")
        print(f"SAT calls: {sat_calls}")
        print(f"Total SAT time: {total_sat_time:.2f}s")
        print(f"Average time per call: {total_sat_time / sat_calls:.2f}s")
        print(f"="*60)
        
        return optimal_k
    
    def test_k_with_kissat(self, K, time_limit=300):
        """
        Test if bandwidth K is feasible using Kissat
        
        Args:
            K: Bandwidth to test
            time_limit: Time limit in seconds
            
        Returns:
            bool: True if K is feasible, False if not
        """
        # Create Kissat solver
        kissat = KissatSolver(self.kissat_path)
        
        try:
            # Encode constraints
            print(f"  Encoding constraints for K={K}...")
            
            # Position constraints
            pos_clauses = self.solver.encode_position_constraints()
            print(f"  Position clauses: {len(pos_clauses)}")
            
            # Distance constraints
            dist_clauses = self.solver.encode_distance_constraints()
            print(f"  Distance clauses: {len(dist_clauses)}")
            
            # Bandwidth constraints
            bw_clauses = self.solver.encode_advanced_bandwidth_constraint(K)
            print(f"  Bandwidth clauses: {len(bw_clauses)}")
            
            # Add all clauses to Kissat
            all_clauses = pos_clauses + dist_clauses + bw_clauses
            print(f"  Total clauses: {len(all_clauses)}")
            
            for clause in all_clauses:
                kissat.add_clause(clause)
            
            # Solve with Kissat
            print(f"  Running Kissat (timeout: {time_limit}s)...")
            result = kissat.solve(timeout=time_limit)
            
            return result
            
        finally:
            kissat.delete()

def main():
    """
    Main function
    """
    if len(sys.argv) < 2:
        print("Usage: python3 run_mtx_with_kissat.py <mtx_file> [time_limit]")
        print("Example: python3 run_mtx_with_kissat.py mtx/1138_bus.mtx 600")
        sys.exit(1)
    
    mtx_file = sys.argv[1]
    time_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    
    print(f"=== MTX BANDWIDTH OPTIMIZATION WITH KISSAT ===")
    print(f"MTX file: {mtx_file}")
    print(f"Time limit per SAT call: {time_limit}s")
    print(f"Kissat version: 4.0.3")
    
    try:
        # Create solver
        solver = MTXBandwidthSolver(mtx_file)
        
        # Load graph
        graph_data = solver.load_graph()
        
        # Estimate difficulty
        difficulty_info = solver.estimate_difficulty()
        
        # Ask user confirmation for large instances
        if difficulty_info['difficulty'] in ['HARD', 'VERY HARD']:
            print(f"\n⚠️  WARNING: This is a {difficulty_info['difficulty']} instance!")
            print(f"   Estimated variables: {difficulty_info['estimated_vars']:,}")
            print(f"   Estimated clauses: {difficulty_info['estimated_clauses']:,}")
            
            response = input("\nContinue? (y/N): ").strip().lower()
            if response != 'y' and response != 'yes':
                print("Aborted by user.")
                sys.exit(0)
        
        # Solve
        optimal_k = solver.solve_with_kissat(time_limit=time_limit)
        
        print(f"\n🎉 SOLVED! Optimal bandwidth = {optimal_k}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
