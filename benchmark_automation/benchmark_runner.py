# benchmark_runner.py
# Automated benchmark runner for 2D Bandwidth Minimization
# Runs multiple MTX files with timeout and exports results to CSV

import os
import sys
import time
import csv
import signal
import threading
from datetime import datetime

# Add parent directory to path to import solvers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from bandwidth_optimization_solver import BandwidthOptimizationSolver
    from rectangular_bandwidth_solver import RectangularBandwidthOptimizationSolver
    from unified_mtx_parser import parse_mtx_unified
    print("Solvers and unified parser imported successfully")
except ImportError as e:
    print(f"Error importing solvers: {e}")
    sys.exit(1)

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

class MTXReader:
    """Enhanced MTX file reader that distinguishes between square and rectangular grids"""
    
    @staticmethod
    def read_mtx(filepath):
        """
        Read MTX file and determine grid type
        Returns: (grid_type, n_rows, n_cols, num_vertices, edges)
        grid_type: 'square' or 'rectangular'
        """
        try:
            # Use unified parser to detect format
            result = parse_mtx_unified(filepath)
            
            if result[0] == 'adjacency':
                # Adjacency matrix format (like cage4)
                format_type, num_vertices, edges = result
                
                # For adjacency matrices, assume square grid (n×n)
                grid_size = num_vertices
                grid_type = 'square'
                n_rows = n_cols = grid_size
                
                print(f"  Detected: ADJACENCY matrix → {grid_size}×{grid_size} SQUARE grid, {num_vertices} vertices, {len(edges)} edges")
                return grid_type, n_rows, n_cols, num_vertices, edges
                
            elif result[0] == 'grid':
                # Grid position format (like Trec5)
                format_type, n_rows, n_cols, num_vertices, edges = result
                
                # Determine grid type
                if n_rows == n_cols:
                    grid_type = 'square'
                    print(f"  Detected: GRID positions → {n_rows}×{n_cols} SQUARE grid, {num_vertices} vertices, {len(edges)} edges")
                else:
                    grid_type = 'rectangular'
                    print(f"  Detected: GRID positions → {n_rows}×{n_cols} RECTANGULAR grid, {num_vertices} vertices, {len(edges)} edges")
                
                return grid_type, n_rows, n_cols, num_vertices, edges
            
            else:
                raise ValueError(f"Unknown format type: {result[0]}")
                
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
            return None, None, None, None, None
    
    @staticmethod
    def _read_legacy_mtx(filepath):
        """
        Legacy MTX reader for old format files
        Assumes square grid (n×n)
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Skip comments
            data_lines = []
            for line in lines:
                if not line.startswith('%') and line.strip():
                    data_lines.append(line.strip())
            
            if not data_lines:
                return None, None, None, None, None
            
            # First line: rows cols entries (old format assumes square)
            header = data_lines[0].split()
            n = int(header[0])  # number of vertices (assumes square)
            edges = []
            
            # Parse edges
            for line in data_lines[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    # Convert to 1-based indexing if needed
                    if u != v:  # Skip self-loops
                        edges.append((u, v))
            
            print(f"  Legacy format: {n}×{n} SQUARE grid, {n} vertices, {len(edges)} edges")
            return 'square', n, n, n, edges
            
        except Exception as e:
            print(f"  Error reading legacy format {filepath}: {e}")
            return None, None, None, None, None

class TimeoutManager:
    """Manages timeout for solver execution"""
    
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.timed_out = False
        
    def timeout_handler(self):
        """Called when timeout occurs"""
        self.timed_out = True
        print(f"  TIMEOUT after {self.timeout_seconds}s")
        
    def run_with_timeout(self, func, *args, **kwargs):
        """Run function with timeout"""
        self.timed_out = False
        result = None
        error = None
        
        def target():
            nonlocal result, error
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                error = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(self.timeout_seconds)
        
        if thread.is_alive():
            self.timed_out = True
            return None, True
        
        if error:
            raise error
            
        return result, False

class BenchmarkRunner:
    """Main benchmark runner class"""
    
    # CONFIGURATION SETTINGS - Edit these values to customize behavior
    DEFAULT_TIMEOUT = 300              # Default timeout in seconds
    DEFAULT_SOLVER = 'glucose42'       # Default SAT solver
    MAX_PROBLEM_SIZE = 20              # Skip problems with n > this value
    MTX_FOLDER = 'mtx'                 # MTX files directory name
    RESULTS_FOLDER = 'results'         # Output directory name
    
    def __init__(self, timeout_seconds=None, solver_type=None, grid_filter=None):
        self.timeout_seconds = timeout_seconds or self.DEFAULT_TIMEOUT
        self.solver_type = solver_type or self.DEFAULT_SOLVER
        self.grid_filter = grid_filter or 'auto'  # 'auto', 'square', 'rectangular'
        self.results = []
        
        # Create results directory
        self.results_dir = os.path.join(os.path.dirname(__file__), self.RESULTS_FOLDER)
        os.makedirs(self.results_dir, exist_ok=True)
        
    def find_mtx_files(self):
        """Find all MTX files in the mtx directory"""
        mtx_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.MTX_FOLDER)
        mtx_files = []
        
        if os.path.exists(mtx_dir):
            for file in os.listdir(mtx_dir):
                if file.endswith('.mtx'):
                    mtx_files.append(os.path.join(mtx_dir, file))
        
        return sorted(mtx_files)
    
    def solve_single_problem(self, filepath):
        """Solve a single MTX problem using appropriate solver"""
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")
        
        # Read MTX file with enhanced reader
        grid_type, n_rows, n_cols, num_vertices, edges = MTXReader.read_mtx(filepath)
        if grid_type is None:
            return {
                'filename': filename,
                'grid_type': 'unknown',
                'n_rows': 0,
                'n_cols': 0,
                'nodes': 0,
                'edges': 0,
                'optimal_bandwidth': -1,
                'time_seconds': 0,
                'status': 'READ_ERROR',
                'solver': self.solver_type,
                'timeout_limit': self.timeout_seconds
            }
        
        # Apply grid type filter
        if self.grid_filter != 'auto' and self.grid_filter != grid_type:
            print(f"  Skipping {grid_type} grid (filter: {self.grid_filter})")
            return {
                'filename': filename,
                'grid_type': grid_type,
                'n_rows': n_rows,
                'n_cols': n_cols,
                'nodes': num_vertices,
                'edges': len(edges),
                'optimal_bandwidth': -1,
                'time_seconds': 0,
                'status': 'SKIPPED_FILTER',
                'solver': self.solver_type,
                'timeout_limit': self.timeout_seconds
            }
        
        # Skip very large problems
        max_dim = max(n_rows, n_cols)
        if max_dim > self.MAX_PROBLEM_SIZE:
            print(f"  Skipping large problem (max_dim={max_dim} > {self.MAX_PROBLEM_SIZE})")
            return {
                'filename': filename,
                'grid_type': grid_type,
                'n_rows': n_rows,
                'n_cols': n_cols,
                'nodes': num_vertices,
                'edges': len(edges),
                'optimal_bandwidth': -1,
                'time_seconds': 0,
                'status': 'SKIPPED_LARGE',
                'solver': self.solver_type,
                'timeout_limit': self.timeout_seconds
            }
        
        # Create appropriate solver based on grid type
        try:
            if grid_type == 'square':
                print(f"  Using SQUARE grid solver ({n_rows}×{n_cols})")
                solver = BandwidthOptimizationSolver(num_vertices, self.solver_type)
                solver.set_graph_edges(edges)
                solver.create_position_variables()
                solver.create_distance_variables()
                solve_method = solver.solve_bandwidth_optimization
                max_k_bound = min(20, 2 * (num_vertices - 1))
                
            else:  # rectangular
                print(f"  Using RECTANGULAR grid solver ({n_rows}×{n_cols})")
                solver = RectangularBandwidthOptimizationSolver(
                    num_vertices, n_rows, n_cols, self.solver_type
                )
                solver.set_graph_edges(edges)
                solver.create_position_variables()
                solver.create_distance_variables()
                solve_method = solver.solve_bandwidth_optimization
                max_k_bound = min(20, (n_rows - 1) + (n_cols - 1))
                
        except Exception as e:
            print(f"  Solver creation failed: {e}")
            return {
                'filename': filename,
                'grid_type': grid_type,
                'n_rows': n_rows,
                'n_cols': n_cols,
                'nodes': num_vertices,
                'edges': len(edges),
                'optimal_bandwidth': -1,
                'time_seconds': 0,
                'status': 'SOLVER_ERROR',
                'solver': self.solver_type,
                'timeout_limit': self.timeout_seconds
            }
        
        # Solve with timeout
        start_time = time.time()
        timeout_mgr = TimeoutManager(self.timeout_seconds)
        
        try:
            print(f"  Solving with {self.solver_type.upper()}, timeout={self.timeout_seconds}s")
            print(f"  Max K bound: {max_k_bound}")
            
            result, timed_out = timeout_mgr.run_with_timeout(
                solve_method,
                start_k=1,
                end_k=max_k_bound
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if timed_out:
                status = 'TIMEOUT'
                optimal_bw = -1
                print(f"  Result: TIMEOUT after {elapsed_time:.2f}s")
            elif result is None:
                status = 'UNSOLVABLE'
                optimal_bw = -1
                print(f"  Result: UNSOLVABLE in {elapsed_time:.2f}s")
            else:
                status = 'SOLVED'
                optimal_bw = result
                print(f"  Result: Optimal bandwidth = {optimal_bw} in {elapsed_time:.2f}s")
            
            return {
                'filename': filename,
                'grid_type': grid_type,
                'n_rows': n_rows,
                'n_cols': n_cols,
                'nodes': num_vertices,
                'edges': len(edges),
                'optimal_bandwidth': optimal_bw,
                'time_seconds': round(elapsed_time, 2),
                'status': status,
                'solver': self.solver_type,
                'timeout_limit': self.timeout_seconds
            }
            
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"  Exception: {e}")
            return {
                'filename': filename,
                'grid_type': grid_type,
                'n_rows': n_rows,
                'n_cols': n_cols,
                'nodes': num_vertices,
                'edges': len(edges),
                'optimal_bandwidth': -1,
                'time_seconds': round(elapsed_time, 2),
                'status': 'ERROR',
                'solver': self.solver_type,
                'timeout_limit': self.timeout_seconds
            }
    
    def run_benchmark(self):
        """Run benchmark on all MTX files"""
        print("=" * 60)
        print("2D BANDWIDTH MINIMIZATION BENCHMARK")
        print("=" * 60)
        print(f"Solver: {self.solver_type.upper()}")
        print(f"Timeout: {self.timeout_seconds} seconds")
        print(f"Grid filter: {self.grid_filter}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Find MTX files
        mtx_files = self.find_mtx_files()
        if not mtx_files:
            print("No MTX files found in mtx/ directory")
            return
        
        print(f"Found {len(mtx_files)} MTX files")
        
        # Process each file
        for i, filepath in enumerate(mtx_files, 1):
            print(f"\n[{i}/{len(mtx_files)}] " + "=" * 50)
            result = self.solve_single_problem(filepath)
            self.results.append(result)
        
        # Export results
        self.export_to_csv()
        self.print_summary()
    
    def export_to_csv(self):
        """Export results to CSV file with enhanced grid information"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"benchmark_results_{timestamp}.csv"
        csv_path = os.path.join(self.results_dir, csv_filename)
        
        print(f"\nExporting results to: {csv_path}")
        
        fieldnames = [
            'filename', 'grid_type', 'n_rows', 'n_cols', 'nodes', 'edges', 
            'optimal_bandwidth', 'time_seconds', 'status', 'solver', 'timeout_limit'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"CSV export completed: {len(self.results)} records")
        return csv_path
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        total = len(self.results)
        solved = len([r for r in self.results if r['status'] == 'SOLVED'])
        timeout = len([r for r in self.results if r['status'] == 'TIMEOUT'])
        errors = len([r for r in self.results if r['status'] in ['ERROR', 'READ_ERROR', 'SOLVER_ERROR']])
        skipped = len([r for r in self.results if r['status'] == 'SKIPPED_LARGE'])
        filtered = len([r for r in self.results if r['status'] == 'SKIPPED_FILTER'])
        
        # Grid type statistics
        square_count = len([r for r in self.results if r.get('grid_type') == 'square'])
        rect_count = len([r for r in self.results if r.get('grid_type') == 'rectangular'])
        
        print(f"Total problems: {total}")
        print(f"  Square grids (n×n): {square_count}")
        print(f"  Rectangular grids (n×m): {rect_count}")
        print(f"Solved: {solved} ({solved/total*100:.1f}%)")
        print(f"Timeout: {timeout} ({timeout/total*100:.1f}%)")
        print(f"Errors: {errors} ({errors/total*100:.1f}%)")
        print(f"Skipped (large): {skipped} ({skipped/total*100:.1f}%)")
        print(f"Skipped (filter): {filtered} ({filtered/total*100:.1f}%)")
        
        if solved > 0:
            solved_results = [r for r in self.results if r['status'] == 'SOLVED']
            avg_time = sum(r['time_seconds'] for r in solved_results) / len(solved_results)
            avg_bw = sum(r['optimal_bandwidth'] for r in solved_results) / len(solved_results)
            print(f"\nAverage solve time: {avg_time:.2f}s")
            print(f"Average bandwidth: {avg_bw:.1f}")
        
        print("\nDetailed results:")
        print("-" * 90)
        print(f"{'Filename':<15} {'Type':<6} {'Grid':<8} {'Nodes':<5} {'Edges':<5} {'BW':<3} {'Time':<8} {'Status':<12}")
        print("-" * 90)
        
        for result in self.results:
            grid_str = f"{result.get('n_rows', '?')}×{result.get('n_cols', '?')}"
            type_str = result.get('grid_type', 'unk')[:6]
            bw_str = str(result['optimal_bandwidth']) if result['optimal_bandwidth'] >= 0 else '-'
            print(f"{result['filename']:<15} {type_str:<6} {grid_str:<8} {result.get('nodes', 0):<5} "
                  f"{result['edges']:<5} {bw_str:<3} {result['time_seconds']:<8.2f} {result['status']:<12}")
        
        print("=" * 60)

def main():
    """Main function with command line argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description='2D Bandwidth Minimization Benchmark Runner')
    parser.add_argument('--timeout', type=int, default=300, 
                       help='Timeout in seconds per problem (default: 300)')
    parser.add_argument('--solver', choices=['glucose42', 'cadical195'], default='glucose42',
                       help='SAT solver to use (default: glucose42)')
    parser.add_argument('--grid-type', choices=['auto', 'square', 'rectangular'], default='auto',
                       help='Filter by grid type (default: auto - process all)')
    
    args = parser.parse_args()
    
    # Run benchmark
    runner = BenchmarkRunner(
        timeout_seconds=args.timeout,
        solver_type=args.solver,
        grid_filter=args.grid_type
    )
    
    try:
        runner.run_benchmark()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")

if __name__ == '__main__':
    main()
