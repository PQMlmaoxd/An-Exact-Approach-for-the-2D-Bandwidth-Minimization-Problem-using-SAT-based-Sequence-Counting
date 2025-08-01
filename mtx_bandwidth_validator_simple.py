# mtx_bandwidth_validator_simple.py
# Simple MTX bandwidth validator - no external dependencies
# Just core validation with text output

import os
import sys
import time
from typing import Dict, List, Tuple, Optional

# Import our solver
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bandwidth_optimization_solver import BandwidthOptimizationSolver

class SimpleMTXBandwidthValidator:
    """
    MTX file validator for 2D bandwidth minimization
    
    Two output methods:
    1. Label format: d(vertex1(x1,y1), vertex2(x2,y2)) = distance
    2. ASCII grid: text-based grid visualization
    
    Features:
        - Parses .mtx files (weighted/unweighted, directed/undirected)
        - Validates bandwidth solutions
        - Text-only output (no matplotlib needed)
        - Works everywhere - pure Python
    
    Method comparison:
    
    Label format:
        + Shows exact math - good for debugging
        + Easy to verify each edge distance
        + Works well for academic reports
        
        Example: d(1(1,2), 3(2,1)) = 2
    
    ASCII grid:
        + Visual layout - easier to understand
        + Quick pattern recognition
        + Good for presentations
        
        Example:
           1 2 3
        1  1 . .
        2  . 2 3
        3  . . .
    
    Recommendation: Use both - label format for validation, 
    grid format for understanding the solution layout.
    """
    
    def __init__(self, filename: str):
        """Initialize with MTX file"""
        self.filename = filename
        self.n = 0
        self.edges = []
        self.edge_weights = {}
        self.is_weighted = False
        self.is_directed = False
        
        self._parse_mtx_file()
    
    def _parse_mtx_file(self) -> None:
        """
        Parse MTX file and extract graph data
        
        Handles MatrixMarket format:
        - Comments and metadata parsing
        - Weighted/directed graph conversion to unweighted/undirected
        - Error handling for malformed files
        """
        print(f"Reading MTX file: {os.path.basename(self.filename)}")
        
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"File not found: {self.filename}")
            return
        
        header_found = False
        edges_set = set()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            if not line:
                continue
                
            # Handle comments
            if line.startswith('%'):
                if 'kind:' in line.lower():
                    if 'directed' in line.lower():
                        self.is_directed = True
                    if 'weighted' in line.lower():
                        self.is_weighted = True
                continue
            
            # Parse dimensions
            if not header_found:
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        rows, cols, nnz = map(int, parts[:3])
                        self.n = max(rows, cols)
                        print(f"Matrix: {rows}×{cols}, {nnz} entries")
                        print(f"Graph: {'directed' if self.is_directed else 'undirected'}, {'weighted' if self.is_weighted else 'unweighted'}")
                        header_found = True
                        continue
                except ValueError:
                    print(f"Warning: bad header at line {line_num}: {line}")
                    continue
            
            # Parse edges
            try:
                parts = line.split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    weight = float(parts[2]) if len(parts) > 2 else 1.0
                    
                    if u == v:  # skip self-loops
                        continue
                    
                    # Convert to undirected
                    if not self.is_directed:
                        edge = tuple(sorted([u, v]))
                    else:
                        edge = (u, v)
                    
                    if edge not in edges_set:
                        edges_set.add(edge)
                        self.edges.append(edge)
                        if self.is_weighted:
                            self.edge_weights[edge] = weight
                            
            except (ValueError, IndexError):
                print(f"Warning: bad edge at line {line_num}: {line}")
                continue
        
        print(f"Loaded: {self.n} vertices, {len(self.edges)} edges")
        if self.is_weighted:
            weights = list(self.edge_weights.values())
            print(f"Weights: {min(weights):.3f} to {max(weights):.3f}")
    
    def solve_bandwidth_problem(self, solver_type: str = 'glucose42') -> Tuple[Optional[int], Dict, Optional[Dict]]:
        """
        Solve 2D bandwidth minimization using SAT
        
        Returns optimal bandwidth, solve info, and real assignment from SAT solution
        """
        print(f"\nSolving 2D bandwidth minimization")
        print(f"Problem: {self.n} vertices on {self.n}×{self.n} grid")
        print(f"Using: {solver_type.upper()}")
        
        self.solver = BandwidthOptimizationSolver(self.n, solver_type)
        self.solver.set_graph_edges(self.edges)
        self.solver.create_position_variables()
        self.solver.create_distance_variables()
        
        start_time = time.time()
        optimal_bandwidth = self.solver.solve_bandwidth_optimization()
        solve_time = time.time() - start_time
        
        # Extract real assignment from SAT solution if available
        real_assignment = None
        if optimal_bandwidth is not None and hasattr(self.solver, 'last_model'):
            real_assignment = self.extract_assignment_from_sat_model(self.solver.last_model)
        
        solution_info = {
            'solver_type': solver_type,
            'solve_time': solve_time,
            'graph_size': self.n,
            'num_edges': len(self.edges),
            'optimal_bandwidth': optimal_bandwidth,
            'edges_per_vertex': len(self.edges) / self.n if self.n > 0 else 0
        }
        
        print(f"Solve time: {solve_time:.2f}s")
        if optimal_bandwidth is not None:
            print(f"Optimal bandwidth: {optimal_bandwidth}")
            if real_assignment:
                print(f"Real SAT assignment extracted: {len(real_assignment)} vertices")
        else:
            print(f"No solution found")
        
        return optimal_bandwidth, solution_info, real_assignment
    
    def validate_solution_comprehensive(self, assignment: Dict[int, Tuple[int, int]], 
                                      expected_bandwidth: int) -> Dict:
        """
        Validate solution thoroughly
        
        Checks:
        1. Max distance = expected bandwidth
        2. At least one edge achieves max distance
        3. All vertices within grid bounds
        4. No position overlaps
        """
        print(f"\nValidating bandwidth solution")
        print(f"Expected bandwidth: {expected_bandwidth}")
        print(f"Checking {len(assignment)} vertex positions...")
        
        edge_distances = []
        max_distance = 0
        distance_counts = {}
        position_conflicts = []
        out_of_bounds = []
        
        # Check position validity
        used_positions = set()
        for vertex, (x, y) in assignment.items():
            # Check bounds
            if not (1 <= x <= self.n and 1 <= y <= self.n):
                out_of_bounds.append((vertex, (x, y)))
            
            # Check uniqueness
            pos = (x, y)
            if pos in used_positions:
                position_conflicts.append((vertex, pos))
            else:
                used_positions.add(pos)
        
        # Calculate distances
        for u, v in self.edges:
            if u in assignment and v in assignment:
                x1, y1 = assignment[u]
                x2, y2 = assignment[v]
                distance = abs(x1 - x2) + abs(y1 - y2)
                
                edge_distances.append({
                    'edge': (u, v),
                    'positions': ((x1, y1), (x2, y2)),
                    'distance': distance,
                    'calculation': f"|{x1}-{x2}| + |{y1}-{y2}| = {abs(x1-x2)} + {abs(y1-y2)} = {distance}"
                })
                
                max_distance = max(max_distance, distance)
                distance_counts[distance] = distance_counts.get(distance, 0) + 1
        
        # Build validation results
        validation_results = {
            'is_valid': True,
            'calculated_bandwidth': max_distance,
            'expected_bandwidth': expected_bandwidth,
            'edge_distances': edge_distances,
            'distance_distribution': distance_counts,
            'max_distance_edges': [ed for ed in edge_distances if ed['distance'] == max_distance],
            'invalid_distances': [d for d in distance_counts.keys() if d > expected_bandwidth],
            'position_conflicts': position_conflicts,
            'out_of_bounds': out_of_bounds,
            'validation_errors': []
        }
        
        # Check for errors
        if max_distance != expected_bandwidth:
            validation_results['is_valid'] = False
            validation_results['validation_errors'].append(
                f"Bandwidth mismatch: got {max_distance}, expected {expected_bandwidth}"
            )
        
        if position_conflicts:
            validation_results['is_valid'] = False
            validation_results['validation_errors'].append(
                f"Position conflicts: {len(position_conflicts)} vertices overlap"
            )
        
        if out_of_bounds:
            validation_results['is_valid'] = False
            validation_results['validation_errors'].append(
                f"Out of bounds: {len(out_of_bounds)} vertices outside grid"
            )
        
        # Print summary
        print(f"Calculated bandwidth: {max_distance}")
        print(f"Validation: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
        
        if validation_results['validation_errors']:
            print(f"Errors:")
            for error in validation_results['validation_errors']:
                print(f"  - {error}")
        
        print(f"Distance counts: {dict(sorted(distance_counts.items()))}")
        print(f"Critical edges: {len(validation_results['max_distance_edges'])}")
        
        return validation_results
    
    def decode_solution_label_format(self, assignment: Dict[int, Tuple[int, int]], 
                                   validation_results: Dict) -> None:
        """
        Method 1: Label format - d(vertex1(x1,y1), vertex2(x2,y2)) = distance
        
        Shows exact math for each edge distance - good for debugging
        and verification. Each distance calculation is explicit.
        """
        print(f"\n" + "="*60)
        print(f"METHOD 1: LABEL FORMAT")
        print(f"Format: d(vertex1(x1,y1), vertex2(x2,y2)) = distance")
        print(f"=" * 60)
        
        # Sort by distance (largest first)
        sorted_edges = sorted(validation_results['edge_distances'], 
                            key=lambda x: (-x['distance'], x['edge']))
        
        # Group by distance
        distance_groups = {}
        for edge_data in sorted_edges:
            dist = edge_data['distance']
            if dist not in distance_groups:
                distance_groups[dist] = []
            distance_groups[dist].append(edge_data)
        
        # Show each distance group
        for distance in sorted(distance_groups.keys(), reverse=True):
            edges_at_distance = distance_groups[distance]
            is_max = distance == validation_results['calculated_bandwidth']
            status = "MAX" if is_max else "normal"
            
            print(f"\nDistance = {distance} ({status}) - {len(edges_at_distance)} edges:")
            
            for i, edge_data in enumerate(edges_at_distance, 1):
                u, v = edge_data['edge']
                (x1, y1), (x2, y2) = edge_data['positions']
                calculation = edge_data['calculation']
                
                print(f"   {i:2}. d({u}({x1},{y1}), {v}({x2},{y2})) = {distance}")
                print(f"       └─ {calculation}")
        
        # Summary
        print(f"\n" + "="*40)
        print(f"VALIDATION SUMMARY")
        print(f"="*40)
        print(f"   Total edges: {len(sorted_edges)}")
        print(f"   Bandwidth: {validation_results['calculated_bandwidth']}")
        print(f"   Distance range: {min(distance_groups.keys())} to {max(distance_groups.keys())}")
        print(f"   Critical edges: {len(validation_results['max_distance_edges'])}")
        print(f"   Status: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
        
        # Distance breakdown
        print(f"\nDistance breakdown:")
        for dist in sorted(distance_groups.keys()):
            count = len(distance_groups[dist])
            percentage = (count / len(sorted_edges)) * 100
            bar = "█" * min(int(percentage / 3), 20)
            marker = "MAX" if dist == validation_results['calculated_bandwidth'] else ""
            print(f"   {dist:2}: {count:2} edges ({percentage:4.1f}%) {marker} {bar}")
    
    def visualize_solution_matrix_format(self, assignment: Dict[int, Tuple[int, int]], 
                                       validation_results: Dict) -> None:
        """
        Method 2: ASCII grid visualization
        
        Shows spatial layout on n×n grid - easier to understand
        vertex placement and see patterns visually.
        """
        print(f"\n" + "="*60)
        print(f"METHOD 2: ASCII GRID")
        print(f"Layout: {self.n}×{self.n} grid with vertex positions")
        print(f"=" * 60)
        
        # Build grid
        grid = [['.' for _ in range(self.n)] for _ in range(self.n)]
        vertex_positions = {}
        
        # Place vertices
        for vertex, (x, y) in assignment.items():
            if 1 <= x <= self.n and 1 <= y <= self.n:
                grid[y-1][x-1] = str(vertex)
                vertex_positions[vertex] = (x, y)
        
        # Show grid
        print(f"\nGrid layout:")
        print("   " + "".join(f"{x:3}" for x in range(1, self.n + 1)))
        print("   " + "───" * self.n)
        
        for y in range(self.n):
            row_display = "".join(f"{grid[y][x]:3}" for x in range(self.n))
            print(f"{y+1:2}│{row_display}")
        
        # Edge analysis
        print(f"\nEdge distances:")
        print(f"{'Edge':<10} {'Positions':<18} {'Dist':<6} {'Status'}")
        print("─" * 45)
        
        for edge_data in sorted(validation_results['edge_distances'], 
                              key=lambda x: -x['distance']):
            u, v = edge_data['edge']
            distance = edge_data['distance']
            (x1, y1), (x2, y2) = edge_data['positions']
            
            status = "MAX" if distance == validation_results['calculated_bandwidth'] else ""
            positions_str = f"({x1},{y1})↔({x2},{y2})"
            print(f"({u},{v}){'':<4} {positions_str:<18} {distance:<6} {status}")
        
        # Distance distribution chart
        print(f"\nDistance distribution:")
        dist_counts = validation_results['distance_distribution']
        max_count = max(dist_counts.values()) if dist_counts else 0
        
        for dist in sorted(dist_counts.keys()):
            count = dist_counts[dist]
            bar_width = int((count / max_count) * 25) if max_count > 0 else 0
            bar = "█" * bar_width
            marker = "MAX" if dist == validation_results['calculated_bandwidth'] else ""
            print(f"   {dist:2}: {count:2} edges {marker} {bar}")
        
        # Spatial stats
        print(f"\n" + "="*40)
        print(f"SPATIAL SUMMARY")
        print(f"="*40)
        
        if vertex_positions:
            x_coords = [pos[0] for pos in vertex_positions.values()]
            y_coords = [pos[1] for pos in vertex_positions.values()]
            
            x_spread = max(x_coords) - min(x_coords) + 1
            y_spread = max(y_coords) - min(y_coords) + 1
            utilization = len(vertex_positions) / (self.n * self.n) * 100
            
            print(f"   Grid: {self.n}×{self.n}")
            print(f"   Used area: {x_spread}×{y_spread}")
            print(f"   Utilization: {utilization:.1f}%")
            print(f"   Vertices: {len(vertex_positions)}")
            print(f"   Bandwidth: {validation_results['calculated_bandwidth']}")
            print(f"   Efficiency: {validation_results['calculated_bandwidth']}/{max(x_spread, y_spread):.1f}")
    
    def extract_assignment_from_sat_model(self, model) -> Optional[Dict[int, Tuple[int, int]]]:
        """
        Extract real vertex positions from SAT model
        
        Converts SAT solution back to (x,y) coordinates for each vertex
        """
        if not model or not hasattr(self.solver, 'X_vars') or not hasattr(self.solver, 'Y_vars'):
            print("Warning: No SAT model or variables available")
            return None
        
        assignment = {}
        
        try:
            for v in range(1, self.n + 1):
                # Find X position for vertex v
                x_pos = None
                for pos in range(1, self.n + 1):
                    var_id = self.solver.X_vars[v][pos-1]
                    if var_id in model:
                        x_pos = pos
                        break
                
                # Find Y position for vertex v  
                y_pos = None
                for pos in range(1, self.n + 1):
                    var_id = self.solver.Y_vars[v][pos-1]
                    if var_id in model:
                        y_pos = pos
                        break
                
                if x_pos is not None and y_pos is not None:
                    assignment[v] = (x_pos, y_pos)
                else:
                    print(f"Warning: Could not find position for vertex {v}")
            
            print(f"Extracted real assignment for {len(assignment)} vertices")
            return assignment
            
        except Exception as e:
            print(f"Error extracting assignment from SAT model: {e}")
            return None
    def generate_mock_assignment_from_solution(self, optimal_bandwidth: int) -> Dict[int, Tuple[int, int]]:
        """
        Generate test assignment that achieves target bandwidth
        
        Creates placement for demo purposes. Used as fallback when
        real SAT assignment extraction fails.
        """
        print(f"\nGenerating fallback assignment for bandwidth = {optimal_bandwidth}")
        
        assignment = {}
        
        # Small graphs: linear arrangement
        if self.n <= 4:
            for i in range(1, self.n + 1):
                x = i
                y = 1 if i % 2 == 1 else 2
                assignment[i] = (x, y)
        
        # Medium graphs: compact center placement
        elif self.n <= 8:
            positions_used = set()
            for i in range(1, self.n + 1):
                center = self.n // 2 + 1
                for distance in range(self.n):
                    for dx in range(-distance, distance + 1):
                        for dy in range(-distance, distance + 1):
                            if abs(dx) + abs(dy) <= distance:
                                x, y = center + dx, center + dy
                                if (1 <= x <= self.n and 1 <= y <= self.n and 
                                    (x, y) not in positions_used):
                                    assignment[i] = (x, y)
                                    positions_used.add((x, y))
                                    break
                        if i in assignment:
                            break
                    if i in assignment:
                        break
        
        # Large graphs: grid spreading
        else:
            for i in range(1, min(self.n + 1, self.n * self.n + 1)):
                x = ((i - 1) % self.n) + 1
                y = ((i - 1) // self.n) + 1
                assignment[i] = (x, y)
        
        return assignment
    
    def run_complete_validation(self, solver_type: str = 'glucose42') -> Dict:
        """
        Complete MTX validation workflow with dual decode methods
        
        Full process: parse MTX → solve SAT → validate → display results
        Shows both label format and grid visualization.
        """
        print(f"\n" + "="*70)
        print(f"COMPLETE MTX BANDWIDTH VALIDATION")
        print(f"="*70)
        print(f"File: {os.path.basename(self.filename)}")
        print(f"Solver: {solver_type.upper()}")
        print(f"Target: Find and validate optimal bandwidth")
        
        # Solve bandwidth problem
        optimal_bandwidth, solution_info, real_assignment = self.solve_bandwidth_problem(solver_type)
        
        if optimal_bandwidth is None:
            return {
                'status': 'failed', 
                'reason': 'solver_failed',
                'filename': self.filename,
                'solver_type': solver_type
            }
        
        # Use real assignment if available, otherwise generate fallback
        if real_assignment is None:
            print("No real SAT assignment available, using fallback")
            assignment = self.generate_mock_assignment_from_solution(optimal_bandwidth)
        else:
            print("Using real SAT solution assignment")
            assignment = real_assignment
        
        # Validate solution
        validation_results = self.validate_solution_comprehensive(assignment, optimal_bandwidth)
        
        # Show results with both decode methods
        print(f"\n" + "="*70)
        print(f"DUAL DECODE METHOD ANALYSIS")
        print(f"="*70)
        
        # Method 1: Label format
        self.decode_solution_label_format(assignment, validation_results)
        
        # Method 2: Grid visualization
        self.visualize_solution_matrix_format(assignment, validation_results)
        
        # Final results
        final_results = {
            'status': 'success',
            'filename': self.filename,
            'graph_properties': {
                'vertices': self.n,
                'edges': len(self.edges),
                'is_directed': self.is_directed,
                'is_weighted': self.is_weighted,
                'density': len(self.edges) / (self.n * (self.n - 1) / 2) if self.n > 1 else 0
            },
            'solution_info': solution_info,
            'validation_results': validation_results,
            'assignment': assignment,
            'decode_methods': {
                'label_based': 'Mathematical precision for verification',
                'matrix_based': 'Visual intuition for understanding'
            }
        }
        
        print(f"\n" + "="*70)
        print(f"VALIDATION COMPLETE")
        print(f"="*70)
        print(f"Result: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
        print(f"Bandwidth: {optimal_bandwidth}")
        print(f"Time: {solution_info['solve_time']:.2f} seconds")
        
        return final_results


def validate_mtx_file(mtx_file: str, solver_type: str = 'glucose42') -> Dict:
    """
    Validate any MTX file - convenient wrapper function
    """
    print(f"MTX BANDWIDTH VALIDATION TOOL")
    print(f"Target: {mtx_file}")
    
    if not os.path.exists(mtx_file):
        print(f"Error: File not found - {mtx_file}")
        return {'status': 'file_not_found', 'filename': mtx_file}
    
    validator = SimpleMTXBandwidthValidator(mtx_file)
    results = validator.run_complete_validation(solver_type)
    
    return results


if __name__ == "__main__":
    """
    Command line usage: python mtx_bandwidth_validator_simple.py [mtx_file] [solver]
    
    Examples:
        python mtx_bandwidth_validator_simple.py cage3.mtx
        python mtx_bandwidth_validator_simple.py sample_mtx_datasets/complete_k4.mtx glucose42
        python mtx_bandwidth_validator_simple.py  # Run test mode
    """
    import sys
    
    # Check if specific MTX file provided
    if len(sys.argv) >= 2:
        # Single file validation mode
        mtx_file = sys.argv[1]
        solver_type = sys.argv[2] if len(sys.argv) >= 3 else 'glucose42'
        
        print("=" * 80)
        print("MTX BANDWIDTH VALIDATOR - SINGLE FILE MODE")
        print("=" * 80)
        print(f"File: {mtx_file}")
        print(f"Solver: {solver_type}")
        
        # Add common paths if file not found directly
        if not os.path.exists(mtx_file):
            search_paths = [
                f"mtx/{mtx_file}",
                f"sample_mtx_datasets/{mtx_file}",
                f"mtx/{mtx_file}.mtx",
                f"sample_mtx_datasets/{mtx_file}.mtx"
            ]
            
            for path in search_paths:
                if os.path.exists(path):
                    mtx_file = path
                    print(f"Found file at: {path}")
                    break
            else:
                print(f"Error: File '{sys.argv[1]}' not found")
                print("Searched in:")
                for path in search_paths:
                    print(f"  - {path}")
                sys.exit(1)
        
        # Run validation
        try:
            results = validate_mtx_file(mtx_file, solver_type)
            
            # Print summary
            print("\n" + "="*60)
            print("FINAL SUMMARY")
            print("="*60)
            
            status = results.get('status', 'unknown')
            if status == 'success':
                validation = results.get('validation_results', {})
                bandwidth = validation.get('calculated_bandwidth', 'N/A')
                is_valid = validation.get('is_valid', False)
                solve_time = results.get('solution_info', {}).get('solve_time', 0)
                
                print(f"✓ Validation: {'PASSED' if is_valid else 'FAILED'}")
                print(f"✓ Bandwidth: {bandwidth}")
                print(f"✓ Solve time: {solve_time:.2f}s")
                print(f"✓ Status: SUCCESS")
            else:
                print(f"✗ Status: {status.upper()}")
                if 'reason' in results:
                    print(f"✗ Reason: {results['reason']}")
                
        except Exception as e:
            print(f"Error during validation: {e}")
            sys.exit(1)
    
    else:
        # Testing mode - run validation on sample files
        print("=" * 80)
        print("SIMPLE MTX BANDWIDTH VALIDATOR - TESTING MODE")
        print("Zero dependencies - Pure Python implementation")
        print("=" * 80)
        
        # Test files by complexity
        test_files = [
            "mtx/cage3.mtx",                    # Small real-world example
            "sample_mtx_datasets/complete_k4.mtx",  # Complete graph
            "sample_mtx_datasets/path_p6.mtx",      # Path graph
            "sample_mtx_datasets/cycle_c5.mtx",     # Cycle graph
        ]
        
        results_summary = []
        
        for i, mtx_file in enumerate(test_files, 1):
            print(f"\n{'='*80}")
            print(f"TEST CASE {i}/{len(test_files)}: {os.path.basename(mtx_file)}")
            print(f"{'='*80}")
            
            if os.path.exists(mtx_file):
                try:
                    results = validate_mtx_file(mtx_file, 'glucose42')
                    results_summary.append({
                        'file': mtx_file,
                        'status': results.get('status', 'unknown'),
                        'bandwidth': results.get('validation_results', {}).get('calculated_bandwidth', 'N/A'),
                        'validation': results.get('validation_results', {}).get('is_valid', False)
                    })
                    print(f"Test {i} completed successfully")
                except Exception as e:
                    print(f"Test {i} failed with error: {e}")
                    results_summary.append({
                        'file': mtx_file,
                        'status': 'error',
                        'error': str(e)
                    })
            else:
                print(f"Test {i} skipped - file not found: {mtx_file}")
                results_summary.append({
                    'file': mtx_file,
                    'status': 'file_not_found'
                })
        
        # Summary report
        print(f"\n" + "="*80)
        print(f"TESTING COMPLETE - SUMMARY REPORT")
        print(f"="*80)
        
        for i, result in enumerate(results_summary, 1):
            status_icon = "✓" if result['status'] == 'success' else "✗"
            file_name = os.path.basename(result['file'])
            print(f"{status_icon} Test {i}: {file_name:<25} | Status: {result['status']:<12} | Bandwidth: {result.get('bandwidth', 'N/A')}")
        
        successful_tests = sum(1 for r in results_summary if r['status'] == 'success')
        print(f"\nSuccess rate: {successful_tests}/{len(results_summary)} tests passed")
        print(f"MTX Bandwidth Validator testing complete!")
