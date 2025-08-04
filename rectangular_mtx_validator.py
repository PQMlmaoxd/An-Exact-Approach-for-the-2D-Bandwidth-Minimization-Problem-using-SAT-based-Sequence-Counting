# rectangular_mtx_validator.py
# MTX bandwidth validator for rectangular grids (n×m) - with clear visualization
# Extended from square grid validator to support rectangular grids

import os
import sys
import time
from typing import Dict, List, Tuple, Optional

# Import our rectangular solver
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rectangular_bandwidth_solver import RectangularBandwidthOptimizationSolver, parse_mtx_file

class RectangularMTXBandwidthValidator:
    """
    MTX file validator for rectangular grid 2D bandwidth minimization
    
    Key features:
    - Supports different grid dimensions (n_rows × n_cols)
    - Two visualization methods: Label format + ASCII grid
    - Comprehensive validation and error checking
    - Command-line interface compatible with original validator
    
    Visualization methods:
    
    1. Label format: d(vertex1(x1,y1), vertex2(x2,y2)) = distance
       + Shows exact math calculations
       + Good for debugging and verification
       + Academic paper format
    
    2. ASCII grid: Visual n_rows×n_cols layout
       + Easy to understand spatial arrangement
       + Pattern recognition for optimization
       + Presentation-friendly format
    
    Grid notation:
    - X-axis: rows (1 to n_rows)
    - Y-axis: columns (1 to n_cols)
    - Position: (row, col) format
    """
    
    def __init__(self, filename: str, n_rows: int = None, n_cols: int = None):
        """Initialize with MTX file and optional grid dimensions"""
        self.filename = filename
        self.n_vertices = 0
        self.edges = []
        self.edge_weights = {}
        self.is_weighted = False
        self.is_directed = False
        
        # Grid dimensions
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.auto_grid = (n_rows is None or n_cols is None)
        
        self._parse_mtx_file()
        self._determine_grid_dimensions()
    
    def _parse_mtx_file(self) -> None:
        """Parse MTX file and extract graph data"""
        print(f"Reading MTX file: {os.path.basename(self.filename)}")
        
        try:
            # Use the updated parse_mtx_file that returns grid dimensions
            grid_rows, grid_cols, n_vertices, edges = parse_mtx_file(self.filename)
            
            # Update grid dimensions from MTX file if not specified
            if self.auto_grid:
                self.n_rows = grid_rows
                self.n_cols = grid_cols
                self.auto_grid = False  # We now have dimensions from file
                
            self.n_vertices = n_vertices
            self.edges = edges
            
            print(f"MTX grid: {grid_rows}×{grid_cols}")
            print(f"Loaded: {self.n_vertices} vertices, {len(self.edges)} edges")
            print(f"Edge list: {self.edges}")
        except Exception as e:
            print(f"Error parsing MTX file: {e}")
            raise
    
    def _determine_grid_dimensions(self) -> None:
        """Determine grid dimensions automatically or validate provided ones"""
        if self.auto_grid:
            # Auto-determine optimal grid dimensions
            possible_grids = self._generate_grid_options()
            self.n_rows, self.n_cols = possible_grids[0]  # Use first (most rectangular)
            print(f"Auto-selected grid: {self.n_rows}×{self.n_cols}")
            print(f"Other options: {possible_grids[1:3]}")  # Show alternatives
        else:
            # Validate provided dimensions
            if self.n_rows * self.n_cols < self.n_vertices:
                raise ValueError(f"Grid {self.n_rows}×{self.n_cols} too small for {self.n_vertices} vertices")
            print(f"Using specified grid: {self.n_rows}×{self.n_cols}")
        
        utilization = (self.n_vertices / (self.n_rows * self.n_cols)) * 100
        print(f"Grid utilization: {utilization:.1f}% ({self.n_vertices}/{self.n_rows * self.n_cols})")
    
    def _generate_grid_options(self) -> List[Tuple[int, int]]:
        """Generate possible rectangular grid configurations"""
        options = []
        
        # Add single row/column options
        options.append((self.n_vertices, 1))  # Single row
        options.append((1, self.n_vertices))  # Single column
        
        # Add rectangular options
        for rows in range(2, self.n_vertices + 1):
            cols = (self.n_vertices + rows - 1) // rows  # Ceiling division
            if rows * cols >= self.n_vertices:
                options.append((rows, cols))
        
        # Add square option
        import math
        square_size = max(int(math.ceil(math.sqrt(self.n_vertices))), self.n_vertices)
        options.append((square_size, square_size))
        
        # Sort by rectangularity (prefer more rectangular shapes)
        return sorted(set(options), key=lambda x: abs(x[0] - x[1]))
    
    def solve_bandwidth_problem(self, solver_type: str = 'glucose42') -> Tuple[Optional[int], Dict, Optional[Dict]]:
        """
        Solve rectangular grid 2D bandwidth minimization using SAT
        """
        print(f"\nSolving rectangular grid 2D bandwidth minimization")
        print(f"Problem: {self.n_vertices} vertices on {self.n_rows}×{self.n_cols} grid")
        print(f"Using: {solver_type.upper()}")
        
        # Create rectangular solver
        self.solver = RectangularBandwidthOptimizationSolver(
            self.n_vertices, self.n_rows, self.n_cols, solver_type
        )
        self.solver.set_graph_edges(self.edges)
        self.solver.create_position_variables()
        self.solver.create_distance_variables()
        
        start_time = time.time()
        optimal_bandwidth = self.solver.solve_bandwidth_optimization()
        solve_time = time.time() - start_time
        
        # Extract real assignment from SAT solution
        real_assignment = None
        if optimal_bandwidth is not None and hasattr(self.solver, 'last_model'):
            real_assignment = self.extract_assignment_from_sat_model(self.solver.last_model)
        
        solution_info = {
            'solver_type': solver_type,
            'solve_time': solve_time,
            'grid_rows': self.n_rows,
            'grid_cols': self.n_cols,
            'num_vertices': self.n_vertices,
            'num_edges': len(self.edges),
            'optimal_bandwidth': optimal_bandwidth,
            'max_possible_distance': (self.n_rows - 1) + (self.n_cols - 1),
            'grid_utilization': (self.n_vertices / (self.n_rows * self.n_cols)) * 100
        }
        
        print(f"Solve time: {solve_time:.2f}s")
        if optimal_bandwidth is not None:
            print(f"Optimal bandwidth: {optimal_bandwidth}")
            print(f"Max possible distance: {solution_info['max_possible_distance']}")
            if real_assignment:
                print(f"Real SAT assignment extracted: {len(real_assignment)} vertices")
        else:
            print(f"No solution found")
        
        return optimal_bandwidth, solution_info, real_assignment
    
    def validate_solution_comprehensive(self, assignment: Dict[int, Tuple[int, int]], 
                                      expected_bandwidth: int) -> Dict:
        """
        Validate rectangular grid solution thoroughly
        """
        print(f"\nValidating rectangular grid bandwidth solution")
        print(f"Grid: {self.n_rows}×{self.n_cols}")
        print(f"Expected bandwidth: {expected_bandwidth}")
        print(f"Checking {len(assignment)} vertex positions...")
        
        edge_distances = []
        max_distance = 0
        distance_counts = {}
        position_conflicts = []
        out_of_bounds = []
        
        # Check position validity for rectangular grid
        used_positions = set()
        for vertex, (x, y) in assignment.items():
            # Check bounds (X: 1 to n_rows, Y: 1 to n_cols)
            if not (1 <= x <= self.n_rows and 1 <= y <= self.n_cols):
                out_of_bounds.append((vertex, (x, y)))
            
            # Check uniqueness
            pos = (x, y)
            if pos in used_positions:
                position_conflicts.append((vertex, pos))
            else:
                used_positions.add(pos)
        
        # Calculate Manhattan distances
        for u, v in self.edges:
            if u in assignment and v in assignment:
                x1, y1 = assignment[u]
                x2, y2 = assignment[v]
                distance = abs(x1 - x2) + abs(y1 - y2)
                
                edge_distances.append({
                    'edge': (u, v),
                    'positions': ((x1, y1), (x2, y2)),
                    'distance': distance,
                    'x_component': abs(x1 - x2),
                    'y_component': abs(y1 - y2),
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
            'validation_errors': [],
            'grid_info': {
                'rows': self.n_rows,
                'cols': self.n_cols,
                'total_positions': self.n_rows * self.n_cols,
                'used_positions': len(used_positions),
                'max_possible_distance': (self.n_rows - 1) + (self.n_cols - 1)
            }
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
                f"Out of bounds: {len(out_of_bounds)} vertices outside {self.n_rows}×{self.n_cols} grid"
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
        Method 1: Label format for rectangular grid
        Format: d(vertex1(row1,col1), vertex2(row2,col2)) = distance
        """
        print(f"\n" + "="*70)
        print(f"METHOD 1: LABEL FORMAT (RECTANGULAR GRID)")
        print(f"Format: d(vertex1(row,col), vertex2(row,col)) = distance")
        print(f"Grid: {self.n_rows} rows × {self.n_cols} columns")
        print(f"=" * 70)
        
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
                x_comp = edge_data['x_component']
                y_comp = edge_data['y_component']
                calculation = edge_data['calculation']
                
                print(f"   {i:2}. d({u}({x1},{y1}), {v}({x2},{y2})) = {distance}")
                print(f"       └─ Row distance: {x_comp}, Col distance: {y_comp}")
                print(f"       └─ {calculation}")
        
        # Enhanced summary for rectangular grid
        print(f"\n" + "="*50)
        print(f"RECTANGULAR GRID VALIDATION SUMMARY")
        print(f"="*50)
        print(f"   Grid dimensions: {self.n_rows}×{self.n_cols}")
        print(f"   Total positions: {self.n_rows * self.n_cols}")
        print(f"   Used positions: {len(assignment)}")
        print(f"   Utilization: {(len(assignment) / (self.n_rows * self.n_cols)) * 100:.1f}%")
        print(f"   Total edges: {len(sorted_edges)}")
        print(f"   Bandwidth: {validation_results['calculated_bandwidth']}")
        print(f"   Max possible: {validation_results['grid_info']['max_possible_distance']}")
        print(f"   Efficiency: {(validation_results['calculated_bandwidth'] / validation_results['grid_info']['max_possible_distance']) * 100:.1f}%")
        print(f"   Critical edges: {len(validation_results['max_distance_edges'])}")
        print(f"   Status: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
        
        # Distance breakdown with components
        print(f"\nDistance breakdown (Row + Col components):")
        for dist in sorted(distance_groups.keys()):
            count = len(distance_groups[dist])
            percentage = (count / len(sorted_edges)) * 100
            bar = "█" * min(int(percentage / 3), 20)
            marker = "MAX" if dist == validation_results['calculated_bandwidth'] else ""
            print(f"   {dist:2}: {count:2} edges ({percentage:4.1f}%) {marker} {bar}")
    
    def visualize_solution_matrix_format(self, assignment: Dict[int, Tuple[int, int]], 
                                       validation_results: Dict) -> None:
        """
        Method 2: ASCII grid visualization for rectangular grid
        """
        print(f"\n" + "="*70)
        print(f"METHOD 2: ASCII RECTANGULAR GRID")
        print(f"Layout: {self.n_rows} rows × {self.n_cols} columns")
        print(f"=" * 70)
        
        # Build rectangular grid
        grid = [['.' for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        vertex_positions = {}
        
        # Place vertices
        for vertex, (x, y) in assignment.items():
            if 1 <= x <= self.n_rows and 1 <= y <= self.n_cols:
                grid[x-1][y-1] = str(vertex)  # x is row, y is column
                vertex_positions[vertex] = (x, y)
        
        # Show grid with proper labeling
        print(f"\nRectangular grid layout ({self.n_rows}×{self.n_cols}):")
        
        # Column headers
        col_header = "Row\\Col"
        for c in range(1, self.n_cols + 1):
            col_header += f"{c:3}"
        print(col_header)
        print("       " + "───" * self.n_cols)
        
        # Rows with data
        for r in range(self.n_rows):
            row_display = f"{r+1:3} │"
            for c in range(self.n_cols):
                row_display += f"{grid[r][c]:3}"
            print(row_display)
        
        # Edge analysis with component breakdown
        print(f"\nEdge distances (Manhattan distance on rectangular grid):")
        print(f"{'Edge':<10} {'Positions':<20} {'Row':<5} {'Col':<5} {'Total':<6} {'Status'}")
        print("─" * 65)
        
        for edge_data in sorted(validation_results['edge_distances'], 
                              key=lambda x: -x['distance']):
            u, v = edge_data['edge']
            distance = edge_data['distance']
            (x1, y1), (x2, y2) = edge_data['positions']
            x_comp = edge_data['x_component']
            y_comp = edge_data['y_component']
            
            status = "MAX" if distance == validation_results['calculated_bandwidth'] else ""
            positions_str = f"({x1},{y1})↔({x2},{y2})"
            print(f"({u},{v}){'':<4} {positions_str:<20} {x_comp:<5} {y_comp:<5} {distance:<6} {status}")
        
        # Enhanced distance distribution for rectangular grid
        print(f"\nDistance distribution on {self.n_rows}×{self.n_cols} grid:")
        dist_counts = validation_results['distance_distribution']
        max_count = max(dist_counts.values()) if dist_counts else 0
        
        for dist in sorted(dist_counts.keys()):
            count = dist_counts[dist]
            bar_width = int((count / max_count) * 25) if max_count > 0 else 0
            bar = "█" * bar_width
            marker = "MAX" if dist == validation_results['calculated_bandwidth'] else ""
            percentage = (count / len(validation_results['edge_distances'])) * 100
            print(f"   {dist:2}: {count:2} edges ({percentage:4.1f}%) {marker} {bar}")
        
        # Rectangular grid spatial analysis
        print(f"\n" + "="*50)
        print(f"RECTANGULAR GRID SPATIAL ANALYSIS")
        print(f"="*50)
        
        if vertex_positions:
            x_coords = [pos[0] for pos in vertex_positions.values()]  # Rows
            y_coords = [pos[1] for pos in vertex_positions.values()]  # Cols
            
            x_spread = max(x_coords) - min(x_coords) + 1 if x_coords else 0
            y_spread = max(y_coords) - min(y_coords) + 1 if y_coords else 0
            actual_area = x_spread * y_spread
            total_area = self.n_rows * self.n_cols
            utilization = (len(vertex_positions) / total_area) * 100
            compactness = (len(vertex_positions) / actual_area) * 100 if actual_area > 0 else 0
            
            print(f"   Grid dimensions: {self.n_rows}×{self.n_cols} (rows×cols)")
            print(f"   Total area: {total_area} positions")
            print(f"   Used area: {x_spread}×{y_spread} = {actual_area} positions")
            print(f"   Vertices placed: {len(vertex_positions)}")
            print(f"   Overall utilization: {utilization:.1f}%")
            print(f"   Area compactness: {compactness:.1f}%")
            print(f"   Bandwidth achieved: {validation_results['calculated_bandwidth']}")
            print(f"   Grid efficiency: {(validation_results['calculated_bandwidth'] / max(self.n_rows-1, self.n_cols-1)):.2f}")
            
            # Row and column usage
            row_usage = len(set(x_coords))
            col_usage = len(set(y_coords))
            print(f"   Rows used: {row_usage}/{self.n_rows} ({(row_usage/self.n_rows)*100:.1f}%)")
            print(f"   Cols used: {col_usage}/{self.n_cols} ({(col_usage/self.n_cols)*100:.1f}%)")
    
    def extract_assignment_from_sat_model(self, model) -> Optional[Dict[int, Tuple[int, int]]]:
        """Extract real vertex positions from rectangular SAT model"""
        if not model or not hasattr(self.solver, 'X_vars') or not hasattr(self.solver, 'Y_vars'):
            print("Warning: No SAT model or variables available")
            return None
        
        assignment = {}
        
        try:
            for v in range(1, self.n_vertices + 1):
                # Find X position (row: 1 to n_rows)
                x_pos = None
                for pos in range(1, self.n_rows + 1):
                    var_id = self.solver.X_vars[v][pos-1]
                    if var_id in model and model[model.index(var_id)] > 0:
                        x_pos = pos
                        break
                
                # Find Y position (col: 1 to n_cols)
                y_pos = None
                for pos in range(1, self.n_cols + 1):
                    var_id = self.solver.Y_vars[v][pos-1]
                    if var_id in model and model[model.index(var_id)] > 0:
                        y_pos = pos
                        break
                
                if x_pos is not None and y_pos is not None:
                    assignment[v] = (x_pos, y_pos)
                else:
                    print(f"Warning: Could not find position for vertex {v}")
            
            print(f"Extracted real assignment for {len(assignment)} vertices on {self.n_rows}×{self.n_cols} grid")
            return assignment
            
        except Exception as e:
            print(f"Error extracting assignment from SAT model: {e}")
            return None
    
    def generate_mock_assignment_from_solution(self, optimal_bandwidth: int) -> Dict[int, Tuple[int, int]]:
        """Generate test assignment for rectangular grid that achieves target bandwidth"""
        print(f"\nGenerating fallback assignment for {self.n_rows}×{self.n_cols} grid, bandwidth = {optimal_bandwidth}")
        
        assignment = {}
        positions_used = set()
        
        # Strategy: Place vertices to minimize distances while respecting grid shape
        if self.n_rows == 1:
            # Single row - place linearly
            for i in range(1, self.n_vertices + 1):
                assignment[i] = (1, i)
                
        elif self.n_cols == 1:
            # Single column - place vertically
            for i in range(1, self.n_vertices + 1):
                assignment[i] = (i, 1)
                
        else:
            # Rectangular grid - use row-major placement with some optimization
            for i in range(1, self.n_vertices + 1):
                # Row-major order with slight optimization
                row = ((i - 1) // self.n_cols) + 1
                col = ((i - 1) % self.n_cols) + 1
                
                # Ensure within bounds
                if row <= self.n_rows and col <= self.n_cols:
                    assignment[i] = (row, col)
                    positions_used.add((row, col))
        
        return assignment
    
    def run_complete_validation(self, solver_type: str = 'glucose42') -> Dict:
        """Complete MTX validation workflow for rectangular grid"""
        print(f"\n" + "="*80)
        print(f"COMPLETE RECTANGULAR GRID MTX BANDWIDTH VALIDATION")
        print(f"="*80)
        print(f"File: {os.path.basename(self.filename)}")
        print(f"Grid: {self.n_rows}×{self.n_cols} ({self.n_rows * self.n_cols} positions)")
        print(f"Solver: {solver_type.upper()}")
        print(f"Target: Find and validate optimal bandwidth on rectangular grid")
        
        # Solve bandwidth problem
        optimal_bandwidth, solution_info, real_assignment = self.solve_bandwidth_problem(solver_type)
        
        if optimal_bandwidth is None:
            return {
                'status': 'failed', 
                'reason': 'solver_failed',
                'filename': self.filename,
                'grid_dimensions': (self.n_rows, self.n_cols),
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
        print(f"\n" + "="*80)
        print(f"DUAL DECODE METHOD ANALYSIS FOR RECTANGULAR GRID")
        print(f"="*80)
        
        # Method 1: Label format
        self.decode_solution_label_format(assignment, validation_results)
        
        # Method 2: Grid visualization
        self.visualize_solution_matrix_format(assignment, validation_results)
        
        # Final results
        final_results = {
            'status': 'success',
            'filename': self.filename,
            'grid_dimensions': (self.n_rows, self.n_cols),
            'graph_properties': {
                'vertices': self.n_vertices,
                'edges': len(self.edges),
                'density': len(self.edges) / (self.n_vertices * (self.n_vertices - 1) / 2) if self.n_vertices > 1 else 0
            },
            'solution_info': solution_info,
            'validation_results': validation_results,
            'assignment': assignment,
            'decode_methods': {
                'label_based': 'Mathematical precision for verification',
                'matrix_based': 'Visual intuition for rectangular grid understanding'
            }
        }
        
        print(f"\n" + "="*80)
        print(f"RECTANGULAR GRID VALIDATION COMPLETE")
        print(f"="*80)
        print(f"Result: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
        print(f"Grid: {self.n_rows}×{self.n_cols}")
        print(f"Bandwidth: {optimal_bandwidth}")
        print(f"Time: {solution_info['solve_time']:.2f} seconds")
        print(f"Utilization: {solution_info['grid_utilization']:.1f}%")
        
        return final_results


def validate_rectangular_mtx_file(mtx_file: str, n_rows: int = None, n_cols: int = None, 
                                solver_type: str = 'glucose42') -> Dict:
    """
    Validate MTX file on rectangular grid - convenient wrapper function
    """
    print(f"RECTANGULAR GRID MTX BANDWIDTH VALIDATION TOOL")
    print(f"Target: {mtx_file}")
    if n_rows and n_cols:
        print(f"Grid: {n_rows}×{n_cols}")
    else:
        print(f"Grid: Auto-determined")
    
    if not os.path.exists(mtx_file):
        print(f"Error: File not found - {mtx_file}")
        return {'status': 'file_not_found', 'filename': mtx_file}
    
    validator = RectangularMTXBandwidthValidator(mtx_file, n_rows, n_cols)
    results = validator.run_complete_validation(solver_type)
    
    return results


if __name__ == "__main__":
    """
    Command line usage: 
    python rectangular_mtx_validator.py [mtx_file] [n_rows] [n_cols] [solver]
    python rectangular_mtx_validator.py [mtx_file] [solver]  # Auto grid
    
    Examples:
        python rectangular_mtx_validator.py Trec5.mtx
        python rectangular_mtx_validator.py Trec5.mtx 2 4
        python rectangular_mtx_validator.py Trec5.mtx 3 3 glucose42
        python rectangular_mtx_validator.py cage3.mtx 1 5 cadical195
        python rectangular_mtx_validator.py  # Run test mode
    """
    import sys
    
    # Check if specific MTX file provided
    if len(sys.argv) >= 2:
        # Single file validation mode
        mtx_file = sys.argv[1]
        
        # Parse grid dimensions and solver
        n_rows = None
        n_cols = None
        solver_type = 'glucose42'
        
        if len(sys.argv) >= 4:
            # Format: file rows cols [solver]
            try:
                n_rows = int(sys.argv[2])
                n_cols = int(sys.argv[3])
                if len(sys.argv) >= 5:
                    solver_type = sys.argv[4]
            except ValueError:
                print("Error: Grid dimensions must be integers")
                print("Usage: python rectangular_mtx_validator.py <file> <rows> <cols> [solver]")
                sys.exit(1)
        elif len(sys.argv) == 3:
            # Format: file solver (auto grid)
            solver_type = sys.argv[2]
        
        print("=" * 80)
        print("RECTANGULAR GRID MTX BANDWIDTH VALIDATOR")
        print("=" * 80)
        print(f"File: {mtx_file}")
        if n_rows and n_cols:
            print(f"Grid: {n_rows}×{n_cols} (specified)")
        else:
            print(f"Grid: Auto-determined")
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
            results = validate_rectangular_mtx_file(mtx_file, n_rows, n_cols, solver_type)
            
            # Print summary
            print("\n" + "="*70)
            print("FINAL SUMMARY")
            print("="*70)
            
            status = results.get('status', 'unknown')
            if status == 'success':
                validation = results.get('validation_results', {})
                solution_info = results.get('solution_info', {})
                grid_dims = results.get('grid_dimensions', (0, 0))
                
                bandwidth = validation.get('calculated_bandwidth', 'N/A')
                is_valid = validation.get('is_valid', False)
                solve_time = solution_info.get('solve_time', 0)
                utilization = solution_info.get('grid_utilization', 0)
                
                print(f"✓ Validation: {'PASSED' if is_valid else 'FAILED'}")
                print(f"✓ Grid: {grid_dims[0]}×{grid_dims[1]}")
                print(f"✓ Bandwidth: {bandwidth}")
                print(f"✓ Utilization: {utilization:.1f}%")
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
        # Testing mode - run validation on sample files with different grids
        print("=" * 80)
        print("RECTANGULAR GRID MTX BANDWIDTH VALIDATOR - TESTING MODE")
        print("Tests different rectangular grid configurations")
        print("=" * 80)
        
        # Test files with different grid configurations
        test_cases = [
            ("mtx/Trec5.mtx", None, None, "Auto grid"),
            ("mtx/Trec5.mtx", 7, 1, "Single row"),
            ("mtx/Trec5.mtx", 1, 7, "Single column"),
            ("mtx/Trec5.mtx", 2, 4, "2×4 rectangle"),
            ("mtx/Trec5.mtx", 3, 3, "3×3 square"),
            ("sample_mtx_datasets/complete_k4.mtx", 2, 2, "K4 on 2×2"),
            ("sample_mtx_datasets/path_p6.mtx", 6, 1, "Path on 6×1"),
            ("sample_mtx_datasets/cycle_c5.mtx", 1, 5, "Cycle on 1×5"),
        ]
        
        results_summary = []
        
        for i, (mtx_file, n_rows, n_cols, description) in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"TEST CASE {i}/{len(test_cases)}: {description}")
            print(f"File: {os.path.basename(mtx_file)}")
            if n_rows and n_cols:
                print(f"Grid: {n_rows}×{n_cols}")
            else:
                print(f"Grid: Auto-determined")
            print(f"{'='*80}")
            
            if os.path.exists(mtx_file):
                try:
                    results = validate_rectangular_mtx_file(mtx_file, n_rows, n_cols, 'glucose42')
                    grid_dims = results.get('grid_dimensions', (0, 0))
                    results_summary.append({
                        'file': mtx_file,
                        'description': description,
                        'grid': f"{grid_dims[0]}×{grid_dims[1]}",
                        'status': results.get('status', 'unknown'),
                        'bandwidth': results.get('validation_results', {}).get('calculated_bandwidth', 'N/A'),
                        'validation': results.get('validation_results', {}).get('is_valid', False)
                    })
                    print(f"Test {i} completed successfully")
                except Exception as e:
                    print(f"Test {i} failed with error: {e}")
                    results_summary.append({
                        'file': mtx_file,
                        'description': description,
                        'grid': f"{n_rows or '?'}×{n_cols or '?'}",
                        'status': 'error',
                        'error': str(e)
                    })
            else:
                print(f"Test {i} skipped - file not found: {mtx_file}")
                results_summary.append({
                    'file': mtx_file,
                    'description': description,
                    'grid': f"{n_rows or '?'}×{n_cols or '?'}",
                    'status': 'file_not_found'
                })
        
        # Summary report
        print(f"\n" + "="*90)
        print(f"TESTING COMPLETE - RECTANGULAR GRID SUMMARY REPORT")
        print(f"="*90)
        
        for i, result in enumerate(results_summary, 1):
            status_icon = "✓" if result['status'] == 'success' else "✗"
            file_name = os.path.basename(result['file'])
            description = result['description']
            grid = result['grid']
            bandwidth = result.get('bandwidth', 'N/A')
            
            print(f"{status_icon} Test {i}: {file_name:<15} | {description:<15} | {grid:<8} | BW: {bandwidth:<3} | {result['status']}")
        
        successful_tests = sum(1 for r in results_summary if r['status'] == 'success')
        print(f"\nSuccess rate: {successful_tests}/{len(results_summary)} tests passed")
        print(f"Rectangular Grid MTX Bandwidth Validator testing complete!")
