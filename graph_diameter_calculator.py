#!/usr/bin/env python3
# graph_diameter_calculator.py
# Calculate graph diameter D(G) using BFS on unweighted graphs
# Uses MTX parser from incremental_bandwidth_solver.py

import os
import sys
import time
from collections import deque, defaultdict

def parse_mtx_file(filename):
    """
    Parse MTX file and return n, edges
    
    Handles MatrixMarket format:
    - Comments and metadata parsing
    - Self-loop removal  
    - Undirected graph processing only
    - Error handling for malformed files
    
    Optimized: Stream reading (for line in f) to reduce RAM on large files
    Based on parser from incremental_bandwidth_solver.py
    """
    print(f"Reading MTX file: {os.path.basename(filename)}")
    
    try:
        with open(filename, 'r') as f:
            header_found = False
            edges_set = set()
            n = 0
            line_num = 0
            
            # Stream reading - process line by line to reduce RAM
            for line in f:
                line_num += 1
                line = line.strip()
                
                if not line:
                    continue
                    
                # Handle comments and metadata
                if line.startswith('%'):
                    # Skip metadata - dataset is all undirected/unweighted
                    continue
                
                # Parse dimensions
                if not header_found:
                    try:
                        parts = line.split()
                        if len(parts) >= 3:
                            rows, cols, nnz = map(int, parts[:3])
                            n = max(rows, cols)
                            print(f"Matrix: {rows}×{cols}, {nnz} entries")
                            print(f"Graph: undirected, unweighted (dataset standard)")
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
                        # Ignore weights (parts[2]) - dataset is unweighted
                        
                        if u == v:  # skip self-loops
                            continue
                        
                        # Always convert to undirected edge (sorted tuple)
                        edge = tuple(sorted([u, v]))
                        
                        if edge not in edges_set:
                            edges_set.add(edge)
                            
                except (ValueError, IndexError):
                    print(f"Warning: bad edge at line {line_num}: {line}")
                    continue
            
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None, None
    
    edges = list(edges_set)
    print(f"Loaded: {n} vertices, {len(edges)} edges")
    return n, edges


def graph_diameter(n, edges):
    """
    Calculate exact graph diameter using optimized BFS from every vertex.
    
    For unweighted undirected graphs, uses BFS to find shortest paths
    measured by number of edges.
    
    Optimizations:
    - Use list instead of dict for distances (faster access)
    - Use list instead of set for adjacency (faster iteration)  
    - Return distance array and farthest vertex directly from BFS
    - Process only nodes within components for better cache locality
    
    Complexity: O(|V| * (|V| + |E|)) - acceptable for graphs with 
    thousands to tens of thousands of vertices.
    
    Args:
        n: Number of vertices (1-indexed)
        edges: List of edges as (u, v) tuples
        
    Returns:
        (overall_diameter, component_info)
        - overall_diameter: D(G) = max diameter across all components
        - component_info: List of (size, diameter, endpoint_a, endpoint_b) 
                         for each connected component
    """
    print(f"\nCalculating graph diameter using optimized BFS...")
    print(f"Graph: {n} vertices, {len(edges)} edges")
    
    # Build adjacency list using lists for faster iteration (1-indexed)
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def bfs_optimized(start, component_nodes):
        """
        Optimized BFS: Use list for distances, return distance array and farthest vertex.
        Only considers vertices within the given component for efficiency.
        """
        # Use list instead of dict for O(1) access
        dist = [-1] * (n + 1)
        queue = deque([start])
        dist[start] = 0
        
        while queue:
            u = queue.popleft()
            du = dist[u] + 1  # Pre-compute distance to avoid repeated addition
            
            for neighbor in adj[u]:
                if dist[neighbor] == -1:
                    dist[neighbor] = du
                    queue.append(neighbor)
        
        # Find farthest vertex within component only (faster than scanning all n vertices)
        farthest_vertex = start
        max_distance = 0
        for node in component_nodes:
            if dist[node] > max_distance:
                max_distance = dist[node]
                farthest_vertex = node
        
        return dist, farthest_vertex, max_distance
    
    # Track visited vertices across components
    seen = [False] * (n + 1)
    overall_diameter = 0
    component_info = []
    
    print(f"Finding connected components and computing diameters...")
    
    for start_vertex in range(1, n + 1):
        if seen[start_vertex]:
            continue
        
        # Get connected component containing start_vertex using initial BFS
        initial_dist, _, _ = bfs_optimized(start_vertex, [start_vertex])
        component_nodes = [v for v in range(1, n + 1) if initial_dist[v] != -1]
        
        # Mark all nodes in this component as visited
        for node in component_nodes:
            seen[node] = True
        
        print(f"  Component with {len(component_nodes)} vertices: ", end="", flush=True)
        
        # Calculate exact diameter within this component
        # Optimized: BFS from every vertex in component, use fast distance array access
        component_diameter = 0
        diameter_endpoint_a = start_vertex
        diameter_endpoint_b = start_vertex
        
        for vertex in component_nodes:
            _, farthest_vertex, max_distance = bfs_optimized(vertex, component_nodes)
            
            # Update component diameter if we found a longer path
            if max_distance > component_diameter:
                component_diameter = max_distance
                diameter_endpoint_a = vertex
                diameter_endpoint_b = farthest_vertex
        
        print(f"diameter = {component_diameter} (from vertex {diameter_endpoint_a} to {diameter_endpoint_b})")
        
        # Store component information
        component_info.append((
            len(component_nodes), 
            component_diameter, 
            diameter_endpoint_a, 
            diameter_endpoint_b
        ))
        
        # Update overall diameter
        overall_diameter = max(overall_diameter, component_diameter)
    
    return overall_diameter, component_info


def calculate_graph_diameter(mtx_file):
    """
    Main function to calculate graph diameter from MTX file
    
    Args:
        mtx_file: Path to MTX file
        
    Returns:
        Dictionary with diameter results and statistics
    """
    print("="*80)
    print("GRAPH DIAMETER CALCULATOR")
    print("="*80)
    print(f"File: {mtx_file}")
    print(f"Method: Optimized BFS from every vertex (exact)")
    
    # Search for file in common locations
    if not os.path.exists(mtx_file):
        search_paths = [
            mtx_file,
            f"mtx/{mtx_file}",
            f"mtx/group 1/{mtx_file}",
            f"mtx/group 2/{mtx_file}",
            f"mtx/group 3/{mtx_file}",
            f"mtx/regular/{mtx_file}",
            f"mtx/{mtx_file}.mtx",
            f"mtx/group 1/{mtx_file}.mtx",
            f"mtx/group 2/{mtx_file}.mtx",
            f"mtx/group 3/{mtx_file}.mtx",
            f"mtx/regular/{mtx_file}.mtx"
        ]
        
        found_file = None
        for path in search_paths:
            if os.path.exists(path):
                found_file = path
                print(f"Found file at: {path}")
                break
        
        if found_file is None:
            print(f"Error: File '{mtx_file}' not found")
            print("Searched in:")
            for path in search_paths:
                print(f"  - {path}")
            return None
        
        mtx_file = found_file
    
    # Parse MTX file
    start_time = time.time()
    n, edges = parse_mtx_file(mtx_file)
    parse_time = time.time() - start_time
    
    if n is None or edges is None:
        print("Failed to parse MTX file")
        return None
    
    # Calculate diameter
    start_time = time.time()
    diameter, components = graph_diameter(n, edges)
    diameter_time = time.time() - start_time
    
    # Print results
    print(f"\n" + "="*60)
    print(f"DIAMETER CALCULATION RESULTS")
    print(f"="*60)
    
    print(f"Graph properties:")
    print(f"  Vertices: {n}")
    print(f"  Edges: {len(edges)}")
    print(f"  Connected components: {len(components)}")
    
    if len(components) == 1:
        print(f"  Graph connectivity: Connected")
    else:
        print(f"  Graph connectivity: Disconnected ({len(components)} components)")
    
    print(f"\nDiameter results:")
    print(f"  Overall diameter D(G): {diameter}")
    
    if len(components) > 1:
        print(f"  Note: For disconnected graphs, D(G) = max diameter over all components")
    
    print(f"\nComponent details:")
    for i, (size, comp_diameter, end_a, end_b) in enumerate(components, 1):
        print(f"  Component {i}: {size} vertices, diameter {comp_diameter}")
        print(f"    Diameter path: vertex {end_a} → vertex {end_b}")
    
    print(f"\nPerformance:")
    print(f"  Parse time: {parse_time:.3f}s")
    print(f"  Diameter calculation: {diameter_time:.3f}s")
    print(f"  Total time: {parse_time + diameter_time:.3f}s")
    
    # Complexity analysis
    bfs_calls = n  # BFS from every vertex
    complexity_estimate = n * (n + len(edges))
    print(f"  BFS calls: {bfs_calls}")
    print(f"  Complexity: O({n} × ({n} + {len(edges)})) ≈ {complexity_estimate:,} operations")
    
    print(f"="*60)
    
    # Return structured results
    return {
        'diameter': diameter,
        'vertices': n,
        'edges': len(edges),
        'components': len(components),
        'component_details': components,
        'parse_time': parse_time,
        'diameter_time': diameter_time,
        'total_time': parse_time + diameter_time,
        'is_connected': len(components) == 1
    }


if __name__ == "__main__":
    """
    Command line usage: python graph_diameter_calculator.py <mtx_file>
    
    Arguments:
        mtx_file: Name of MTX file (searches in mtx/group 1/, mtx/group 2/, etc.)
    
    Examples:
        python graph_diameter_calculator.py bcsstk01.mtx
        python graph_diameter_calculator.py jgl009.mtx
        python graph_diameter_calculator.py ash85.mtx
        python graph_diameter_calculator.py can___24.mtx
    
    Output:
        - Graph diameter D(G) 
        - Connected component analysis
        - Performance statistics
    """
    
    if len(sys.argv) < 2:
        print("="*80)
        print("GRAPH DIAMETER CALCULATOR")
        print("="*80)
        print("Usage: python graph_diameter_calculator.py <mtx_file>")
        print()
        print("Arguments:")
        print("  mtx_file: Name of MTX file")
        print()
        print("Examples:")
        print("  python graph_diameter_calculator.py bcsstk01.mtx")
        print("  python graph_diameter_calculator.py jgl009.mtx")
        print("  python graph_diameter_calculator.py ash85.mtx")
        print("  python graph_diameter_calculator.py can___24.mtx")
        print()
        print("Method:")
        print("  - Optimized BFS from every vertex (exact algorithm)")
        print("  - Uses list instead of dict/set for better performance")
        print("  - Stream reading for large files to reduce RAM usage")
        print("  - Handles disconnected graphs (reports max diameter)")
        print("  - Complexity: O(|V| × (|V| + |E|))")
        print()
        print("Available MTX files:")
        print("  Group 1: bcspwr01.mtx, bcspwr02.mtx, bcsstk01.mtx, can___24.mtx,")
        print("           fidap005.mtx, fidapm05.mtx, ibm32.mtx, jgl009.mtx,")
        print("           jgl011.mtx, lap_25.mtx, pores_1.mtx, rgg010.mtx")
        print("  Group 2: ash85.mtx")
        print("  Group 3: ck104.mtx, bcsstk04.mtx, bcsstk05.mtx, etc.")
        print()
        print("Output:")
        print("  - Overall diameter D(G)")
        print("  - Component-wise diameter analysis") 
        print("  - Performance and complexity statistics")
        sys.exit(1)
    
    # Parse arguments
    mtx_file = sys.argv[1]
    
    # Calculate diameter
    try:
        results = calculate_graph_diameter(mtx_file)
        
        if results is None:
            print("✗ Failed to calculate diameter")
            sys.exit(1)
        
        # Summary output
        print(f"\n" + "="*80)
        print(f"SUMMARY")
        print(f"="*80)
        print(f"File: {os.path.basename(mtx_file)}")
        print(f"Graph diameter D(G): {results['diameter']}")
        print(f"Vertices: {results['vertices']}")
        print(f"Edges: {results['edges']}")
        print(f"Connected: {'Yes' if results['is_connected'] else f'No ({results['components']} components)'}")
        print(f"Calculation time: {results['diameter_time']:.3f}s")
        print(f"="*80)
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
