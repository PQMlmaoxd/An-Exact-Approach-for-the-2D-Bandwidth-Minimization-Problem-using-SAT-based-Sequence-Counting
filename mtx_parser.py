# mtx_parser.py
# Parser cho Matrix Market format files (.mtx)

import os

def parse_mtx_file(filepath):
    """
    Parse MTX file format to extract graph information
    
    Args:
        filepath: Path to .mtx file
        
    Returns:
        tuple: (n, edges) where n is number of vertices and edges is list of tuples
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MTX file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip comment lines starting with %
    data_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('%'):
            data_lines.append(line)
    
    if not data_lines:
        raise ValueError("No data found in MTX file")
    
    # First non-comment line: rows cols nnz
    header = data_lines[0].split()
    if len(header) != 3:
        raise ValueError(f"Invalid MTX header format: {header}")
    
    rows, cols, nnz = map(int, header)
    
    # For graph representation, assume symmetric matrix (undirected graph)
    n = max(rows, cols)
    
    # Parse edges
    edges = []
    for i in range(1, min(len(data_lines), nnz + 1)):
        edge_data = data_lines[i].split()
        if len(edge_data) >= 2:
            u, v = int(edge_data[0]), int(edge_data[1])
            
            # Skip self-loops
            if u != v:
                # Add edge (ensure u < v for consistency)
                if u > v:
                    u, v = v, u
                edges.append((u, v))
    
    # Remove duplicates
    edges = list(set(edges))
    
    print(f"Parsed MTX: {n} vertices, {len(edges)} edges")
    return n, edges

def parse_simple_graph_format(graph_str):
    """
    Parse simple graph format for testing
    
    Format: "n: edge1 edge2 ..." where edges are "u-v"
    Example: "4: 1-2 2-3 3-4" for path graph
    """
    parts = graph_str.split(': ')
    if len(parts) != 2:
        raise ValueError("Invalid format, expected 'n: edge1 edge2 ...'")
    
    n = int(parts[0])
    edge_strs = parts[1].split()
    
    edges = []
    for edge_str in edge_strs:
        u, v = map(int, edge_str.split('-'))
        edges.append((u, v))
    
    return n, edges

if __name__ == '__main__':
    # Test with sample MTX files
    sample_files = [
        "sample_mtx_datasets/path_p6.mtx",
        "sample_mtx_datasets/cycle_c5.mtx", 
        "sample_mtx_datasets/complete_k4.mtx"
    ]
    
    for filepath in sample_files:
        if os.path.exists(filepath):
            try:
                n, edges = parse_mtx_file(filepath)
                print(f"{filepath}: n={n}, edges={edges}")
            except Exception as e:
                print(f"Error parsing {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")
