"""
Unified MTX Parser for 2D Bandwidth Minimization
Supports both adjacency matrix and grid position formats
"""

def detect_mtx_format(file_path):
    """
    Detect MTX file format by analyzing header and first few entries
    Returns: ('adjacency', n_vertices, edges) or ('grid', n_rows, n_cols, positions)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find header line
    header_line = None
    data_start = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith('%'):
            header_line = line
            data_start = i + 1
            break
    
    if not header_line:
        raise ValueError(f"No valid header found in {file_path}")
    
    # Parse header
    header_parts = header_line.split()
    if len(header_parts) < 3:
        raise ValueError(f"Invalid header format: {header_line}")
    
    rows, cols, nnz = map(int, header_parts[:3])
    
    # Sample first few data entries to detect format
    sample_entries = []
    for i in range(data_start, min(data_start + 5, len(lines))):
        line = lines[i].strip()
        if line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    # Try to parse third column as float
                    val = float(parts[2])
                    sample_entries.append((int(parts[0]), int(parts[1]), val))
                except ValueError:
                    continue
    
    if not sample_entries:
        raise ValueError(f"No valid data entries found in {file_path}")
    
    # Analyze patterns to determine format
    values = [entry[2] for entry in sample_entries]
    
    # Check if values are in [0,1] range (typical for adjacency matrices)
    is_adjacency_pattern = all(0 <= v <= 1 for v in values) and any(v != int(v) for v in values)
    
    # Check if all values are integers (typical for grid positions)
    is_grid_pattern = all(v == int(v) for v in values)
    
    print(f"MTX Analysis: {rows}×{cols}, {nnz} entries")
    print(f"Sample values: {values[:3]}...")
    print(f"Adjacency pattern: {is_adjacency_pattern}, Grid pattern: {is_grid_pattern}")
    
    if is_adjacency_pattern:
        # Adjacency matrix format
        return parse_adjacency_mtx(file_path, rows, cols, nnz)
    elif is_grid_pattern:
        # Grid position format  
        return parse_grid_mtx(file_path, rows, cols, nnz)
    else:
        # Default to adjacency for mixed/unknown patterns
        print("Warning: Unknown pattern, defaulting to adjacency matrix")
        return parse_adjacency_mtx(file_path, rows, cols, nnz)


def parse_adjacency_mtx(file_path, rows, cols, nnz):
    """
    Parse adjacency matrix format (like cage4)
    Format: n n nnz, then edge list with weights
    Returns: ('adjacency', n_vertices, edges)
    """
    edges = []
    n_vertices = max(rows, cols)  # Use larger dimension as vertex count
    
    with open(file_path, 'r') as f:
        # Skip to data
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()
        
        # Skip header
        line = f.readline()
        
        # Parse edges
        edges_set = set()
        for _ in range(nnz):
            line = f.readline().strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    
                    # Skip self-loops
                    if u == v:
                        continue
                    
                    # Convert to undirected edge
                    edge = tuple(sorted([u, v]))
                    if edge not in edges_set:
                        edges_set.add(edge)
                        edges.append(edge)
    
    print(f"Adjacency matrix: {n_vertices} vertices, {len(edges)} edges")
    return 'adjacency', n_vertices, edges


def parse_grid_mtx(file_path, rows, cols, nnz):
    """
    Parse grid position format (like Trec5)
    Format: rows cols nnz, then position coordinates
    Returns: ('grid', rows, cols, positions)
    """
    positions = []
    
    with open(file_path, 'r') as f:
        # Skip to data
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()
        
        # Skip header
        line = f.readline()
        
        # Parse positions
        for _ in range(nnz):
            line = f.readline().strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    i, j = int(parts[0]), int(parts[1])
                    positions.append((i, j))
    
    # Generate edges between adjacent positions
    edges = []
    for idx, (i1, j1) in enumerate(positions):
        for i2, j2 in positions[idx+1:]:
            # Manhattan distance = 1 (adjacent)
            if abs(i1-i2) + abs(j1-j2) == 1:
                edges.append((idx+1, idx+2))  # 1-indexed vertices
    
    print(f"Grid: {rows}×{cols}, {len(positions)} positions, {len(edges)} edges")
    return 'grid', rows, cols, len(positions), edges


def parse_mtx_unified(file_path):
    """
    Unified MTX parser that auto-detects format
    Returns: (format_type, ...) where format_type is 'adjacency' or 'grid'
    """
    return detect_mtx_format(file_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python unified_mtx_parser.py <mtx_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    try:
        result = parse_mtx_unified(file_path)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
