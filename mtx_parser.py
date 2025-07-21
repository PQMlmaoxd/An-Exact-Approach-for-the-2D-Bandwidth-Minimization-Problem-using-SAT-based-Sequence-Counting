# mtx_parser.py
# Parser for MTX (Matrix Market) sparse matrix format
# Converts sparse matrix to graph edge list for bandwidth optimization

import re
import os

class MTXParser:
    """
    Parser for MTX (Matrix Market) format files
    Converts sparse matrix representation to graph edge list
    """
    
    def __init__(self, mtx_file_path):
        """
        Initialize MTX parser
        
        Args:
            mtx_file_path: Path to the .mtx file
        """
        self.mtx_file_path = mtx_file_path
        self.matrix_info = {}
        self.edges = []
        self.num_nodes = 0
        self.num_edges = 0
        
    def parse_mtx_file(self):
        """
        Parse MTX file and extract graph structure
        
        Returns:
            dict: Dictionary with graph information
                - num_nodes: Number of nodes
                - num_edges: Number of edges  
                - edges: List of tuples [(u,v), ...]
                - matrix_info: Header information
        """
        print(f"Parsing MTX file: {self.mtx_file_path}")
        
        with open(self.mtx_file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        header_line = lines[0].strip()
        if not header_line.startswith('%%MatrixMarket'):
            raise ValueError("Invalid MTX file: Missing MatrixMarket header")
        
        print(f"Header: {header_line}")
        
        # Skip comment lines
        data_start_idx = 0
        for i, line in enumerate(lines[1:], 1):
            if not line.strip().startswith('%'):
                data_start_idx = i
                break
        
        # Parse matrix dimensions
        dims_line = lines[data_start_idx].strip().split()
        rows = int(dims_line[0])
        cols = int(dims_line[1])
        nnz = int(dims_line[2])  # Number of non-zero entries
        
        self.matrix_info = {
            'rows': rows,
            'cols': cols,
            'nnz': nnz,
            'header': header_line
        }
        
        print(f"Matrix dimensions: {rows} x {cols}, Non-zeros: {nnz}")
        
        # For graph problems, assume square matrix (rows == cols)
        if rows != cols:
            print(f"Warning: Non-square matrix ({rows} x {cols}). Using max dimension.")
            self.num_nodes = max(rows, cols)
        else:
            self.num_nodes = rows
        
        # Parse matrix entries and convert to edges
        edges_set = set()  # Use set to avoid duplicate edges
        
        for line in lines[data_start_idx + 1:]:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 2:
                continue
                
            row = int(parts[0])
            col = int(parts[1])
            
            # Skip diagonal entries (self-loops)
            if row == col:
                continue
            
            # Add edge (ensure undirected by ordering)
            edge = tuple(sorted([row, col]))
            edges_set.add(edge)
        
        # Convert to list
        self.edges = list(edges_set)
        self.num_edges = len(self.edges)
        
        print(f"Graph extracted: {self.num_nodes} nodes, {self.num_edges} edges")
        print(f"Sample edges: {self.edges[:5]}...")
        
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'edges': self.edges,
            'matrix_info': self.matrix_info
        }
    
    def get_graph_statistics(self):
        """
        Get basic graph statistics
        
        Returns:
            dict: Graph statistics
        """
        if not self.edges:
            return {}
        
        # Node degrees
        degree_count = {}
        for u, v in self.edges:
            degree_count[u] = degree_count.get(u, 0) + 1
            degree_count[v] = degree_count.get(v, 0) + 1
        
        degrees = list(degree_count.values())
        
        stats = {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'min_degree': min(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
            'density': (2 * self.num_edges) / (self.num_nodes * (self.num_nodes - 1)) if self.num_nodes > 1 else 0
        }
        
        return stats
    
    def save_graph_info(self, output_file=None):
        """
        Save graph information to file
        
        Args:
            output_file: Output file path (default: based on input filename)
        """
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(self.mtx_file_path))[0]
            output_file = f"{base_name}_graph_info.txt"
        
        stats = self.get_graph_statistics()
        
        with open(output_file, 'w') as f:
            f.write(f"MTX File: {self.mtx_file_path}\n")
            f.write(f"Matrix Info: {self.matrix_info}\n\n")
            f.write(f"Graph Statistics:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nEdges ({len(self.edges)} total):\n")
            for u, v in self.edges[:50]:  # Show first 50 edges
                f.write(f"  ({u}, {v})\n")
            if len(self.edges) > 50:
                f.write(f"  ... and {len(self.edges) - 50} more edges\n")
        
        print(f"Graph information saved to: {output_file}")
        return output_file

def test_mtx_parser():
    """
    Test MTX parser with sample files
    """
    print("=== TESTING MTX PARSER ===")
    
    # Test with 1138_bus.mtx
    mtx_file = "mtx/1138_bus.mtx"
    
    if os.path.exists(mtx_file):
        print(f"\nTesting with {mtx_file}:")
        
        parser = MTXParser(mtx_file)
        graph_data = parser.parse_mtx_file()
        
        print(f"\nGraph Data Summary:")
        print(f"  Nodes: {graph_data['num_nodes']}")
        print(f"  Edges: {graph_data['num_edges']}")
        print(f"  Matrix Info: {graph_data['matrix_info']}")
        
        stats = parser.get_graph_statistics()
        print(f"\nGraph Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        # Save detailed info
        info_file = parser.save_graph_info()
        print(f"\nDetailed info saved to: {info_file}")
        
        return graph_data
    else:
        print(f"MTX file not found: {mtx_file}")
        return None

if __name__ == '__main__':
    test_mtx_parser()
