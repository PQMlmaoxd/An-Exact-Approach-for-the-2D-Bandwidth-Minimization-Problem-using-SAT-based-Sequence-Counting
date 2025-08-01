# position_constraints.py
# Position constraint encoding for 2D Bandwidth Minimization Problem

from pysat.formula import IDPool
from nsc_encoder import encode_nsc_exactly_k, encode_nsc_at_most_k

def encode_vertex_position_constraints(n, X_vars, Y_vars, vpool):
    """
    Each vertex gets exactly one X and Y position
    
    Uses NSC exactly-k encoding for efficient constraint generation.
    """
    clauses = []
    
    print(f"Encoding vertex position constraints for {n} vertices...")
    
    for v in range(1, n + 1):
        # Exactly-One for X using NSC
        nsc_x_clauses = encode_nsc_exactly_k(X_vars[v], 1, vpool)
        clauses.extend(nsc_x_clauses)
        
        # Exactly-One for Y using NSC
        nsc_y_clauses = encode_nsc_exactly_k(Y_vars[v], 1, vpool)
        clauses.extend(nsc_y_clauses)
    
    print(f"Generated {len(clauses)} clauses for vertex position constraints")
    return clauses

def encode_position_uniqueness_constraints(n, X_vars, Y_vars, vpool):
    """
    Each grid position (x,y) gets at most one vertex
    
    Creates indicator variables and uses NSC at-most-k encoding
    for O(n²) complexity per position.
    """
    clauses = []
    
    print(f"Encoding position uniqueness for {n}x{n} grid...")
    
    # Each position (x,y) has at most one vertex
    for x in range(n):
        for y in range(n):
            # Create indicator variables: node_at_pos[v] = (X_v_x ∧ Y_v_y)
            node_indicators = []
            for v in range(1, n + 1):
                indicator = vpool.id(f'node_{v}_at_{x}_{y}')
                node_indicators.append(indicator)
                
                # Equivalence: indicator ↔ (X_v_x ∧ Y_v_y)
                # indicator → X_v_x
                clauses.append([-indicator, X_vars[v][x]])
                # indicator → Y_v_y
                clauses.append([-indicator, Y_vars[v][y]])
                # (X_v_x ∧ Y_v_y) → indicator
                clauses.append([indicator, -X_vars[v][x], -Y_vars[v][y]])
            
            # NSC constraint: at most 1 node at position (x,y)
            nsc_at_most_1 = encode_nsc_at_most_k(node_indicators, 1, vpool)
            clauses.extend(nsc_at_most_1)
    
    print(f"Generated {len(clauses)} clauses for position uniqueness")
    return clauses

def encode_all_position_constraints(n, X_vars, Y_vars, vpool):
    """
    Encode all position constraints for the 2D grid
    
    Combines vertex position constraints (exactly-one) with
    position uniqueness constraints (at-most-one).
    """
    print(f"\nEncoding position constraints")
    print(f"Problem: {n} vertices on {n}x{n} grid")
    print(f"Using NSC Sequential Counter for efficiency")
    
    all_clauses = []
    
    # 1. Vertex position constraints (exactly-one)
    vertex_clauses = encode_vertex_position_constraints(n, X_vars, Y_vars, vpool)
    all_clauses.extend(vertex_clauses)
    
    # 2. Position uniqueness constraints (at-most-one)
    uniqueness_clauses = encode_position_uniqueness_constraints(n, X_vars, Y_vars, vpool)
    all_clauses.extend(uniqueness_clauses)
    
    print(f"Total position clauses: {len(all_clauses)}")
    print(f"Complexity: O(n³) = O({n}³) = {n**3}")
    print(f"Position constraints complete\n")
    
    return all_clauses

def create_position_variables(n, vpool):
    """
    Create position variables for all vertices
    
    X_vars[v][pos] = vertex v at X position pos
    Y_vars[v][pos] = vertex v at Y position pos
    """
    X_vars = {}  # X_vars[v][pos] = variable for vertex v at X position pos
    Y_vars = {}  # Y_vars[v][pos] = variable for vertex v at Y position pos
    
    for v in range(1, n + 1):
        X_vars[v] = [vpool.id(f'X_{v}_{pos}') for pos in range(1, n + 1)]
        Y_vars[v] = [vpool.id(f'Y_{v}_{pos}') for pos in range(1, n + 1)]
    
    print(f"Created position variables for {n} vertices")
    print(f"X variables: {n} × {n} = {n*n}")
    print(f"Y variables: {n} × {n} = {n*n}")
    print(f"Total: {2*n*n} position variables")
    
    return X_vars, Y_vars

def test_position_constraints():
    """
    Test position constraints with small example
    """
    from pysat.solvers import Glucose4
    
    print("Testing position constraints")
    
    n = 3  # Small test case
    vpool = IDPool()
    
    # Create position variables
    X_vars, Y_vars = create_position_variables(n, vpool)
    
    # Encode all position constraints
    clauses = encode_all_position_constraints(n, X_vars, Y_vars, vpool)
    
    print(f"\nTesting with {len(clauses)} clauses...")
    
    # Test with SAT solver
    solver = Glucose4()
    for clause in clauses:
        solver.add_clause(clause)
    
    if solver.solve():
        model = solver.get_model()
        print("SAT: Position constraints are satisfiable")
        
        # Extract and display solution
        print("\nSolution:")
        for v in range(1, n + 1):
            x_pos = None
            y_pos = None
            
            # Find X position
            for pos in range(n):
                if X_vars[v][pos] in model:
                    x_pos = pos + 1
                    break
            
            # Find Y position  
            for pos in range(n):
                if Y_vars[v][pos] in model:
                    y_pos = pos + 1
                    break
            
            print(f"  Vertex {v}: X={x_pos}, Y={y_pos}")
    else:
        print("UNSAT: Position constraints are unsatisfiable - ERROR!")
    
    solver.delete()
    print("Position constraints test complete")

if __name__ == '__main__':
    test_position_constraints()
