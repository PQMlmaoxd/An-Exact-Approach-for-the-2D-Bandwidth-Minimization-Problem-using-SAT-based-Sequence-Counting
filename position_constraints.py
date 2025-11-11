# position_constraints.py
# Position constraint encoding for 2D Bandwidth Minimization Problem

from pysat.formula import IDPool
from pysat.card import CardEnc, EncType

def encode_vertex_position_constraints(n, X_vars, Y_vars, vpool):
    """
    Each vertex gets exactly one X and Y position
    
    Uses Sequential Counter exactly-k encoding for efficient constraint generation.
    MEMORY OPTIMIZED: Yields clauses one by one instead of accumulating in list.
    """
    print(f"Encoding vertex position constraints for {n} vertices...")
    clause_count = 0
    
    for v in range(1, n + 1):
        # Exactly-One for X using Sequential Counter
        sc_x_clauses = CardEnc.equals(X_vars[v], 1, vpool=vpool, encoding=EncType.seqcounter)
        for clause in sc_x_clauses.clauses:
            yield clause
            clause_count += 1
        
        # Exactly-One for Y using Sequential Counter
        sc_y_clauses = CardEnc.equals(Y_vars[v], 1, vpool=vpool, encoding=EncType.seqcounter)
        for clause in sc_y_clauses.clauses:
            yield clause
            clause_count += 1
    
    print(f"Generated {clause_count} clauses for vertex position constraints")

def encode_position_uniqueness_constraints(n, X_vars, Y_vars, vpool):
    """
    Each grid position (x,y) gets at most one vertex
    
    Creates indicator variables and uses Sequential Counter at-most-k encoding
    for O(n²) complexity per position.
    MEMORY OPTIMIZED: Yields clauses one by one instead of accumulating in list.
    """
    print(f"Encoding position uniqueness for {n}x{n} grid...")
    clause_count = 0
    
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
                yield [-indicator, X_vars[v][x]]
                clause_count += 1
                # indicator → Y_v_y
                yield [-indicator, Y_vars[v][y]]
                clause_count += 1
                # (X_v_x ∧ Y_v_y) → indicator
                yield [indicator, -X_vars[v][x], -Y_vars[v][y]]
                clause_count += 1
            
            # Sequential Counter constraint: at most 1 node at position (x,y)
            sc_at_most_1 = CardEnc.atmost(node_indicators, 1, vpool=vpool, encoding=EncType.seqcounter)
            for clause in sc_at_most_1.clauses:
                yield clause
                clause_count += 1
    
    print(f"Generated {clause_count} clauses for position uniqueness")

def encode_all_position_constraints(n, X_vars, Y_vars, vpool):
    """
    Encode all position constraints for the 2D grid
    
    Combines vertex position constraints (exactly-one) with
    position uniqueness constraints (at-most-one).
    Uses Sequential Counter encoding for efficient constraint generation.
    MEMORY OPTIMIZED: Generator that yields clauses one by one instead of returning huge list.
    """
    print(f"\nEncoding position constraints")
    print(f"Problem: {n} vertices on {n}x{n} grid")
    print(f"Memory optimization: Streaming clauses via generator (no intermediate list)")
    
    # 1. Vertex position constraints (exactly-one) - stream via generator
    for clause in encode_vertex_position_constraints(n, X_vars, Y_vars, vpool):
        yield clause
    
    # 2. Position uniqueness constraints (at-most-one) - stream via generator
    for clause in encode_position_uniqueness_constraints(n, X_vars, Y_vars, vpool):
        yield clause

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
    MEMORY OPTIMIZED: Works with generator to stream clauses directly to solver
    """
    from pysat.solvers import Glucose4
    
    print("Testing position constraints")
    
    n = 3  # Small test case
    vpool = IDPool()
    
    # Create position variables
    X_vars, Y_vars = create_position_variables(n, vpool)
    
    # Encode all position constraints (generator - no intermediate list)
    clause_generator = encode_all_position_constraints(n, X_vars, Y_vars, vpool)
    
    print(f"\nTesting with streaming clauses...")
    
    # Test with SAT solver - stream clauses directly
    solver = Glucose4()
    clause_count = 0
    for clause in clause_generator:
        solver.add_clause(clause)
        clause_count += 1
    
    print(f"Total clauses added: {clause_count}")
    
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
    print("Position constraints test complete (memory optimized)")

if __name__ == '__main__':
    test_position_constraints()
