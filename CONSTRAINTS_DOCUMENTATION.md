# SAT-based Constraint Encoding for 2D Bandwidth Minimization

## Overview

This document provides a comprehensive analysis of the constraint encoding strategies employed in our SAT-based approach to the 2D Bandwidth Minimization Problem. The solver transforms the optimization problem into a satisfiability instance through sophisticated constraint formulations.

**Problem Definition**: Given a graph G=(V,E) with n vertices, place all vertices on an n×n grid to minimize bandwidth.
**Bandwidth Metric**: `bandwidth = max{|x_u - x_v| + |y_u - y_v| : (u,v) ∈ E}` (Manhattan distance)

---

## 1. POSITION CONSTRAINTS

The position constraint system ensures valid vertex placement on the 2D grid through two complementary mechanisms.

### 1.1 Vertex Assignment Constraints (Exactly-One)

These constraints guarantee that each vertex occupies exactly one position along both X and Y axes.

**Source**: `position_constraints.py` - `encode_vertex_position_constraints()`

**Variable Schema**:
- `X_vars[v][pos]`: Boolean indicating vertex v is at X-coordinate pos
- `Y_vars[v][pos]`: Boolean indicating vertex v is at Y-coordinate pos

**Mathematical Formulation**:
```
∀v ∈ {1..n}: exactly_one(X_v_1, X_v_2, ..., X_v_n)
∀v ∈ {1..n}: exactly_one(Y_v_1, Y_v_2, ..., Y_v_n)
```

**SAT Encoding Strategy**:
- Employs Network Sequential Counter (NSC) with exactly-k encoding where k=1
- Each vertex requires 2n boolean variables (n for X-axis, n for Y-axis)
- Ensures exactly one X-variable and one Y-variable evaluate to True per vertex

**Implementation**:
```python
def encode_vertex_position_constraints(n, X_vars, Y_vars, vpool):
    clauses = []
    for v in range(1, n + 1):
        # X-axis exactly-one constraint
        nsc_x_clauses = encode_nsc_exactly_k(X_vars[v], 1, vpool)
        clauses.extend(nsc_x_clauses)
        
        # Y-axis exactly-one constraint
        nsc_y_clauses = encode_nsc_exactly_k(Y_vars[v], 1, vpool)
        clauses.extend(nsc_y_clauses)
    return clauses
```

**Complexity**: O(n²) - derived from n vertices × 2 axes × O(n) NSC encoding overhead

### 1.2 Grid Occupancy Constraints (At-Most-One)

These constraints prevent multiple vertices from occupying the same grid position, ensuring a valid placement.

**Source**: `position_constraints.py` - `encode_position_uniqueness_constraints()`

**Auxiliary Variables**:
- `node_v_at_x_y`: Indicator variable for vertex v at grid position (x,y)

**Mathematical Formulation**:
```
∀(x,y) ∈ {1..n}×{1..n}: at_most_one(node_1_at_x_y, node_2_at_x_y, ..., node_n_at_x_y)
```

**Variable Equivalence Relations**:
```
node_v_at_x_y ↔ (X_v_x ∧ Y_v_y)
```

**SAT Clause Generation**:
```
# Bidirectional equivalence: indicator ↔ (X_v_x ∧ Y_v_y)
indicator → X_v_x           : [-indicator, X_v_x]
indicator → Y_v_y           : [-indicator, Y_v_y]
(X_v_x ∧ Y_v_y) → indicator : [indicator, -X_v_x, -Y_v_y]
```

**Implementation**:
```python
def encode_position_uniqueness_constraints(n, X_vars, Y_vars, vpool):
    clauses = []
    for x in range(n):
        for y in range(n):
            node_indicators = []
            for v in range(1, n + 1):
                indicator = vpool.id(f'node_{v}_at_{x}_{y}')
                node_indicators.append(indicator)
                
                # Establish equivalence constraints
                clauses.append([-indicator, X_vars[v][x]])
                clauses.append([-indicator, Y_vars[v][y]])
                clauses.append([indicator, -X_vars[v][x], -Y_vars[v][y]])
            
            # Apply NSC at-most-1 constraint to indicators
            nsc_at_most_1 = encode_nsc_at_most_k(node_indicators, 1, vpool)
            clauses.extend(nsc_at_most_1)
    return clauses
```

**Complexity**: O(n³) - n² grid positions × n vertices × constant encoding overhead

---

## 2. DISTANCE CONSTRAINTS

The distance constraint subsystem handles the precise encoding of Manhattan distances between vertex pairs connected by edges.

### 2.1 Manhattan Distance Representation

Our approach employs thermometer encoding to represent the Manhattan distance `|x_u - x_v| + |y_u - y_v|` for each edge in the graph.

**Source**: `distance_encoder.py` - `encode_abs_distance_final()`

**Thermometer Variables**:
- `T_vars[d]`: Boolean indicator meaning "distance ≥ d+1"
- Each edge maintains two separate arrays: `Tx_vars` (X-distance) and `Ty_vars` (Y-distance)

**Encoding Principle**: 
- Utilizes thermometer representation: T[0], T[1], ..., T[n-1]
- Semantic: T[i] = True ⟺ distance ≥ i+1
- Monotonicity property: T[0] ≥ T[1] ≥ ... ≥ T[n-1]

### 2.2 Thermometer Activation Rules

These constraints ensure thermometer variables are properly activated based on actual vertex positions.

**Mathematical Formulation**:
```
∀k,d: (V_k ∧ U_{k-d}) → T_d    (case: V > U)
∀k,d: (U_k ∧ V_{k-d}) → T_d    (case: U > V)
```

**SAT Clause Translation**:
```
# V > U scenario
¬V_k ∨ ¬U_{k-d} ∨ T_d

# U > V scenario  
¬U_k ∨ ¬V_{k-d} ∨ T_d
```

**Implementation Pattern**:
```python
# Symmetric activation rules with O(n²) complexity
for k in range(1, n + 1):
    for d in range(1, k):
        if d - 1 < len(T_vars):
            u_pos = k - d
            if u_pos >= 1:
                # Constraint: (V_k ∧ U_{k-d}) → T_d
                clauses.append([-V_vars[k - 1], -U_vars[u_pos - 1], T_vars[d - 1]])
```

### 2.3 Thermometer Deactivation Rules

These constraints provide tight encoding by preventing thermometer variables from exceeding actual distances.

**Mathematical Formulation**:
```
∀k,d: (V_k ∧ U_{k-d}) → ¬T_{d+1}    (V > U, distance = d)
∀k,d: (U_k ∧ V_{k-d}) → ¬T_{d+1}    (U > V, distance = d)
```

**SAT Clause Translation**:
```
# Tight deactivation constraints
¬V_k ∨ ¬U_{k-d} ∨ ¬T_{d+1}
¬U_k ∨ ¬V_{k-d} ∨ ¬T_{d+1}
```

**Implementation Pattern**:
```python
# Precise distance bounds with O(n²) complexity
for k in range(1, n + 1):
    for d in range(1, k):
        if d < len(T_vars):
            u_pos = k - d
            if u_pos >= 1:
                # Constraint: (V_k ∧ U_{k-d}) → ¬T_{d+1}
                clauses.append([-V_vars[k - 1], -U_vars[u_pos - 1], -T_vars[d]])
```

### 2.4 Zero Distance Handling

Special case constraints for vertices occupying identical positions.

**Mathematical Formulation**:
```
∀k: (U_k ∧ V_k) → ¬T_1    (co-location → distance < 1)
```

**SAT Clause Translation**:
```
¬U_k ∨ ¬V_k ∨ ¬T_1
```

**Implementation Pattern**:
```python
# Zero distance constraint handling
for k in range(1, n + 1):
    if len(T_vars) > 0:
        # Constraint: (U_k ∧ V_k) → ¬T_1
        clauses.append([-U_vars[k - 1], -V_vars[k - 1], -T_vars[0]])
```

### 2.5 Monotonicity Enforcement

Ensures the thermometer property is maintained across all distance variables.

**Mathematical Formulation**:
```
∀d ∈ {1..n-2}: ¬T_d → ¬T_{d+1}
```

**SAT Clause Translation**:
```
T_d ∨ ¬T_{d+1}
```

**Implementation Pattern**:
```python
# Monotonicity: T[d-1] → T[d] equivalent to ¬T[d-1] ∨ T[d]
for d in range(1, len(T_vars)):
    clauses.append([T_vars[d - 1], -T_vars[d]])
```

**Total Complexity**: O(n² × |E|) across all edges

---

## 3. BANDWIDTH CONSTRAINTS

The bandwidth constraint system enforces the optimization objective by limiting maximum edge distances.

### 3.1 Distance Upper Bounds

These constraints ensure all edge distances remain within the target bandwidth K.

**Source**: `bandwidth_optimization_solver.py` - `encode_thermometer_bandwidth_constraints()`

**Mathematical Formulation**:
```
∀edge ∈ E: Tx_edge ≤ K
∀edge ∈ E: Ty_edge ≤ K
```

**SAT Encoding via Thermometer Logic**:
```
# Tx ≤ K equivalent to ¬(Tx ≥ K+1)
¬Tx_edge[K]

# Ty ≤ K equivalent to ¬(Ty ≥ K+1)  
¬Ty_edge[K]
```

**Implementation Framework**:
```python
def encode_thermometer_bandwidth_constraints(self, K):
    clauses = []
    for edge_id in self.Tx_vars:
        Tx = self.Tx_vars[edge_id]
        Ty = self.Ty_vars[edge_id]
        
        # Constrain X-distance ≤ K
        if K < len(Tx):
            clauses.append([-Tx[K]])
            
        # Constrain Y-distance ≤ K
        if K < len(Ty):
            clauses.append([-Ty[K]])
```

### 3.2 Distance Coupling Constraints

These constraints enforce the fundamental relationship between X and Y distances to maintain the bandwidth bound.

**Core Insight**: When X-distance is large, Y-distance must be correspondingly small to keep total distance ≤ K.

**Mathematical Formulation**:
```
∀edge ∈ E, ∀i ∈ {1..K}: Tx_edge ≥ i → Ty_edge ≤ K-i
```

**SAT Clause Translation**:
```
# Tx ≥ i → Ty ≤ K-i equivalent to ¬Tx[i-1] ∨ ¬Ty[K-i]
¬Tx_edge[i-1] ∨ ¬Ty_edge[K-i]
```

**Implementation Framework**:
```python
# Distance coupling implications
for i in range(1, K + 1):
    if K - i >= 0:
        tx_geq_i = None
        ty_leq_ki = None
        
        if i-1 < len(Tx):
            tx_geq_i = Tx[i-1]  # Tx ≥ i
        
        if K-i < len(Ty):
            ty_leq_ki = -Ty[K-i]  # Ty ≤ K-i
        
        if tx_geq_i is not None and ty_leq_ki is not None:
            clauses.append([-tx_geq_i, ty_leq_ki])
```

**Concrete Example**: 
- Target bandwidth K = 5, threshold i = 3
- If Tx ≥ 3, then Ty ≤ 2
- Guarantees: Tx + Ty ≤ 3 + 2 = 5 ≤ K

**Complexity**: O(K × |E|) - K threshold values × edge count

---

## 4. NSC (NETWORK SEQUENTIAL COUNTER) CONSTRAINTS

NSC provides an efficient encoding technique for cardinality constraints (counting True variables) with linear clause overhead.

### 4.1 Sequential Counter Foundation

The base sequential counter creates auxiliary variables to track cumulative counts across variable sequences.

**Source**: `nsc_encoder.py` - `_base_sequential_counter()`

**Auxiliary Variable Schema**: 
- `R[i,j]`: "Among the first i variables {x_1, ..., x_i}, at least j variables are True"
- Domain: `R[i,j]` where `i ∈ {1..n-1}` and `j ∈ {1..min(i,k)}`

**Implementation Framework**:
```python
def _base_sequential_counter(variables, k, vpool):
    n = len(variables)
    R = {}
    clauses = []
    
    # Generate auxiliary variables R[i,j]
    for i in range(1, n):  # Excludes R[n,j] for efficiency
        for j in range(1, min(i, k) + 1):
            R[i, j] = vpool.id(f'R_group{group_id}_{i}_{j}')
```

### 4.2 Count Initiation Rules (Formula 1)

Establishes the connection between input variables and the counting mechanism.

**Mathematical Formulation**:
```
∀i ∈ {1..n-1}: X_i → R_{i,1}
```

**SAT Clause Translation**:
```
¬X_i ∨ R_{i,1}
```

**Implementation**:
```python
# FORMULA (1): Count initiation
for i in range(1, n):
    xi = variables[i - 1]
    clauses.append([-xi, R[i, 1]])
```

### 4.3 Count Propagation Rules (Formula 2)

Ensures count information flows forward through the variable sequence.

**Mathematical Formulation**:
```
∀i ∈ {2..n-1}, ∀j ∈ {1..min(i-1,k)}: R_{i-1,j} → R_{i,j}
```

**SAT Clause Translation**:
```
¬R_{i-1,j} ∨ R_{i,j}
```

**Implementation**:
```python
# FORMULA (2): Count propagation mechanism
for i in range(2, n):
    for j in range(1, min(i, k) + 1):
        if j <= i - 1:
            clauses.append([-R[i - 1, j], R[i, j]])
```

### 4.4 Count Increment Rules (Formula 3)

Handles count incrementation when encountering True variables in the sequence.

**Mathematical Formulation**:
```
∀i ∈ {2..n-1}, ∀j ∈ {2..min(i,k)}: (X_i ∧ R_{i-1,j-1}) → R_{i,j}
```

**SAT Clause Translation**:
```
¬X_i ∨ ¬R_{i-1,j-1} ∨ R_{i,j}
```

**Implementation**:
```python
# FORMULA (3): Count increment logic
for i in range(2, n):
    xi = variables[i - 1]
    for j in range(2, min(i, k) + 1):
        if j - 1 <= i - 1:
            clauses.append([-xi, -R[i - 1, j - 1], R[i, j]])
```

### 4.5 Count Suppression Rules (Formulas 4, 5, 6)

Ensures count variables are not incorrectly activated through comprehensive suppression constraints.

**Formula 4 - Base Suppression**:
```
∀i ∈ {2..n-1}, ∀j ∈ {1..min(i-1,k)}: (¬X_i ∧ ¬R_{i-1,j}) → ¬R_{i,j}
```

**SAT Translation**:
```
X_i ∨ R_{i-1,j} ∨ ¬R_{i,j}
```

**Formula 5 - Threshold Suppression**:
```
∀i ∈ {1..min(n,k)}: ¬X_i → ¬R_{i,i}
```

**SAT Translation**:
```
X_i ∨ ¬R_{i,i}
```

**Formula 6 - Transitional Suppression**:
```
∀i ∈ {2..n-1}, ∀j ∈ {2..min(i,k)}: ¬R_{i-1,j-1} → ¬R_{i,j}
```

**SAT Translation**:
```
R_{i-1,j-1} ∨ ¬R_{i,j}
```

**Implementation**:
```python
# FORMULA (4): Base suppression mechanism
for i in range(2, n):
    xi = variables[i - 1]
    for j in range(1, min(i, k) + 1):
        if j <= i - 1:
            clauses.append([xi, R[i - 1, j], -R[i, j]])

# FORMULA (5): Threshold-based suppression
for i in range(1, min(n, k + 1)):
    xi = variables[i - 1]
    if i <= k and (i, i) in R:
        clauses.append([xi, -R[i, i]])

# FORMULA (6): Transitional suppression
for i in range(2, n):
    for j in range(2, min(i, k) + 1):
        if (i - 1, j - 1) in R and (i, j) in R:
            clauses.append([R[i - 1, j - 1], -R[i, j]])
```

### 4.6 At-Least-K Final Constraints (Formula 7)

Enforces the lower bound requirement for cardinality constraints.

**Mathematical Formulation**:
```
R_{n-1,k} ∨ (X_n ∧ R_{n-1,k-1})
```

**CNF Conversion**:
```
# Disjunctive normal form to conjunctive normal form
R_{n-1,k} ∨ X_n
R_{n-1,k} ∨ R_{n-1,k-1}
```

**Implementation**:
```python
def encode_nsc_at_least_k(variables, k, vpool):
    clauses, R = _base_sequential_counter(variables, k, vpool)
    
    xn = variables[n - 1]
    if k == 1:
        clauses.append([R[n - 1, 1], xn])
    else:
        if (n - 1, k) in R:
            if (n - 1, k - 1) in R:
                clauses.append([R[n - 1, k], xn])
                clauses.append([R[n - 1, k], R[n - 1, k - 1]])
    return clauses
```

### 4.7 At-Most-K Prevention Constraints (Formula 8)

Enforces the upper bound requirement by preventing count overflow.

**Mathematical Formulation**:
```
∀i ∈ {k+1..n}: X_i → ¬R_{i-1,k}
```

**SAT Clause Translation**:
```
¬X_i ∨ ¬R_{i-1,k}
```

**Implementation**:
```python
def encode_nsc_at_most_k(variables, k, vpool):
    clauses, R = _base_sequential_counter(variables, k, vpool)
    
    # FORMULA (8): Count overflow prevention
    for i in range(k + 1, n + 1):
        xi = variables[i - 1]
        if (i - 1, k) in R:
            clauses.append([-xi, -R[i - 1, k]])
    return clauses
```

### 4.8 Exactly-K Constraint Composition

Combines at-most-k and at-least-k constraints to achieve exact cardinality.

**Implementation**:
```python
def encode_nsc_exactly_k(variables, k, vpool):
    clauses_at_most = encode_nsc_at_most_k(variables, k, vpool)
    clauses_at_least = encode_nsc_at_least_k(variables, k, vpool)
    return clauses_at_most + clauses_at_least
```

**NSC Complexity**: O(n × k) auxiliary variables, O(n × k) clauses per constraint

---

## Computational Complexity Analysis

| Constraint Type | Space Complexity | Time Complexity | Notes |
|-----------------|------------------|-----------------|-------|
| Position Constraints | O(n³) | O(n³) | n² grid positions × n vertices |
| Distance Constraints | O(n² × \|E\|) | O(n² × \|E\|) | n² encoding per edge × edge count |
| Bandwidth Constraints | O(K × \|E\|) | O(K × \|E\|) | K bandwidth levels × edge count |
| NSC Constraints | O(n²) | O(n²) | Per constraint instance |
| **Overall System** | **O(n³ + n² × \|E\| + K × \|E\|)** | **O(n³ + n² × \|E\| + K × \|E\|)** | **Combined complexity** |

## Conclusion

This constraint encoding framework demonstrates a sophisticated approach to SAT-based optimization through four integrated subsystems:

1. **Position Constraints**: Establish valid vertex-to-grid mappings with uniqueness guarantees
2. **Distance Constraints**: Provide precise Manhattan distance representations using thermometer encoding  
3. **Bandwidth Constraints**: Implement optimization objectives through distance coupling mechanisms
4. **NSC Constraints**: Enable efficient cardinality constraint encoding with linear overhead

The synergy between these constraint types produces a complete SAT formulation capable of finding optimal solutions to the 2D bandwidth minimization problem while maintaining computational tractability.
