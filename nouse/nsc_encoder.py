# nsc_encoder.py

def _base_sequential_counter(variables, k, vpool):
    """
    Build sequential counter according to original NSC (code.txt).
    R[i, j] means: "among the first i variables {x_1, ..., x_i}, at least j variables are True"
    """
    if not variables:
        return [], {}
    if k < 0:
        return [[1], [-1]], {}

    n = len(variables)
    R = {}  # Dictionary to store auxiliary variables R_i,j
    clauses = []
    group_id = hash(str(variables))

    # Initialize auxiliary variables R_i,j for i from 1 to n-1, j from 1 to k
    for i in range(1, n):  # Only up to n-1, no R_n,j
        for j in range(1, min(i, k) + 1):
            R[i, j] = vpool.id(f'R_group{group_id}_{i}_{j}')

    # =================================================================
    # FORMULA (1): Push bit 1
    # For each i from 1 to n-1: (X_i) --> (R_i,1)
    # =================================================================
    for i in range(1, n):
        xi = variables[i - 1]
        clauses.append([-xi, R[i, 1]])

    # =================================================================
    # FORMULA (2): Previous count bit implies next count bit
    # For each i from 2 to n-1, j from 1 to min(i-1, k): (R_{i-1,j}) --> (R_i,j)
    # =================================================================
    for i in range(2, n):
        for j in range(1, min(i, k) + 1):
            if j <= i - 1:  # Ensure R[i-1, j] exists
                clauses.append([-R[i - 1, j], R[i, j]])

    # =================================================================
    # FORMULA (3): Adding one TRUE variable increases counter
    # For each i from 2 to n-1, j from 2 to min(i, k): (X_i AND R_{i-1,j-1}) --> (R_i,j)
    # =================================================================
    for i in range(2, n):
        xi = variables[i - 1]
        for j in range(2, min(i, k) + 1):
            if j - 1 <= i - 1:  # Ensure R[i-1, j-1] exists
                clauses.append([-xi, -R[i - 1, j - 1], R[i, j]])

    # =================================================================
    # FORMULA (4): Push bit 0 - Base condition
    # For each i from 2 to n-1, j from 1 to min(i-1, k): (NOT X_i AND NOT R_{i-1,j}) --> (NOT R_i,j)
    # =================================================================
    for i in range(2, n):
        xi = variables[i - 1]
        for j in range(1, min(i, k) + 1):
            if j <= i - 1:  # Ensure R[i-1, j] exists
                clauses.append([xi, R[i - 1, j], -R[i, j]])

    # =================================================================
    # FORMULA (5): Push bit 0 - Threshold
    # For each i from 1 to k: (NOT X_i) --> (NOT R_i,i)
    # =================================================================
    for i in range(1, min(n, k + 1)):
        xi = variables[i - 1]
        if i <= k and (i, i) in R:
            clauses.append([xi, -R[i, i]])

    # =================================================================
    # FORMULA (6): Push bit 0 - Transition
    # For each i from 2 to n-1, j from 2 to min(i, k): (NOT R_{i-1,j-1}) --> (NOT R_i,j)
    # =================================================================
    for i in range(2, n):
        for j in range(2, min(i, k) + 1):
            if (i - 1, j - 1) in R and (i, j) in R:
                clauses.append([R[i - 1, j - 1], -R[i, j]])

    return clauses, R

def encode_nsc_at_least_k(variables, k, vpool):
    """Encode At-Least-K according to original NSC."""
    n = len(variables)
    if k <= 0: 
        return []
    if k > n: 
        return [[1], [-1]]

    clauses, R = _base_sequential_counter(variables, k, vpool)

    # =================================================================
    # FORMULA (7): Ensure final sum is at least k
    # (R_{n-1,k}) OR (X_n AND R_{n-1,k-1})
    # =================================================================
    xn = variables[n - 1]
    
    if k == 1:
        # Special case k=1: at least one variable must be True
        clauses.append([R[n - 1, 1], xn])
    else:
        # General case
        if (n - 1, k) in R:
            if (n - 1, k - 1) in R:
                # R_{n-1,k} OR (X_n AND R_{n-1,k-1})
                # Equivalent: R_{n-1,k} OR X_n, R_{n-1,k} OR R_{n-1,k-1}
                clauses.append([R[n - 1, k], xn])
                clauses.append([R[n - 1, k], R[n - 1, k - 1]])
            else:
                clauses.append([R[n - 1, k], xn])

    return clauses

def encode_nsc_at_most_k(variables, k, vpool):
    """Encode At-Most-K according to original NSC."""
    n = len(variables)
    if k < 0: 
        return [[1], [-1]]
    if k >= n: 
        return []

    clauses, R = _base_sequential_counter(variables, k, vpool)

    # =================================================================
    # FORMULA (8): Prevent counter from exceeding k
    # For each i from k+1 to n: (X_i) --> (NOT R_{i-1,k})
    # =================================================================
    for i in range(k + 1, n + 1):
        xi = variables[i - 1]
        if (i - 1, k) in R:
            clauses.append([-xi, -R[i - 1, k]])

    return clauses

def encode_nsc_exactly_k(variables, k, vpool):
    """Encode Exactly-K by combining At-Most-K and At-Least-K."""
    if k < 0 or k > len(variables):
        return [[1], [-1]]

    clauses_at_most = encode_nsc_at_most_k(variables, k, vpool)
    clauses_at_least = encode_nsc_at_least_k(variables, k, vpool)
    
    return clauses_at_most + clauses_at_least