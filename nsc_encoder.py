# nsc_encoder.py

def _base_sequential_counter(variables, k, vpool):
    """
    Xây dựng bộ đếm tuần tự theo đúng NSC gốc (code.txt).
    R[i, j] nghĩa là: "trong số i biến đầu tiên {x_1, ..., x_i}, có ít nhất j biến là True"
    """
    if not variables:
        return [], {}
    if k < 0:
        return [[1], [-1]], {}

    n = len(variables)
    R = {}  # Dictionary để lưu các biến phụ R_i,j
    clauses = []
    group_id = hash(str(variables))

    # Khởi tạo các biến phụ R_i,j cho i từ 1 đến n-1, j từ 1 đến k
    for i in range(1, n):  # Chỉ đến n-1, không có R_n,j
        for j in range(1, min(i, k) + 1):
            R[i, j] = vpool.id(f'R_group{group_id}_{i}_{j}')

    # =================================================================
    # CÔNG THỨC (1): Dồn bit 1
    # Với mỗi i từ 1 đến n-1: (X_i) --> (R_i,1)
    # =================================================================
    for i in range(1, n):
        xi = variables[i - 1]
        clauses.append([-xi, R[i, 1]])

    # =================================================================
    # CÔNG THỨC (2): Bit đếm trước kéo theo bit đếm sau
    # Với mỗi i từ 2 đến n-1, j từ 1 đến min(i-1, k): (R_{i-1,j}) --> (R_i,j)
    # =================================================================
    for i in range(2, n):
        for j in range(1, min(i, k) + 1):
            if j <= i - 1:  # Đảm bảo R[i-1, j] tồn tại
                clauses.append([-R[i - 1, j], R[i, j]])

    # =================================================================
    # CÔNG THỨC (3): Thêm một biến TRUE sẽ tăng bộ đếm
    # Với mỗi i từ 2 đến n-1, j từ 2 đến min(i, k): (X_i AND R_{i-1,j-1}) --> (R_i,j)
    # =================================================================
    for i in range(2, n):
        xi = variables[i - 1]
        for j in range(2, min(i, k) + 1):
            if j - 1 <= i - 1:  # Đảm bảo R[i-1, j-1] tồn tại
                clauses.append([-xi, -R[i - 1, j - 1], R[i, j]])

    # =================================================================
    # CÔNG THỨC (4): Dồn bit 0 - Điều kiện cơ sở
    # Với mỗi i từ 2 đến n-1, j từ 1 đến min(i-1, k): (NOT X_i AND NOT R_{i-1,j}) --> (NOT R_i,j)
    # =================================================================
    for i in range(2, n):
        xi = variables[i - 1]
        for j in range(1, min(i, k) + 1):
            if j <= i - 1:  # Đảm bảo R[i-1, j] tồn tại
                clauses.append([xi, R[i - 1, j], -R[i, j]])

    # =================================================================
    # CÔNG THỨC (5): Dồn bit 0 - Ngưỡng
    # Với mỗi i từ 1 đến k: (NOT X_i) --> (NOT R_i,i)
    # =================================================================
    for i in range(1, min(n, k + 1)):
        xi = variables[i - 1]
        if i <= k and (i, i) in R:
            clauses.append([xi, -R[i, i]])

    # =================================================================
    # CÔNG THỨC (6): Dồn bit 0 - Chuyển tiếp
    # Với mỗi i từ 2 đến n-1, j từ 2 đến min(i, k): (NOT R_{i-1,j-1}) --> (NOT R_i,j)
    # =================================================================
    for i in range(2, n):
        for j in range(2, min(i, k) + 1):
            if (i - 1, j - 1) in R and (i, j) in R:
                clauses.append([R[i - 1, j - 1], -R[i, j]])

    return clauses, R

def encode_nsc_at_least_k(variables, k, vpool):
    """Mã hóa At-Least-K theo NSC gốc."""
    n = len(variables)
    if k <= 0: 
        return []
    if k > n: 
        return [[1], [-1]]

    clauses, R = _base_sequential_counter(variables, k, vpool)

    # =================================================================
    # CÔNG THỨC (7): Đảm bảo tổng cuối cùng ít nhất là k
    # (R_{n-1,k}) OR (X_n AND R_{n-1,k-1})
    # =================================================================
    xn = variables[n - 1]
    
    if k == 1:
        # Trường hợp đặc biệt k=1: ít nhất một biến phải True
        clauses.append([R[n - 1, 1], xn])
    else:
        # Trường hợp tổng quát
        if (n - 1, k) in R:
            if (n - 1, k - 1) in R:
                # R_{n-1,k} OR (X_n AND R_{n-1,k-1})
                # Tương đương: R_{n-1,k} OR X_n, R_{n-1,k} OR R_{n-1,k-1}
                clauses.append([R[n - 1, k], xn])
                clauses.append([R[n - 1, k], R[n - 1, k - 1]])
            else:
                clauses.append([R[n - 1, k], xn])

    return clauses

def encode_nsc_at_most_k(variables, k, vpool):
    """Mã hóa At-Most-K theo NSC gốc."""
    n = len(variables)
    if k < 0: 
        return [[1], [-1]]
    if k >= n: 
        return []

    clauses, R = _base_sequential_counter(variables, k, vpool)

    # =================================================================
    # CÔNG THỨC (8): Ngăn bộ đếm vượt quá k
    # Với mỗi i từ k+1 đến n: (X_i) --> (NOT R_{i-1,k})
    # =================================================================
    for i in range(k + 1, n + 1):
        xi = variables[i - 1]
        if (i - 1, k) in R:
            clauses.append([-xi, -R[i - 1, k]])

    return clauses

def encode_nsc_exactly_k(variables, k, vpool):
    """Mã hóa Exactly-K bằng cách kết hợp At-Most-K và At-Least-K."""
    if k < 0 or k > len(variables):
        return [[1], [-1]]

    clauses_at_most = encode_nsc_at_most_k(variables, k, vpool)
    clauses_at_least = encode_nsc_at_least_k(variables, k, vpool)
    
    return clauses_at_most + clauses_at_least