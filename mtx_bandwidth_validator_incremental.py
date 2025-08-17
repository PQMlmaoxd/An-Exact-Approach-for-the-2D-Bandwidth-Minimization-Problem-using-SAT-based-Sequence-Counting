"""
mtx_bandwidth_validator_incremental.py
Incremental validator for a fixed K that uses the persistent/incremental solver
interface but applies the same thermometer bandwidth encoding and model
decoding as the simple non-incremental validator.

Usage: python mtx_bandwidth_validator_incremental.py <mtx_file> <solver> <K>
"""
import os
import sys
import time
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from incremental_bandwidth_solver import IncrementalBandwidthSolver


def parse_mtx_file(filename: str) -> Tuple[int, List[Tuple[int,int]]]:
    """Parse a MatrixMarket (.mtx) file and return n, edges list.

    Returns:
        n: number of vertices (max(rows, cols))
        edges: list of undirected edges as (u,v) with u<v
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    header_found = False
    edges_set = set()
    n = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('%'):
            continue
        if not header_found:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows, cols, nnz = map(int, parts[:3])
                    n = max(rows, cols)
                    header_found = True
                    continue
                except ValueError:
                    # malformed header; skip
                    continue

        # parse edge
        parts = line.split()
        if len(parts) >= 2:
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            edge = tuple(sorted((u, v)))
            if edge not in edges_set:
                edges_set.add(edge)

    edges = list(edges_set)
    return n, edges


def encode_thermometer_bandwidth_clauses(tx_vars: Dict[str, List[int]], ty_vars: Dict[str, List[int]], K: int) -> List[List[int]]:
    """Create thermometer bandwidth clauses identical to the simple validator.

    Args:
        tx_vars: mapping edge_id -> list of Tx variables (Tx[i] encodes Tx >= i+1)
        ty_vars: mapping edge_id -> list of Ty variables
        K: target bandwidth

    Returns: list of clauses (each clause is list[int])
    """
    clauses: List[List[int]] = []

    for edge_id in tx_vars:
        Tx = tx_vars[edge_id]
        Ty = ty_vars.get(edge_id, [])

        # Tx <= K  -> not Tx >= K+1  => -Tx[K]
        if K < len(Tx):
            clauses.append([-Tx[K]])

        # Ty <= K
        if K < len(Ty):
            clauses.append([-Ty[K]])

        # Implications: Tx >= i -> Ty <= K-i  encoded as (-Tx[i-1] or -Ty[K-i])
        for i in range(1, K + 1):
            tx_geq_i = None
            ty_leq_ki = None
            if i - 1 < len(Tx):
                tx_geq_i = Tx[i - 1]
            if (K - i) < len(Ty):
                ty_leq_ki = -Ty[K - i]
            if tx_geq_i is not None and ty_leq_ki is not None:
                clauses.append([-tx_geq_i, ty_leq_ki])

    return clauses


def extract_and_verify_from_model(model: List[int], solver: IncrementalBandwidthSolver, K: int) -> Dict:
    """Decode model (using solver's X/Y vars) and verify bandwidth <= K.

    Returns a dict similar to the simple validator's solution_info.
    """
    if not model:
        return {}

    posset = {lit for lit in model if lit > 0}
    n = solver.n
    positions: Dict[int, Tuple[int,int]] = {}
    violations = []

    for v in range(1, n + 1):
        Xrow = solver.X_vars.get(v, [])
        Yrow = solver.Y_vars.get(v, [])

        xs = [i for i, var in enumerate(Xrow, start=1) if var in posset]
        ys = [i for i, var in enumerate(Yrow, start=1) if var in posset]

        if len(xs) != 1 or len(ys) != 1:
            violations.append((v, xs, ys))
        else:
            positions[v] = (xs[0], ys[0])

    if violations:
        return {
            'positions': positions,
            'actual_bandwidth': None,
            'constraint_K': K,
            'is_valid': False,
            'edge_distances': [],
            'reason': 'positions_not_exactly_one'
        }

    max_distance = 0
    edge_distances = []
    for u, v in solver.edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        d = abs(x1 - x2) + abs(y1 - y2)
        edge_distances.append((u, v, d))
        if d > max_distance:
            max_distance = d

    is_valid = (max_distance <= K)

    return {
        'positions': positions,
        'actual_bandwidth': max_distance,
        'constraint_K': K,
        'is_valid': is_valid,
        'edge_distances': edge_distances
    }


def incremental_validate(mtx_file: str, solver_type: str, K: int) -> Dict:
    print(f"INCREMENTAL VALIDATOR: file={mtx_file}, solver={solver_type}, K={K}")

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
            f"mtx/regular/{mtx_file}.mtx",
        ]
        found = None
        for p in search_paths:
            if os.path.exists(p):
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"MTX file not found: {mtx_file}")
        mtx_file = found

    n, edges = parse_mtx_file(mtx_file)
    print(f"Parsed: n={n}, edges={len(edges)}")

    solver = IncrementalBandwidthSolver(n, solver_type)
    solver.set_graph_edges(edges)
    solver.create_position_variables()
    solver.create_distance_variables()

    # Initialize persistent solver and add base constraints
    solver._initialize_persistent_solver()

    # Build bandwidth clauses (using same thermometer logic as validator)
    bandwidth_clauses = encode_thermometer_bandwidth_clauses(solver.Tx_vars, solver.Ty_vars, K)
    print(f"Adding {len(bandwidth_clauses)} bandwidth clauses for K={K}")
    for c in bandwidth_clauses:
        solver.persistent_solver.add_clause(c)

    # Solve
    t0 = time.time()
    is_sat = solver.persistent_solver.solve()
    solve_time = time.time() - t0

    model = None
    solution_info = None
    if is_sat:
        model = solver.persistent_solver.get_model()
        solution_info = extract_and_verify_from_model(model, solver, K)

    # Clean up
    solver.cleanup_solver()

    return {
        'is_sat': is_sat,
        'solve_time': solve_time,
        'model': model,
        'solution_info': solution_info
    }


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python mtx_bandwidth_validator_incremental.py <mtx_file> <solver> <K>")
        sys.exit(1)

    mtx_file = sys.argv[1]
    solver_type = sys.argv[2]
    try:
        K = int(sys.argv[3])
    except ValueError:
        print("K must be an integer")
        sys.exit(1)

    res = incremental_validate(mtx_file, solver_type, K)
    print("\nRESULT:")
    print(f"  is_sat: {res['is_sat']}")
    print(f"  solve_time: {res['solve_time']:.3f}s")

    if res['is_sat']:
        info = res.get('solution_info')
        if info is None:
            print("  Note: SAT model found but no solution info available")
        else:
            # Mirror the detailed reporting from the simple validator
            positions = info.get('positions', {})
            edge_distances = info.get('edge_distances', [])
            actual_bw = info.get('actual_bandwidth')
            is_valid = info.get('is_valid')

            print("")
            print("Solution verification")
            print("-" * 50)
            print(f"Extracted positions: {len(positions)} vertices")

            # Show vertex positions (first 20)
            max_show = min(20, len(positions))
            for v in sorted(list(positions.keys())[:max_show]):
                x, y = positions[v]
                print(f"  v{v}: ({x}, {y})")
            if len(positions) > max_show:
                print(f"  ... and {len(positions) - max_show} more vertices")

            # Show edge distances
            print(f"\nEdge distances ({len(edge_distances)} edges):")
            for u, v, distance in edge_distances:
                marker = "ok" if distance <= K else "exceeds"
                print(f"  ({u},{v}): {distance} [{marker}]")

            print(f"\nBandwidth summary:")
            print(f"  Actual: {actual_bw}")
            print(f"  Limit:  {K}")
            print(f"  Valid:  {'Yes' if is_valid else 'No'}")
            print(f"  Edges within limit: {sum(1 for _, _, d in edge_distances if d <= K)}/{len(edge_distances)}")

    sys.exit(0 if res['is_sat'] else 2)
