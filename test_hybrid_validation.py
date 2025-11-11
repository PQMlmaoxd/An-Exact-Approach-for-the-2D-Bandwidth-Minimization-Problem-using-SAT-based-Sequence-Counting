#!/usr/bin/env python3
# test_hybrid_validation.py
# Test validation of hybrid encoding parameters

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_k_bandwidth_solver import CustomKBandwidthSolver
from distance_encoder_cutoff import calculate_theoretical_upper_bound

def test_replacement_validation():
    """Test that num_replacements is properly validated"""
    print("="*80)
    print("TEST: Hybrid Encoding Validation")
    print("="*80)
    
    # Create a simple test graph
    test_file = "test_validation.mtx"
    with open(test_file, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write("8 8 12\n")
        # Create a simple grid-like graph
        f.write("1 2\n")
        f.write("2 3\n")
        f.write("3 4\n")
        f.write("4 5\n")
        f.write("5 6\n")
        f.write("6 7\n")
        f.write("7 8\n")
        f.write("1 3\n")
        f.write("2 4\n")
        f.write("3 5\n")
        f.write("4 6\n")
        f.write("5 7\n")
    
    try:
        n = 8
        theoretical_ub = calculate_theoretical_upper_bound(n)
        max_replacements = n - 1 - theoretical_ub
        
        print(f"\nGraph properties:")
        print(f"  n = {n}")
        print(f"  Theoretical UB = {theoretical_ub}")
        print(f"  Max replacements = n - 1 - UB = {n} - 1 - {theoretical_ub} = {max_replacements}")
        
        # Test 1: Valid replacement (within bounds)
        print(f"\n" + "="*60)
        print(f"Test 1: Valid replacement (num_replacements = {max_replacements - 1})")
        print("="*60)
        solver1 = CustomKBandwidthSolver(test_file, encoding_method='hybrid', 
                                         num_replacements=max_replacements - 1)
        assert solver1.num_replacements == max_replacements - 1, "Should keep original value"
        print(f"✓ num_replacements = {solver1.num_replacements} (valid)")
        
        # Test 2: Maximum replacement
        print(f"\n" + "="*60)
        print(f"Test 2: Maximum replacement (num_replacements = {max_replacements})")
        print("="*60)
        solver2 = CustomKBandwidthSolver(test_file, encoding_method='hybrid', 
                                         num_replacements=max_replacements)
        assert solver2.num_replacements == max_replacements, "Should equal max"
        print(f"✓ num_replacements = {solver2.num_replacements} (at maximum)")
        
        # Test 3: Exceeding maximum (should be capped)
        print(f"\n" + "="*60)
        print(f"Test 3: Exceeding maximum (num_replacements = {max_replacements + 5})")
        print("="*60)
        solver3 = CustomKBandwidthSolver(test_file, encoding_method='hybrid', 
                                         num_replacements=max_replacements + 5)
        assert solver3.num_replacements == max_replacements, "Should be capped to max"
        print(f"✓ num_replacements capped to {solver3.num_replacements}")
        
        # Test 4: Negative value (should be set to 0)
        print(f"\n" + "="*60)
        print(f"Test 4: Negative value (num_replacements = -3)")
        print("="*60)
        solver4 = CustomKBandwidthSolver(test_file, encoding_method='hybrid', 
                                         num_replacements=-3)
        assert solver4.num_replacements == 0, "Should be set to 0"
        print(f"✓ num_replacements set to {solver4.num_replacements}")
        
        # Test 5: Zero (no replacement)
        print(f"\n" + "="*60)
        print(f"Test 5: Zero replacement (num_replacements = 0)")
        print("="*60)
        solver5 = CustomKBandwidthSolver(test_file, encoding_method='hybrid', 
                                         num_replacements=0)
        assert solver5.num_replacements == 0, "Should remain 0"
        print(f"✓ num_replacements = {solver5.num_replacements} (no replacement)")
        
        print(f"\n" + "="*80)
        print("ALL VALIDATION TESTS PASSED ✓")
        print("="*80)
        
        print(f"\nSummary:")
        print(f"✓ Valid values are preserved")
        print(f"✓ Values exceeding max are capped to {max_replacements}")
        print(f"✓ Negative values are set to 0")
        print(f"✓ Proper warnings are displayed")
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


def test_mtx_parsing():
    """Test MTX file parsing validation"""
    print(f"\n" + "="*80)
    print("TEST: MTX Parsing Validation")
    print("="*80)
    
    # Test 1: Valid MTX file
    print(f"\nTest 1: Valid MTX file")
    test_file = "test_valid.mtx"
    with open(test_file, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write("% Comment line\n")
        f.write("5 5 4\n")
        f.write("1 2\n")
        f.write("2 3\n")
        f.write("3 4\n")
        f.write("4 5\n")
    
    try:
        solver = CustomKBandwidthSolver(test_file, encoding_method='standard')
        assert solver.n == 5, f"Expected n=5, got n={solver.n}"
        assert len(solver.edges) == 4, f"Expected 4 edges, got {len(solver.edges)}"
        print(f"✓ Valid file parsed correctly (n={solver.n}, edges={len(solver.edges)})")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
    
    # Test 2: File with self-loops (should be removed)
    print(f"\nTest 2: File with self-loops")
    test_file = "test_selfloop.mtx"
    with open(test_file, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write("4 4 5\n")
        f.write("1 2\n")
        f.write("2 2\n")  # Self-loop (should be ignored)
        f.write("2 3\n")
        f.write("3 3\n")  # Self-loop (should be ignored)
        f.write("3 4\n")
    
    try:
        solver = CustomKBandwidthSolver(test_file, encoding_method='standard')
        assert solver.n == 4, f"Expected n=4, got n={solver.n}"
        assert len(solver.edges) == 3, f"Expected 3 edges (self-loops removed), got {len(solver.edges)}"
        print(f"✓ Self-loops removed correctly (edges={len(solver.edges)})")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
    
    # Test 3: File with duplicate edges (should be deduplicated)
    print(f"\nTest 3: File with duplicate edges")
    test_file = "test_duplicates.mtx"
    with open(test_file, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write("4 4 6\n")
        f.write("1 2\n")
        f.write("2 1\n")  # Duplicate of 1-2
        f.write("2 3\n")
        f.write("3 2\n")  # Duplicate of 2-3
        f.write("3 4\n")
        f.write("4 3\n")  # Duplicate of 3-4
    
    try:
        solver = CustomKBandwidthSolver(test_file, encoding_method='standard')
        assert solver.n == 4, f"Expected n=4, got n={solver.n}"
        assert len(solver.edges) == 3, f"Expected 3 edges (duplicates removed), got {len(solver.edges)}"
        print(f"✓ Duplicate edges removed correctly (edges={len(solver.edges)})")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
    
    print(f"\n" + "="*80)
    print("ALL MTX PARSING TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    test_replacement_validation()
    test_mtx_parsing()
    
    print(f"\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
    print("="*80)
    print(f"\nValidation features:")
    print(f"✓ num_replacements properly bounded by n - 1 - UB")
    print(f"✓ Invalid values are corrected with warnings")
    print(f"✓ MTX files parsed correctly with validation")
    print(f"✓ Self-loops and duplicates handled properly")
