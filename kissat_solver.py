# kissat_solver.py
# Integration module for Kissat SAT solver with our bandwidth optimization solver

import subprocess
import tempfile
import os
import time
from pathlib import Path

class KissatSolver:
    """
    Wrapper class for Kissat SAT solver
    Provides interface compatible with pysat solvers
    """
    
    def __init__(self, kissat_path="./bin/kissat"):
        """
        Initialize Kissat solver wrapper
        
        Args:
            kissat_path: Path to kissat executable
        """
        self.kissat_path = kissat_path
        self.clauses = []
        self.num_vars = 0
        self.temp_dir = None
        self.cnf_file = None
        self.result_file = None
        
        # Verify kissat is available
        if not os.path.exists(kissat_path):
            raise FileNotFoundError(f"Kissat executable not found at: {kissat_path}")
        
        # Test kissat
        try:
            result = subprocess.run([kissat_path, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            print(f"Kissat version: {result.stdout.strip()}")
        except Exception as e:
            raise RuntimeError(f"Failed to run kissat: {e}")
    
    def add_clause(self, clause):
        """
        Add a clause to the solver
        
        Args:
            clause: List of literals (positive/negative integers)
        """
        self.clauses.append(clause)
        
        # Update number of variables
        for lit in clause:
            var = abs(lit)
            if var > self.num_vars:
                self.num_vars = var
    
    def solve(self, timeout=300):
        """
        Solve the SAT instance using kissat
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            bool: True if SAT, False if UNSAT, None if timeout/error
        """
        if not self.clauses:
            return True  # Empty formula is SAT
        
        # Create temporary files
        self.temp_dir = tempfile.mkdtemp()
        self.cnf_file = os.path.join(self.temp_dir, "formula.cnf")
        self.result_file = os.path.join(self.temp_dir, "result.out")
        
        try:
            # Write CNF file
            self._write_cnf_file()
            
            # Run kissat
            start_time = time.time()
            result = self._run_kissat(timeout)
            solve_time = time.time() - start_time
            
            print(f"Kissat solve time: {solve_time:.3f}s")
            
            return result
            
        finally:
            # Cleanup temporary files
            self._cleanup()
    
    def get_model(self):
        """
        Get the satisfying assignment (if available)
        
        Returns:
            list: List of literals representing the model
        """
        if not self.result_file or not os.path.exists(self.result_file):
            return None
        
        try:
            with open(self.result_file, 'r') as f:
                content = f.read()
            
            model = []
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('v '):
                    # Parse variable assignment line: "v -1 2 -3 0"
                    literals = line[2:].strip().split()
                    for lit in literals:
                        if lit != '0':  # 0 terminates the assignment
                            model.append(int(lit))
            
            return model if model else None
        except Exception as e:
            print(f"Error reading model: {e}")
            return None
    
    def delete(self):
        """
        Clean up resources
        """
        self._cleanup()
    
    def _write_cnf_file(self):
        """
        Write the CNF formula to file in DIMACS format
        """
        with open(self.cnf_file, 'w') as f:
            # Header: p cnf <num_vars> <num_clauses>
            f.write(f"p cnf {self.num_vars} {len(self.clauses)}\n")
            
            # Write each clause
            for clause in self.clauses:
                clause_str = ' '.join(map(str, clause)) + ' 0\n'
                f.write(clause_str)
        
        print(f"CNF file written: {self.cnf_file}")
        print(f"Variables: {self.num_vars}, Clauses: {len(self.clauses)}")
    
    def _run_kissat(self, timeout):
        """
        Run kissat on the CNF file
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            bool: True if SAT, False if UNSAT, None if timeout/error
        """
        cmd = [self.kissat_path, self.cnf_file]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            
            # Run kissat with timeout
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=timeout)
            
            # Save output for model extraction
            with open(self.result_file, 'w') as f:
                f.write(result.stdout)
            
            # Parse result
            if result.returncode == 10:  # SAT
                print("Kissat result: SAT")
                return True
            elif result.returncode == 20:  # UNSAT
                print("Kissat result: UNSAT")
                return False
            else:
                print(f"Kissat unexpected return code: {result.returncode}")
                print(f"Stdout: {result.stdout}")
                print(f"Stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"Kissat timeout after {timeout}s")
            return None
        except Exception as e:
            print(f"Error running kissat: {e}")
            return None
    
    def _cleanup(self):
        """
        Clean up temporary files
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup temp dir: {e}")

def test_kissat_solver():
    """
    Test kissat solver with simple SAT instances
    """
    print("=== TESTING KISSAT SOLVER ===")
    
    # Test 1: Simple SAT instance
    print("\nTest 1: Simple SAT instance")
    solver = KissatSolver()
    
    # Formula: (1 ∨ 2) ∧ (¬1 ∨ 3) ∧ (¬2 ∨ ¬3)
    solver.add_clause([1, 2])
    solver.add_clause([-1, 3])
    solver.add_clause([-2, -3])
    
    result = solver.solve(timeout=10)
    
    if result:
        print("✓ SAT instance solved correctly")
        model = solver.get_model()
        if model:
            print(f"Model: {model[:10]}...")  # Show first 10 literals
        else:
            print("Model: None (could not extract)")
    elif result is False:
        print("✗ UNSAT result")
    else:
        print("? Timeout or error")
    
    solver.delete()
    
    # Test 2: UNSAT instance
    print("\nTest 2: UNSAT instance")
    solver2 = KissatSolver()
    
    # Formula: (1) ∧ (¬1)
    solver2.add_clause([1])
    solver2.add_clause([-1])
    
    result2 = solver2.solve(timeout=10)
    
    if result2 is False:
        print("✓ UNSAT instance detected correctly")
    elif result2:
        print("✗ Should be UNSAT but got SAT")
    else:
        print("? Timeout or error")
    
    solver2.delete()
    
    print("\nKissat integration test completed!")

if __name__ == '__main__':
    test_kissat_solver()
