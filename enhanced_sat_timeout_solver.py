# enhanced_sat_timeout_solver.py
# Enhanced SAT Timeout Solver with Force-Kill Capabilities
# Addresses the "C++ process không dừng được" issue

import multiprocessing
import threading
import time
import os
import signal
import psutil
import sys
import traceback
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# Import SAT solvers
try:
    from pysat.solvers import Glucose42, Cadical195, Solver
    from pysat.formula import IDPool
except ImportError as e:
    print(f"PySAT import error: {e}")
    raise


@dataclass
class EnhancedSATResult:
    """Enhanced SAT solving result with detailed process information"""
    problem_id: str
    status: str  # 'SAT', 'UNSAT', 'TIMEOUT', 'KILLED', 'ERROR'
    model: Optional[List[int]] = None
    solve_time: Optional[float] = None
    total_time: float = 0.0
    num_clauses: int = 0
    num_variables: int = 0
    
    # Process tracking
    process_id: Optional[int] = None
    process_killed: bool = False
    kill_method: Optional[str] = None  # 'terminate', 'kill', 'force_kill'
    
    # Error handling
    error_message: Optional[str] = None
    exception_type: Optional[str] = None
    
    # Additional metadata
    additional_data: Optional[Dict[str, Any]] = None


def enhanced_sat_worker(clauses: List[List[int]], 
                       problem_id: str, 
                       solver_type: str,
                       return_queue: multiprocessing.Queue,
                       additional_data: Optional[Dict] = None) -> None:
    """
    Enhanced SAT worker with robust error handling and process tracking
    
    This function runs in a separate process and can be forcefully terminated
    """
    result = EnhancedSATResult(
        problem_id=problem_id,
        status='ERROR',
        num_clauses=len(clauses) if clauses else 0,
        additional_data=additional_data,
        process_id=os.getpid()
    )
    
    start_time = time.time()
    
    try:
        print(f"Process {os.getpid()}: Starting SAT solve for {problem_id}")
        
        # Create solver
        if solver_type.lower() == 'glucose42':
            solver = Glucose42()
        elif solver_type.lower() == 'cadical195':
            solver = Cadical195()
        else:
            solver = Glucose42()  # Default fallback
        
        # Add clauses to solver
        if clauses:
            for clause in clauses:
                solver.add_clause(clause)
            
            # Count variables
            max_var = 0
            for clause in clauses:
                for lit in clause:
                    max_var = max(max_var, abs(lit))
            result.num_variables = max_var
        
        print(f"Process {os.getpid()}: Solving {len(clauses)} clauses, {result.num_variables} variables...")
        
        # Solve with timing
        solve_start = time.time()
        is_sat = solver.solve()
        solve_time = time.time() - solve_start
        
        # Process result
        if is_sat:
            result.status = 'SAT'
            result.model = solver.get_model()
            print(f"Process {os.getpid()}: SAT in {solve_time:.3f}s")
        else:
            result.status = 'UNSAT'
            print(f"Process {os.getpid()}: UNSAT in {solve_time:.3f}s")
        
        result.solve_time = solve_time
        result.total_time = time.time() - start_time
        
        # Cleanup solver
        solver.delete()
        
    except KeyboardInterrupt:
        result.status = 'KILLED'
        result.error_message = 'Process interrupted by signal'
        print(f"Process {os.getpid()}: Interrupted by signal")
        
    except Exception as e:
        result.status = 'ERROR'
        result.error_message = str(e)
        result.exception_type = type(e).__name__
        result.total_time = time.time() - start_time
        print(f"Process {os.getpid()}: Error - {e}")
        traceback.print_exc()
    
    finally:
        # Always try to send result back
        try:
            return_queue.put(result)
        except:
            print(f"Process {os.getpid()}: Failed to send result back")


class EnhancedProcessTimeoutSATSolver:
    """
    Enhanced SAT solver with robust process timeout and force-kill capabilities
    
    Key improvements over ProcessTimeoutSATSolver:
    1. Direct multiprocessing.Process control (not ProcessPoolExecutor)
    2. Multiple kill strategies: terminate() -> kill() -> OS force-kill
    3. Process tree killing (kills child processes too)
    4. Windows/Linux compatible process termination
    5. Resource cleanup and monitoring
    """
    
    def __init__(self, solver_type: str = 'glucose42', default_timeout: float = 30.0):
        self.solver_type = solver_type.lower()
        self.default_timeout = default_timeout
        
        # Process tracking
        self.active_processes: Dict[str, multiprocessing.Process] = {}
        self.process_start_times: Dict[str, float] = {}
        
        print(f"✓ Enhanced Process SAT Solver initialized")
        print(f"  - Solver: {self.solver_type.upper()}")
        print(f"  - Default timeout: {default_timeout}s")
        print(f"  - Force-kill: ENABLED")
        print(f"  - Platform: {sys.platform}")
    
    def _kill_process_tree(self, process: multiprocessing.Process, kill_method: str = 'terminate') -> bool:
        """
        Kill process and all its children using multiple strategies
        
        Returns: True if process was killed, False otherwise
        """
        if not process.is_alive():
            return True
        
        try:
            # Get process tree using psutil
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            
            print(f"Killing process tree: parent {process.pid}, {len(children)} children")
            
            # Strategy 1: Graceful termination
            if kill_method in ['terminate', 'graceful']:
                print(f"  -> Graceful termination...")
                for child in children:
                    try:
                        child.terminate()
                    except:
                        pass
                
                try:
                    parent.terminate()
                    process.join(timeout=2.0)  # Wait up to 2 seconds
                except:
                    pass
                
                if not process.is_alive():
                    print(f"  -> SUCCESS: Graceful termination")
                    return True
            
            # Strategy 2: Force kill
            if kill_method in ['kill', 'force']:
                print(f"  -> Force kill...")
                for child in children:
                    try:
                        child.kill()
                    except:
                        pass
                
                try:
                    parent.kill()
                    process.join(timeout=1.0)
                except:
                    pass
                
                if not process.is_alive():
                    print(f"  -> SUCCESS: Force kill")
                    return True
            
            # Strategy 3: OS-level force kill (Windows/Linux specific)
            if sys.platform.startswith('win'):
                # Windows: Use taskkill
                print(f"  -> Windows taskkill...")
                try:
                    os.system(f'taskkill /F /T /PID {process.pid}')
                    time.sleep(0.5)
                    if not process.is_alive():
                        print(f"  -> SUCCESS: Windows taskkill")
                        return True
                except:
                    pass
            else:
                # Linux/Unix: Use SIGKILL
                print(f"  -> Unix SIGKILL...")
                try:
                    os.kill(process.pid, signal.SIGKILL)
                    process.join(timeout=1.0)
                    if not process.is_alive():
                        print(f"  -> SUCCESS: Unix SIGKILL")
                        return True
                except:
                    pass
            
            print(f"  -> FAILED: Process {process.pid} still alive after all kill attempts")
            return False
            
        except psutil.NoSuchProcess:
            # Process already dead
            return True
        except Exception as e:
            print(f"  -> ERROR in kill_process_tree: {e}")
            return False
    
    def solve_single(self, 
                    clauses: List[List[int]], 
                    problem_id: str,
                    timeout: Optional[float] = None,
                    additional_data: Optional[Dict] = None) -> EnhancedSATResult:
        """
        Solve single SAT problem with enhanced timeout and force-kill
        """
        if timeout is None:
            timeout = self.default_timeout
        
        print(f"\nSolving {problem_id} with {timeout}s timeout...")
        print(f"Problem: {len(clauses)} clauses")
        
        # Create communication queue
        return_queue = multiprocessing.Queue()
        
        # Create and start process
        process = multiprocessing.Process(
            target=enhanced_sat_worker,
            args=(clauses, problem_id, self.solver_type, return_queue, additional_data)
        )
        
        start_time = time.time()
        process.start()
        
        # Track process
        self.active_processes[problem_id] = process
        self.process_start_times[problem_id] = start_time
        
        result = None
        process_killed = False
        kill_method = None
        
        try:
            # Wait for result with timeout
            while True:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                
                if remaining <= 0:
                    # Timeout occurred
                    print(f"  -> TIMEOUT after {elapsed:.1f}s")
                    
                    # Kill process using multiple strategies
                    print(f"  -> Killing process {process.pid}...")
                    
                    # Try terminate first
                    if self._kill_process_tree(process, 'terminate'):
                        kill_method = 'terminate'
                        process_killed = True
                    elif self._kill_process_tree(process, 'kill'):
                        kill_method = 'kill'
                        process_killed = True
                    else:
                        kill_method = 'failed'
                        process_killed = False
                        print(f"  -> WARNING: Could not kill process {process.pid}")
                    
                    break
                
                # Check if process finished
                if not process.is_alive():
                    break
                
                # Check for result with short timeout
                try:
                    if not return_queue.empty():
                        result = return_queue.get_nowait()
                        break
                except:
                    pass
                
                # Small sleep to avoid busy waiting
                time.sleep(0.1)
            
            # Get result if available
            if result is None and not return_queue.empty():
                try:
                    result = return_queue.get(timeout=1.0)
                except:
                    pass
            
            # Join process
            if process.is_alive():
                process.join(timeout=2.0)
                if process.is_alive():
                    print(f"  -> WARNING: Process {process.pid} still alive after join")
            
        except Exception as e:
            print(f"  -> ERROR in solve_single: {e}")
        
        finally:
            # Cleanup
            if problem_id in self.active_processes:
                del self.active_processes[problem_id]
            if problem_id in self.process_start_times:
                del self.process_start_times[problem_id]
            
            # Close queue
            try:
                return_queue.close()
            except:
                pass
        
        # Create result if we don't have one
        if result is None:
            total_time = time.time() - start_time
            result = EnhancedSATResult(
                problem_id=problem_id,
                status='TIMEOUT' if process_killed else 'ERROR',
                total_time=total_time,
                num_clauses=len(clauses) if clauses else 0,
                process_id=process.pid if process else None,
                process_killed=process_killed,
                kill_method=kill_method,
                error_message='Process timeout and termination' if process_killed else 'Unknown error',
                additional_data=additional_data
            )
        else:
            # Update result with kill information
            result.process_killed = process_killed
            result.kill_method = kill_method
        
        # Print result summary
        print(f"  -> Result: {result.status}")
        if result.total_time > 0:
            print(f"  -> Time: {result.total_time:.2f}s")
        if process_killed:
            print(f"  -> Process killed: {kill_method}")
        
        return result
    
    def cleanup_all_processes(self):
        """Force cleanup all active processes"""
        if not self.active_processes:
            return
        
        print(f"Cleaning up {len(self.active_processes)} active processes...")
        
        for problem_id, process in list(self.active_processes.items()):
            if process.is_alive():
                print(f"  -> Killing {problem_id} (PID {process.pid})")
                self._kill_process_tree(process, 'kill')
        
        self.active_processes.clear()
        self.process_start_times.clear()
        print("✓ Process cleanup complete")
    
    def get_active_processes_info(self) -> Dict[str, Dict]:
        """Get information about currently active processes"""
        info = {}
        current_time = time.time()
        
        for problem_id, process in self.active_processes.items():
            start_time = self.process_start_times.get(problem_id, current_time)
            info[problem_id] = {
                'pid': process.pid,
                'alive': process.is_alive(),
                'elapsed_time': current_time - start_time
            }
        
        return info
    
    def __del__(self):
        """Destructor: cleanup processes"""
        try:
            self.cleanup_all_processes()
        except:
            pass


def test_enhanced_timeout_functionality():
    """Test the enhanced timeout functionality"""
    print("=" * 60)
    print("ENHANCED SAT TIMEOUT SOLVER TEST")
    print("=" * 60)
    
    solver = EnhancedProcessTimeoutSATSolver(solver_type='glucose42', default_timeout=3.0)
    
    # Test 1: Simple SAT problem (should solve quickly)
    print("\nTest 1: Simple SAT problem")
    simple_clauses = [[1, 2], [-1, 2], [1, -2], [-1, -2]]  # UNSAT
    
    result1 = solver.solve_single(simple_clauses, "simple_test", timeout=10.0)
    print(f"Result: {result1.status} in {result1.total_time:.2f}s")
    
    # Test 2: Timeout test with large problem
    print("\nTest 2: Timeout test (should timeout and kill process)")
    
    # Create a large, difficult SAT problem
    large_clauses = []
    n_vars = 1000
    n_clauses = 10000
    
    import random
    random.seed(42)
    for i in range(n_clauses):
        clause = []
        for j in range(3):  # 3-SAT
            var = random.randint(1, n_vars)
            if random.random() < 0.5:
                var = -var
            clause.append(var)
        large_clauses.append(clause)
    
    result2 = solver.solve_single(large_clauses, "timeout_test", timeout=2.0)
    print(f"Result: {result2.status} in {result2.total_time:.2f}s")
    print(f"Process killed: {result2.process_killed}")
    print(f"Kill method: {result2.kill_method}")
    
    # Test 3: Check no zombie processes
    print("\nTest 3: Process cleanup verification")
    active_info = solver.get_active_processes_info()
    print(f"Active processes: {len(active_info)}")
    
    solver.cleanup_all_processes()
    print("✓ Enhanced timeout testing complete")


if __name__ == '__main__':
    """
    Enhanced SAT Timeout Solver with Force-Kill Capabilities
    
    This module addresses the fundamental issue: "Dù báo timeout nhưng vẫn không dừng process được"
    """
    
    # Ensure proper multiprocessing setup on Windows
    if sys.platform.startswith('win'):
        multiprocessing.set_start_method('spawn', force=True)
    
    try:
        test_enhanced_timeout_functionality()
    except KeyboardInterrupt:
        print("\n*** Interrupted by user ***")
    except Exception as e:
        print(f"*** Error: {e} ***")
        traceback.print_exc()
