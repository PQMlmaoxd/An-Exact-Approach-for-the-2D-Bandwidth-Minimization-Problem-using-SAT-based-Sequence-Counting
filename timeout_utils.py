# timeout_utils.py
# Timeout utilities for long-running SAT solver operations
# Fixed version with Windows compatibility

import signal
import time
import multiprocessing
import threading
import queue
import platform
from typing import Any, Callable, Optional, Tuple
from contextlib import contextmanager


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


# Global function for Windows multiprocessing compatibility
def _global_target_wrapper(func, args, kwargs, result_queue):
    """Global wrapper function for multiprocessing (pickle-compatible)"""
    try:
        result = func(*args, **kwargs)
        result_queue.put(('success', result))
    except Exception as e:
        result_queue.put(('error', e))


class ThreadTimeoutExecutor:
    """
    Thread-based timeout executor for Windows compatibility
    
    Uses threading instead of multiprocessing to avoid pickle issues on Windows.
    Note: Cannot forcefully terminate threads, so timeout is best-effort.
    """
    
    def __init__(self):
        self.thread = None
        self.result_queue = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Threads are created as daemon=True, so they'll clean up automatically
        # Try to empty the queue to prevent memory leaks
        if self.result_queue:
            try:
                while True:
                    self.result_queue.get_nowait()
            except queue.Empty:
                pass
        return False
    
    def execute(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """
        Execute function with timeout using threads
        
        Args:
            func: Function to execute
            timeout: Timeout in seconds
            *args, **kwargs: Arguments for function
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        self.result_queue = queue.Queue()
        
        def target_wrapper():
            try:
                result = func(*args, **kwargs)
                self.result_queue.put(('success', result))
            except Exception as e:
                self.result_queue.put(('error', e))
        
        self.thread = threading.Thread(target=target_wrapper)
        self.thread.daemon = True  # Dies when main program exits
        self.thread.start()
        
        try:
            status, value = self.result_queue.get(timeout=timeout)
            if status == 'success':
                return value
            else:
                raise value
        except queue.Empty:
            # Timeout occurred - thread is already daemon, no need to modify
            raise TimeoutError(f"Function execution exceeded {timeout} seconds")


class ProcessTimeoutExecutor:
    """
    Production-ready timeout executor using multiprocessing (Unix/Linux)
    
    Executes functions in separate processes with strict timeout enforcement.
    Suitable for CPU-intensive SAT solving operations.
    """
    
    def __init__(self):
        self.process = None
        self.queue = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.kill()  # Force kill if terminate doesn't work
    
    def execute(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """
        Execute function with timeout using multiprocessing
        
        Args:
            func: Function to execute  
            timeout: Timeout in seconds
            *args, **kwargs: Arguments for function
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        # Use multiprocessing Queue for inter-process communication
        self.queue = multiprocessing.Queue()
        
        # Create process with global wrapper function
        self.process = multiprocessing.Process(
            target=_global_target_wrapper,
            args=(func, args, kwargs, self.queue)
        )
        
        self.process.start()
        self.process.join(timeout=timeout)
        
        if self.process.is_alive():
            # Timeout occurred
            self.process.terminate()
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.kill()
            raise TimeoutError(f"Function execution exceeded {timeout} seconds")
        
        # Get result from queue
        if not self.queue.empty():
            status, value = self.queue.get()
            if status == 'success':
                return value
            else:
                raise value
        else:
            raise TimeoutError("Process terminated without returning result")


class TimeoutConfig:
    """Configuration class for timeout settings"""
    
    def __init__(self):
        # Default timeout values (in seconds)
        self.random_search_timeout = 30.0
        self.sat_solve_timeout = 300.0  
        self.total_solver_timeout = 600.0
        self.position_constraints_timeout = 60.0
        self.distance_constraints_timeout = 120.0
        self.bandwidth_constraints_timeout = 60.0
        
        # Enable/disable specific timeout types
        self.enable_phase_timeouts = True
        self.enable_constraint_timeouts = True
        self.enable_total_timeout = True
        
        # Multiprocessing settings - disable on Windows due to pickle issues
        self.use_multiprocessing = platform.system() != 'Windows'
        
        print(f"Timeout system: {'Threading' if not self.use_multiprocessing else 'Multiprocessing'} (Platform: {platform.system()})")
    
    def update_timeouts(self, **kwargs):
        """Update timeout values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, float(value))
                print(f"Updated {key} = {value}s")
            else:
                print(f"Warning: Unknown timeout setting '{key}'")
    
    def get_timeout_summary(self) -> str:
        """Get formatted timeout configuration summary"""
        summary = []
        summary.append(f"Random search timeout: {self.random_search_timeout}s")
        summary.append(f"SAT solve timeout: {self.sat_solve_timeout}s")
        summary.append(f"Total solver timeout: {self.total_solver_timeout}s")
        summary.append(f"Position constraints timeout: {self.position_constraints_timeout}s")
        summary.append(f"Distance constraints timeout: {self.distance_constraints_timeout}s")
        summary.append(f"Bandwidth constraints timeout: {self.bandwidth_constraints_timeout}s")
        summary.append(f"Use multiprocessing: {self.use_multiprocessing}")
        return '\n'.join(summary)


def get_timeout_executor():
    """Get appropriate timeout executor based on platform"""
    if platform.system() == 'Windows':
        return ThreadTimeoutExecutor()
    else:
        return ProcessTimeoutExecutor()


# Signal-based timeout for Unix systems
@contextmanager
def signal_timeout(seconds: float):
    """
    Context manager for signal-based timeout (Unix only)
    
    Args:
        seconds: Timeout in seconds
        
    Raises:
        TimeoutError: If timeout is exceeded
    """
    if platform.system() == 'Windows':
        # Signal timeout not supported on Windows
        yield
        return
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))
    
    try:
        yield
    finally:
        # Restore original handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# Decorator for timeout functionality
def timeout_decorator(timeout_seconds: float):
    """
    Decorator to add timeout to functions
    
    Args:
        timeout_seconds: Timeout in seconds
        
    Returns:
        Decorated function with timeout
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with get_timeout_executor() as executor:
                return executor.execute(func, timeout_seconds, *args, **kwargs)
        return wrapper
    return decorator


# Test functions for demonstration
def test_timeout_functionality():
    """Test timeout mechanisms"""
    print("=== Testing Timeout Functionality ===")
    
    # Test function that takes time
    def slow_function(duration: float, return_value: str = "completed"):
        """Test function that sleeps for specified duration"""
        print(f"Starting slow function (duration: {duration}s)")
        time.sleep(duration)
        print(f"Slow function completed")
        return return_value
    
    # Test function that raises exception
    def error_function():
        """Test function that raises an exception"""
        print("Error function called")
        raise ValueError("Test error")
    
    config = TimeoutConfig()
    print(config.get_timeout_summary())
    
    # Test 1: Function completes within timeout
    print("\nTest 1: Function completes within timeout")
    try:
        with get_timeout_executor() as executor:
            result = executor.execute(slow_function, 2.0, 1.0, "success")
            print(f"Result: {result}")
    except TimeoutError as e:
        print(f"Timeout: {e}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Function exceeds timeout
    print("\nTest 2: Function exceeds timeout")
    try:
        with get_timeout_executor() as executor:
            result = executor.execute(slow_function, 1.0, 3.0, "should not see this")
            print(f"Result: {result}")
    except TimeoutError as e:
        print(f"Timeout (expected): {e}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Function raises exception
    print("\nTest 3: Function raises exception")
    try:
        with get_timeout_executor() as executor:
            result = executor.execute(error_function, 5.0)
            print(f"Result: {result}")
    except TimeoutError as e:
        print(f"Timeout: {e}")
    except Exception as e:
        print(f"Error (expected): {e}")
    
    # Test 4: Decorator usage
    print("\nTest 4: Decorator usage")
    
    @timeout_decorator(2.0)
    def decorated_function(duration: float):
        time.sleep(duration)
        return f"Decorated function completed after {duration}s"
    
    try:
        result = decorated_function(1.0)
        print(f"Decorated result: {result}")
    except TimeoutError as e:
        print(f"Decorated timeout: {e}")
    
    try:
        result = decorated_function(3.0)
        print(f"Decorated result: {result}")
    except TimeoutError as e:
        print(f"Decorated timeout (expected): {e}")
    
    print("\n=== Timeout Tests Complete ===")


if __name__ == '__main__':
    test_timeout_functionality()
