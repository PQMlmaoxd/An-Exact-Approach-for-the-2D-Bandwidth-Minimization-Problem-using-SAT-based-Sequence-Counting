# 2D Bandwidth Minimization Benchmark System

Automated benchmark runner for testing both **Square (n×n)** and **Rectangular (n×m)** grid 2D Bandwidth Minimization on multiple MTX instances.

## Key Features

- **Dual Grid Support**: Automatically detects and handles both square and rectangular grids
- **Smart Solver Selection**: Uses appropriate solver for each grid type
  - `bandwidth_optimization_solver.py` for square grids (n×n)
  - `rectangular_bandwidth_solver.py` for rectangular grids (n×m)
- **Enhanced MTX Parsing**: Correctly interprets MTX format for grid dimensions
- **Grid Type Filtering**: Option to test only specific grid types
- **Comprehensive Reporting**: Detailed CSV output with grid information

## Configuration Settings

### Default Settings (can be modified in benchmark_runner.py):
- `DEFAULT_TIMEOUT = 300` - Default timeout in seconds
- `DEFAULT_SOLVER = 'glucose42'` - Default SAT solver
- `MAX_PROBLEM_SIZE = 20` - Skip problems with n > this value
- `MTX_FOLDER = 'mtx'` - MTX files directory name
- `RESULTS_FOLDER = 'results'` - Output directory name

## Quick Start

1. **Navigate to benchmark directory:**
   ```
   cd benchmark_automation
   ```

2. **Run with default settings (300s timeout, glucose42 solver):**
   ```
   run_benchmark.bat
   ```

3. **Custom settings:**
   ```
   run_benchmark.bat --timeout 600 --solver cadical195
   ```

4. **Filter by grid type:**
   ```
   run_benchmark.bat --grid-type rectangular --timeout 120
   run_benchmark.bat --grid-type square --solver cadical195
   ```

5. **Show help:**
   ```
   run_benchmark.bat --help
   ```

## Files

- `benchmark_runner.py` - Main benchmark orchestrator
- `run_benchmark.bat` - Windows batch script 
- `results/` - Output directory (auto-created)
- `README.md` - This file

## Customization

### To change MTX source directory:
Edit `MTX_FOLDER` variable in `benchmark_runner.py` line ~107

### To change output location:
Edit `RESULTS_FOLDER` variable in `benchmark_runner.py` line ~108

### To change timeout default:
Edit `DEFAULT_TIMEOUT` variable in `benchmark_runner.py` line ~104

### To change maximum problem size:
Edit `MAX_PROBLEM_SIZE` variable in `benchmark_runner.py` line ~106

## Output

Results are automatically saved to CSV files in the `results/` directory with timestamp:
- `benchmark_results_YYYYMMDD_HHMMSS.csv`

## CSV Format

| Column | Description |
|--------|-------------|
| filename | MTX file name |
| grid_type | Grid type: 'square' or 'rectangular' |
| n_rows | Number of grid rows |
| n_cols | Number of grid columns |
| nodes | Number of vertices placed |
| edges | Number of edges |
| optimal_bandwidth | Found optimal bandwidth (-1 if not solved) |
| time_seconds | Execution time |
| status | SOLVED/TIMEOUT/ERROR/SKIPPED_LARGE/SKIPPED_FILTER |
| solver | SAT solver used |
| timeout_limit | Timeout setting |

## Status Codes

- **SOLVED** - Found optimal solution
- **TIMEOUT** - Exceeded time limit
- **ERROR** - Solver error
- **READ_ERROR** - Could not read MTX file
- **SKIPPED_LARGE** - Problem too large (max dimension > 20)
- **SKIPPED_FILTER** - Grid type filtered out by --grid-type option
- **UNSOLVABLE** - No solution found within bounds

## Grid Type Detection

The system automatically detects grid types from MTX files:

- **Square grids (n×n)**: When rows = columns in MTX header
  - Uses `bandwidth_optimization_solver.py`
  - Example: `5 5 20` → 5×5 square grid
  
- **Rectangular grids (n×m)**: When rows ≠ columns in MTX header  
  - Uses `rectangular_bandwidth_solver.py`
  - Example: `3 7 12` → 3×7 rectangular grid

## Command Line Options

### Batch File Options:
```
run_benchmark.bat [options]

Options:
  --timeout <seconds>    Timeout per problem (default: 300)
  --solver <name>        SAT solver: glucose42 or cadical195 (default: glucose42)
  --grid-type <type>     Grid filter: auto, square, rectangular (default: auto)
  --help                 Show help message
```

### Python Script Options:
```
python benchmark_runner.py [options]

Same options as batch file, plus direct Python execution
```

## Supported Solvers

- `glucose42` (default)
- `cadical195`

## Requirements

- Python 3.7+
- PyQt5 and python-sat libraries
- MTX files in the `../mtx/` directory
- Both solvers:
  - `bandwidth_optimization_solver.py` (for square grids)
  - `rectangular_bandwidth_solver.py` (for rectangular grids)
