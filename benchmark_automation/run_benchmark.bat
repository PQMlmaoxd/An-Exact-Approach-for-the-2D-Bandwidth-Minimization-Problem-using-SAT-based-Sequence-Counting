@echo off
REM ========================================
REM 2D Bandwidth Minimization Benchmark
REM Automated batch script for Windows
REM ========================================

echo.
echo =========================================
echo  2D BANDWIDTH MINIMIZATION BENCHMARK
echo  Square (n×n) and Rectangular (n×m) Grids
echo =========================================
echo.

REM Set default parameters
set TIMEOUT=300
set SOLVER=glucose42
set GRID_TYPE=auto

REM Parse command line arguments
:parse
if "%1"=="" goto run
if "%1"=="--timeout" (
    set TIMEOUT=%2
    shift
    shift
    goto parse
)
if "%1"=="--solver" (
    set SOLVER=%2
    shift
    shift
    goto parse
)
if "%1"=="--grid-type" (
    set GRID_TYPE=%2
    shift
    shift
    goto parse
)
if "%1"=="--help" (
    echo Usage: run_benchmark.bat [options]
    echo Options:
    echo   --timeout ^<seconds^>    Timeout per problem (default: 300)
    echo   --solver ^<name^>        SAT solver: glucose42 or cadical195 (default: glucose42)
    echo   --grid-type ^<type^>     Grid filter: auto, square, rectangular (default: auto)
    echo   --help                 Show this help message
    echo.
    echo Examples:
    echo   run_benchmark.bat --timeout 60 --solver cadical195
    echo   run_benchmark.bat --grid-type rectangular --timeout 120
    echo.
    pause
    exit /b 0
)
shift
goto parse

:run
echo Timeout: %TIMEOUT% seconds
echo Solver: %SOLVER%
echo Grid Type Filter: %GRID_TYPE%
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if benchmark_runner.py exists
if not exist "benchmark_runner.py" (
    echo ERROR: benchmark_runner.py not found
    echo Please make sure you're in the benchmark_automation directory
    pause
    exit /b 1
)

REM Run the benchmark
echo Starting enhanced benchmark...
echo - Supports both square (n×n) and rectangular (n×m) grids
echo - Auto-detects grid type from MTX format
echo - Uses appropriate solver for each grid type
echo.
python benchmark_runner.py --timeout %TIMEOUT% --solver %SOLVER%

REM Check if benchmark completed successfully
if errorlevel 1 (
    echo.
    echo ERROR: Benchmark failed
    echo Check console output above for details
) else (
    echo.
    echo SUCCESS: Benchmark completed
    echo Check the results/ folder for CSV output with grid type information
    echo.
    echo CSV includes:
    echo - Grid type (square/rectangular)
    echo - Grid dimensions (n_rows × n_cols)
    echo - Solver used for each grid type
    echo - Performance metrics
)

echo.
pause
