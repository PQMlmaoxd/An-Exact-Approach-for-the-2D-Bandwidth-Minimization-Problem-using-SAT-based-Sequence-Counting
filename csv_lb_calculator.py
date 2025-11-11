#!/usr/bin/env python3
# csv_lb_calculator.py
# Auto calculate Lower Bound (LB) from Upper Bound (UB) and Graph Diameter D(G)
# Formula: LB = UB / D(G)
# 
# Input: CSV files in csv/input/ directory
# Output: CSV files with added LB column in csv/output/ directory

import os
import sys
import csv
import time
import math
from datetime import datetime
from glob import glob

# Import graph diameter calculator functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph_diameter_calculator import parse_mtx_file, graph_diameter

class CSVLowerBoundCalculator:
    """
    Calculator that processes CSV files to add Lower Bound column
    based on Upper Bound and Graph Diameter
    """
    
    def __init__(self, input_dir="csv/input", output_dir="csv/output", mtx_base="mtx"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.mtx_base = mtx_base
        self.diameter_cache = {}  # Cache D(G) values to avoid recalculation
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*80)
        print("CSV LOWER BOUND CALCULATOR")
        print("="*80)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"MTX files base: {mtx_base}")
        print(f"Formula: LB = ⌈UB / D(G)⌉ (ceiling)")
    
    def find_mtx_file(self, filename):
        """
        Find MTX file from filename column in CSV
        Supports various naming conventions and search paths
        """
        if not filename:
            return None
        
        # Clean filename - remove extensions and prefixes
        base_name = filename
        if base_name.endswith('.mtx'):
            base_name = base_name[:-4]
        
        # Search paths with support for all MTX subdirectories
        search_paths = [
            f"{self.mtx_base}/{filename}",
            f"{self.mtx_base}/group 1/{filename}",
            f"{self.mtx_base}/group 2/{filename}",
            f"{self.mtx_base}/group 3/{filename}",
            f"{self.mtx_base}/group 4/{filename}",
            f"{self.mtx_base}/regular/{filename}",
            f"{self.mtx_base}/temp/{filename}",
            f"{self.mtx_base}/tmp/{filename}",
            f"{self.mtx_base}/{base_name}.mtx",
            f"{self.mtx_base}/group 1/{base_name}.mtx",
            f"{self.mtx_base}/group 2/{base_name}.mtx",
            f"{self.mtx_base}/group 3/{base_name}.mtx",
            f"{self.mtx_base}/group 4/{base_name}.mtx",
            f"{self.mtx_base}/regular/{base_name}.mtx",
            f"{self.mtx_base}/temp/{base_name}.mtx",
            f"{self.mtx_base}/tmp/{base_name}.mtx"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def get_graph_diameter(self, mtx_file):
        """
        Get graph diameter D(G) with caching
        Returns diameter or None if calculation fails
        """
        if not mtx_file or not os.path.exists(mtx_file):
            return None
        
        # Check cache first
        if mtx_file in self.diameter_cache:
            return self.diameter_cache[mtx_file]
        
        try:
            print(f"  Calculating D(G) for {os.path.basename(mtx_file)}...")
            
            # Parse MTX file
            n, edges = parse_mtx_file(mtx_file)
            if n is None or edges is None:
                print(f"    Failed to parse {mtx_file}")
                return None
            
            # Calculate diameter
            diameter, components = graph_diameter(n, edges)
            
            # Cache result
            self.diameter_cache[mtx_file] = diameter
            
            print(f"    D(G) = {diameter} (vertices: {n}, edges: {len(edges)})")
            return diameter
            
        except Exception as e:
            print(f"    Error calculating diameter for {mtx_file}: {e}")
            return None
    
    def calculate_lower_bound(self, ub, diameter):
        """
        Calculate Lower Bound using formula: LB = ⌈UB / D(G)⌉ (ceiling)
        
        Args:
            ub: Upper Bound value
            diameter: Graph diameter D(G)
            
        Returns:
            Lower Bound value (integer, rounded up) or None if calculation not possible
        """
        if ub is None or diameter is None or diameter == 0:
            return None
        
        try:
            # Convert UB to float if it's a string
            if isinstance(ub, str):
                ub = float(ub)
            
            # Calculate LB = ⌈UB / D(G)⌉ (ceiling function)
            lb_raw = ub / diameter
            lb = math.ceil(lb_raw)
            
            return lb
            
        except (ValueError, TypeError, ZeroDivisionError):
            return None
    
    def detect_filename_column(self, fieldnames):
        """
        Auto-detect column containing MTX filenames
        Common column names: filename, file, mtx, matrix, name
        """
        filename_candidates = ['filename', 'file', 'mtx', 'matrix', 'name', 'instance']
        
        for candidate in filename_candidates:
            if candidate in fieldnames:
                return candidate
            # Case insensitive check
            for field in fieldnames:
                if field.lower() == candidate.lower():
                    return field
        
        # If no obvious candidate, use first column as fallback
        if fieldnames:
            print(f"  Warning: No obvious filename column found, using '{fieldnames[0]}'")
            return fieldnames[0]
        
        return None
    
    def detect_ub_column(self, fieldnames):
        """
        Auto-detect column containing Upper Bound values
        Common column names: UB, ub, upper_bound, upperbound
        """
        ub_candidates = ['UB', 'ub', 'upper_bound', 'upperbound', 'upper', 'bound']
        
        for candidate in ub_candidates:
            if candidate in fieldnames:
                return candidate
            # Case insensitive check
            for field in fieldnames:
                if field.lower() == candidate.lower():
                    return field
        
        print(f"  Warning: No UB column found in: {fieldnames}")
        return None
    
    def process_csv_file(self, input_file):
        """
        Process a single CSV file to add LB column
        
        Args:
            input_file: Path to input CSV file
            
        Returns:
            Path to output CSV file or None if processing failed
        """
        print(f"\nProcessing: {os.path.basename(input_file)}")
        
        # Generate output filename
        input_basename = os.path.basename(input_file)
        name, ext = os.path.splitext(input_basename)
        output_file = os.path.join(self.output_dir, f"{name}_with_LB{ext}")
        
        try:
            with open(input_file, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                
                if not reader.fieldnames:
                    print(f"  Error: No columns found in {input_file}")
                    return None
                
                # Auto-detect filename and UB columns
                filename_col = self.detect_filename_column(reader.fieldnames)
                ub_col = self.detect_ub_column(reader.fieldnames)
                
                if not filename_col:
                    print(f"  Error: Cannot detect filename column")
                    return None
                
                if not ub_col:
                    print(f"  Error: Cannot detect UB column")
                    return None
                
                print(f"  Using filename column: '{filename_col}'")
                print(f"  Using UB column: '{ub_col}'")
                
                # Create output fieldnames (original + LB)
                output_fieldnames = list(reader.fieldnames) + ['LB']
                
                # Process rows
                rows_processed = 0
                rows_with_lb = 0
                
                with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
                    writer.writeheader()
                    
                    for row in reader:
                        rows_processed += 1
                        
                        # Get filename and UB values
                        filename = row.get(filename_col, '').strip()
                        ub_str = row.get(ub_col, '').strip()
                        
                        # Calculate LB
                        lb_value = None
                        
                        if filename and ub_str:
                            # Find MTX file
                            mtx_path = self.find_mtx_file(filename)
                            
                            if mtx_path:
                                # Get diameter
                                diameter = self.get_graph_diameter(mtx_path)
                                
                                if diameter is not None:
                                    # Parse UB value
                                    try:
                                        ub_value = float(ub_str)
                                        lb_value = self.calculate_lower_bound(ub_value, diameter)
                                        
                                        if lb_value is not None:
                                            rows_with_lb += 1
                                            print(f"    {filename}: UB={ub_value}, D(G)={diameter}, LB={lb_value} (⌈{ub_value}/{diameter}⌉)")
                                        
                                    except ValueError:
                                        print(f"    {filename}: Invalid UB value '{ub_str}'")
                                else:
                                    print(f"    {filename}: Could not calculate diameter")
                            else:
                                print(f"    {filename}: MTX file not found")
                        
                        # Add LB to row
                        row['LB'] = lb_value if lb_value is not None else ''
                        writer.writerow(row)
                
                print(f"  Results: {rows_processed} rows processed, {rows_with_lb} with LB calculated")
                print(f"  Output: {output_file}")
                
                return output_file
                
        except Exception as e:
            print(f"  Error processing {input_file}: {e}")
            return None
    
    def process_all_csv_files(self):
        """
        Process all CSV files in input directory
        
        Returns:
            List of successfully processed output files
        """
        # Find all CSV files in input directory
        input_pattern = os.path.join(self.input_dir, "*.csv")
        csv_files = glob(input_pattern)
        
        if not csv_files:
            print(f"\nNo CSV files found in {self.input_dir}")
            print(f"Please place CSV files with UB column in {self.input_dir}")
            return []
        
        print(f"\nFound {len(csv_files)} CSV files to process:")
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")
        
        # Process each file
        output_files = []
        start_time = time.time()
        
        for csv_file in csv_files:
            output_file = self.process_csv_file(csv_file)
            if output_file:
                output_files.append(output_file)
        
        # Summary
        total_time = time.time() - start_time
        
        print(f"\n" + "="*60)
        print(f"PROCESSING SUMMARY")
        print(f"="*60)
        print(f"Input files: {len(csv_files)}")
        print(f"Successfully processed: {len(output_files)}")
        print(f"Diameter calculations cached: {len(self.diameter_cache)}")
        print(f"Total processing time: {total_time:.2f}s")
        
        if output_files:
            print(f"\nOutput files created:")
            for f in output_files:
                print(f"  - {os.path.relpath(f)}")
        
        print(f"="*60)
        
        return output_files


def main():
    """
    Main function for CSV Lower Bound calculation
    """
    # Check if custom directories provided
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "csv/input"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "csv/output"
    
    # Create calculator instance
    calculator = CSVLowerBoundCalculator(input_dir, output_dir)
    
    # Process all CSV files
    output_files = calculator.process_all_csv_files()
    
    if output_files:
        print(f"\n✓ Successfully processed {len(output_files)} CSV files")
        print(f"✓ Lower bounds calculated using formula: LB = ⌈UB / D(G)⌉ (ceiling)")
        print(f"✓ Output files saved to: {output_dir}")
    else:
        print(f"\n✗ No files were successfully processed")
        print(f"✗ Please check that:")
        print(f"  - CSV files exist in {input_dir}")
        print(f"  - CSV files have UB column")
        print(f"  - CSV files have filename column pointing to valid MTX files")


if __name__ == "__main__":
    """
    Command line usage: python csv_lb_calculator.py [input_dir] [output_dir]
    
    Arguments:
        input_dir: Directory containing input CSV files (default: csv/input)
        output_dir: Directory for output CSV files (default: csv/output)
    
    CSV Requirements:
        - Must have a column containing Upper Bound values (UB, ub, upper_bound, etc.)
        - Must have a column containing MTX filenames (filename, file, mtx, etc.)
        - MTX files must exist in mtx/ directory structure (groups 1-4, regular, temp)
    
    Examples:
        python csv_lb_calculator.py
        python csv_lb_calculator.py custom_input custom_output
        python csv_lb_calculator.py benchmark_results/csv results/with_lb
    
    Output:
        - Creates new CSV files with all original columns plus LB column
        - LB = ⌈UB / D(G)⌉ where D(G) is calculated graph diameter (ceiling function)
        - Searches MTX files in: mtx/, mtx/group 1/, mtx/group 2/, mtx/group 3/, mtx/group 4/, mtx/regular/, mtx/temp/, mtx/tmp/
        - Caches diameter calculations for efficiency
    """
    
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print(__doc__)
        sys.exit(0)
    
    try:
        main()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
