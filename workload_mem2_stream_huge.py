import subprocess
import time
import sys
import os

def run_stream_fixed_workload():
    # 29 runs took ~74 seconds.
    # To hit 300 seconds: 29 * (300 / 74) = 118 runs.
    ITERATIONS = 118
    
    # Path to the compiled C executable
    exe_path = r"C:\Users\saidm\Desktop\Programming\Hardware_test\stream_official\stream_huge.exe"
    
    if not os.path.exists(exe_path):
        print(f"ERROR: Executable not found at {exe_path}")
        sys.exit(1)
        
    print(f"Starting STREAM benchmark for exactly {ITERATIONS} iterations...")
    start_time = time.time()
    
    for i in range(ITERATIONS):
        # Print update every 10 runs to avoid console spam
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Executing STREAM run {i + 1}/{ITERATIONS}...")
            
        try:
            # Execute the C program, suppressing its terminal output
            subprocess.run(
                [exe_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        except Exception as e:
            print(f"ERROR during execution: {e}")
            sys.exit(1)
            
    elapsed = time.time() - start_time
    print(f"Completed fixed workload ({ITERATIONS} executions) in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_stream_fixed_workload()