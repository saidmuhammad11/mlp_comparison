import subprocess
import time
import sys
import os

def run_nas_ep_fixed_workload():
    # 1 run of EP Class A takes ~140 seconds on this hardware.
    # 2 runs will mathematically guarantee ~280 seconds of fixed work.
    ITERATIONS = 2
    
    # Updated path: OneDrive removed, pointing directly to the local Desktop
    exe_path = r"C:\Users\saidm\Desktop\Programming\Hardware_test\NPB3.0-omp-C\bin\ep.A.x"
    
    if not os.path.exists(exe_path):
        print(f"ERROR: Executable not found at {exe_path}")
        print("Please check the path and try again.")
        sys.exit(1)
        
    print(f"Starting NAS EP (Class A) benchmark for exactly {ITERATIONS} iterations...")
    start_time = time.time()
    
    for i in range(ITERATIONS):
        print(f"Executing NAS EP run {i + 1}/{ITERATIONS}...")
        
        try:
            # Suppress the heavy console output from the C program
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
    run_nas_ep_fixed_workload()