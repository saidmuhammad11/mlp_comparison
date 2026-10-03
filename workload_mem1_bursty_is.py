import subprocess
import time
import sys
import os

def run_nas_is_bursty_fixed():
    # Tune this number so the total run takes roughly 5-7 minutes.
    # We maintain the 0.5s sleep to preserve the "bursty" memory characteristic.
    ITERATIONS = 30 
    
    # Updated local path without OneDrive
    exe_path = r"C:\Users\saidm\Desktop\Programming\Hardware_test\NPB3.0-omp-C\bin\is.A.x"
    
    if not os.path.exists(exe_path):
        print(f"ERROR: Executable not found at {exe_path}")
        sys.exit(1)
        
    print(f"Starting NAS IS (Bursty Memory) for exactly {ITERATIONS} iterations...")
    start_time = time.time()
    
    for i in range(ITERATIONS):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Executing NAS IS run {i + 1}/{ITERATIONS}...")
        
        try:
            subprocess.run(
                [exe_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Sleep creates the 'bursty' idle gap in the workload
            time.sleep(0.5)
        except Exception as e:
            print(f"ERROR during execution: {e}")
            sys.exit(1)
            
    elapsed = time.time() - start_time
    print(f"Completed fixed bursty workload in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_nas_is_bursty_fixed()