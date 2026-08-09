import time
import subprocess
import os

def run_qsort(duration=120):
    # Update this path to wherever you compiled qsort
    exe_path = r"C:\path\to\mibench\automotive\qsort\qsort_large.exe"
    input_file = r"C:\path\to\mibench\automotive\qsort\input_large.dat"
    
    print(f"Starting MiBench (qsort) for {duration} seconds...")
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < duration:
        # We suppress the output with stdout=subprocess.DEVNULL so it doesn't flood your console
        subprocess.run([exe_path, input_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        iterations += 1
        
    print(f"MiBench (qsort) complete. Executed {iterations} times.")

if __name__ == "__main__":
    run_qsort()