import time
import subprocess
import os

def run_susan(duration=120):
    # Update this path to wherever you compiled susan
    exe_path = r"C:\path\to\mibench\automotive\susan\susan.exe"
    input_file = r"C:\path\to\mibench\automotive\susan\input_large.pgm"
    output_file = "output_temp.pgm"
    
    print(f"Starting MiBench (susan - smoothing) for {duration} seconds...")
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < duration:
        # The '-s' flag tells susan to do image smoothing (memory and CPU mixed)
        subprocess.run([exe_path, input_file, output_file, "-s"], stdout=subprocess.DEVNULL)
        iterations += 1
        
    if os.path.exists(output_file):
        os.remove(output_file)
        
    print(f"MiBench (susan) complete. Executed {iterations} times.")

if __name__ == "__main__":
    run_susan()