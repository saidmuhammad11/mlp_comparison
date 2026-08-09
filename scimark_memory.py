import time
import subprocess

def run_scimark_memory(duration=120):
    exe_path = r"C:\Users\saidm\OneDrive\Desktop\Programming\Hardware_test\scimark2\scimark2.exe"
    
    print(f"Starting SciMark 2.0 (Memory Intensive - Large Dataset) for {duration} seconds...")
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < duration:
        subprocess.run([exe_path, "-large"], stdout=subprocess.DEVNULL)
        iterations += 1
        
    print(f"SciMark 2.0 (Memory) complete. Executed {iterations} full benchmark cycles.")

if __name__ == "__main__":
    run_scimark_memory()