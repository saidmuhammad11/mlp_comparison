import time
import numpy as np

def mixed_vector_workload(duration=120):
    # For 16GB RAM, we use ~2.4GB to ensure we bypass the 8MB CPU cache 
    # and stress the memory controller without causing system swap.
    # 100 million float64 elements = 800MB per array. 3 arrays = 2.4GB.
    N = 100_000_000 
    
    print(f"Starting Mixed Workload on Intel i7-8650U...")
    print(f"Allocating ~2.4GB of RAM for vectorized operations...")
    
    # Initialize arrays (Memory Intensive: Allocation and Fill)
    A = np.random.random(N).astype(np.float64)
    B = np.random.random(N).astype(np.float64)
    C = np.zeros(N, dtype=np.float64)

    start_time = time.time()
    cycle_count = 0

    while time.time() - start_time < duration:
        # COMBINED STEP: Performs complex FPU math while streaming from RAM
        # This is a 'Triad-style' operation from your memory_intensive script
        # but replaces simple addition with transcendental math.
        C[:] = np.sin(A) + np.cos(B) * np.sqrt(A + 0.5)
        
        # Periodic update to prevent the compiler/runtime from optimizing out the loop
        A[0] = C[-1] 
        cycle_count += 1

    print(f"Mixed workload complete. Processed {cycle_count} cycles of 100M elements.")

if __name__ == "__main__":
    mixed_vector_workload()