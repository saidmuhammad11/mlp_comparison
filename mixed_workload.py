# mixed_workload_shuffled.py
import time
import numpy as np

def shuffled_mixed_work(duration=120):
    print(f"Starting Mixed Workload 2: Shuffled Lookup & Branching...")
    
    # 1. Setup: Create ~3GB of data structures
    # A large pool of data and a randomized index map to force random RAM access
    N = 100_000_000 
    data_pool = np.random.normal(0, 1, N).astype(np.float32)
    indices = np.arange(N)
    np.random.shuffle(indices) # This is key: it breaks sequential memory access
    
    print(f"Memory allocated. Starting 120s stress test...")
    
    start_time = time.time()
    total_processed = 0
    chunk_size = 1_000_000 # Process in chunks to maintain responsiveness

    while time.time() - start_time < duration:
        # Pick a random chunk of the shuffled indices
        start_idx = (total_processed) % (N - chunk_size)
        current_indices = indices[start_idx : start_idx + chunk_size]
        
        # MEMORY STRESS: Indirect lookup (data_pool[indices])
        # CPU STRESS: Applying transcendental math + branching logic
        chunk = data_pool[current_indices]
        
        # Simulated workload: 
        # 1. FPU Math (from cpu_intensive2.py logic)
        processed_chunk = np.sqrt(np.abs(chunk)) * np.exp(chunk / 10.0)
        
        # 2. Branching Logic (from cpu_intensive.py logic)
        # We apply different operations based on the data values
        mask = processed_chunk > 1.0
        processed_chunk[mask] *= 0.5
        processed_chunk[~mask] += 1.0
        
        # Write back to the pool
        data_pool[current_indices] = processed_chunk
        
        total_processed += chunk_size

    print(f"Workload 2 done. Total elements shuffled and processed: {total_processed}")

if __name__ == "__main__":
    shuffled_mixed_work()