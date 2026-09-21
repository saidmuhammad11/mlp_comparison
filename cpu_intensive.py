import time
import math
import random
import multiprocessing
import os

def branch_work_worker(iterations, core_id):
    print(f"[Core {core_id}] Starting workload with {iterations} iterations...")
    
    # PHASE 1: Intense CPU Math (High IPC -> Triggers "High" State)
    print(f"[Core {core_id}] Phase 1: Intense CPU Math")
    count = 0
    heavy_list = [random.random() for _ in range(50000)]
    for _ in range(int(iterations * 0.4)):  # 40% of the work (approx. 2 minutes)
        x = random.randint(0, 1000)
        if x % 3 == 0:
            count += sum(math.sqrt(y) for y in heavy_list[:1000])
        elif x % 3 == 1:
            count -= sum(math.log1p(y) for y in heavy_list[:1000])
        else:
            count *= -1
            count += sum(y ** 2 for y in heavy_list[:1000])
            
    # PHASE 2: Mild CPU Math with delays (Lower IPC -> Triggers "Medium" State)
    print(f"[Core {core_id}] Phase 2: Intermittent CPU Load")
    for _ in range(int(iterations * 0.2)):  # 20% of the work (approx. 1 minute)
        count += sum(math.sqrt(y) for y in heavy_list[:100])
        time.sleep(0.001)  # Artificial stall to drop IPC
        
    # PHASE 3: Memory Thrashing (High L3 Misses / Low IPC -> Triggers "Low" State)
    print(f"[Core {core_id}] Phase 3: Memory Thrashing")
    # Allocate a massive array that exceeds cache size to force RAM access
    massive_array = [random.random() for _ in range(2000000)] 
    for _ in range(int(iterations * 0.4)):  # 40% of the work (approx. 2 minutes)
        idx = random.randint(0, len(massive_array) - 1)
        count += massive_array[idx]
        
    print(f"[Core {core_id}] Done. Final count: {count}")

if __name__ == "__main__":
    cores_str = os.environ.get("OMP_NUM_THREADS", "1")
    try:
        num_cores = int(cores_str)
    except ValueError:
        num_cores = 1

    # Increased from 15,000 to 500,000 to stretch execution to ~5 minutes
    ITERATIONS_PER_CORE = 500000 
    
    print(f"Starting multi-processed workload on {num_cores} cores.")
    start_time = time.time()
    
    processes = []
    for i in range(num_cores):
        p = multiprocessing.Process(target=branch_work_worker, args=(ITERATIONS_PER_CORE, i))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    elapsed = time.time() - start_time
    print(f"All cores completed identical work in {elapsed:.2f} seconds.")