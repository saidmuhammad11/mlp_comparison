import time
import random
import math
import multiprocessing
import os

def cpu_phase(duration_s):
    """Pure CPU math that fits inside L1 cache to drive IPC > 2.2 (Triggers HIGH)"""
    end_time = time.time() + duration_s
    heavy_list = [random.random() for _ in range(2000)]
    count = 0
    while time.time() < end_time:
        for _ in range(100):
            count += sum(math.sqrt(y) + math.sin(y) for y in heavy_list)
    return count

def memory_phase(duration_s, arr, arr_len, stride):
    """Random pointer chasing to defeat prefetchers, driving L3 miss > 0.40 (Triggers LOW)"""
    end_time = time.time() + duration_s
    count = 0
    idx = 0
    while time.time() < end_time:
        # Loop unrolling to maximize memory requests per second
        for _ in range(10000):
            count += arr[idx]
            idx = (idx + stride) % arr_len
    return count

def medium_phase(duration_s):
    """Mild CPU math with micro-sleeps to keep IPC between 1.2 and 2.0 (Triggers MEDIUM)"""
    end_time = time.time() + duration_s
    heavy_list = [random.random() for _ in range(1000)]
    count = 0
    while time.time() < end_time:
        for _ in range(10):
            count += sum(y**2 for y in heavy_list)
        time.sleep(0.01)  # Micro-sleep artificially drops the IPC
    return count

def mixed_work_worker(core_id):
    print(f"[Core {core_id}] Allocating massive array for memory phases...")
    # 10M integers (~80MB total) guarantees L3 cache is overwhelmed
    massive_array = [i for i in range(10000000)]
    arr_len = len(massive_array)
    stride = 1048583  # Prime stride for random access

    # Phase 1: High State (60 seconds)
    print(f"[Core {core_id}] -> PHASE 1 (0-60s): Intense CPU Math (Target: HIGH)")
    cpu_phase(60)

    # Phase 2: Low State (60 seconds)
    print(f"[Core {core_id}] -> PHASE 2 (60-120s): Memory Thrashing (Target: LOW)")
    memory_phase(60, massive_array, arr_len, stride)

    # Phase 3: Medium State (60 seconds)
    print(f"[Core {core_id}] -> PHASE 3 (120-180s): Mild Work (Target: MEDIUM)")
    medium_phase(60)

    # Phase 4: High State (60 seconds)
    print(f"[Core {core_id}] -> PHASE 4 (180-240s): Intense CPU Math (Target: HIGH)")
    cpu_phase(60)

    # Phase 5: Low State (60 seconds)
    print(f"[Core {core_id}] -> PHASE 5 (240-300s): Memory Thrashing (Target: LOW)")
    memory_phase(60, massive_array, arr_len, stride)

    print(f"[Core {core_id}] Finished 5-minute macro-profile.")

if __name__ == "__main__":
    cores_str = os.environ.get("OMP_NUM_THREADS", "1")
    try:
        num_cores = int(cores_str)
    except ValueError:
        num_cores = 1

    print(f"Starting multi-processed Mixed Workload on {num_cores} cores.")
    print("Execution time is hard-locked to exactly 300 seconds.")
    start_time = time.time()
    
    processes = []
    for i in range(num_cores):
        p = multiprocessing.Process(target=mixed_work_worker, args=(i,))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    elapsed = time.time() - start_time
    print(f"All cores completed phase-mixed work in {elapsed:.2f} seconds.")