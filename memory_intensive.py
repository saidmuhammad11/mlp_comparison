import time
import random
import math
import multiprocessing
import os

def heavy_cpu_phase(duration_s):
    """Sustained high IPC math to trigger 'High' state."""
    end_time = time.time() + duration_s
    heavy_list = [random.random() for _ in range(5000)]
    count = 0
    while time.time() < end_time:
        count += sum(math.sqrt(y) + math.sin(y) for y in heavy_list)
    return count

def memory_thrash_phase(duration_s, massive_array, array_len, stride):
    """Sustained cache thrashing to trigger 'Low' state."""
    end_time = time.time() + duration_s
    count = 0
    idx = 0
    while time.time() < end_time:
        for _ in range(50000):
            count += massive_array[idx]
            idx = (idx + stride) % array_len
    return count

def mild_cpu_phase(duration_s):
    """Sustained mild IPC with micro-sleeps to trigger 'Medium' state."""
    end_time = time.time() + duration_s
    heavy_list = [random.random() for _ in range(1000)]
    count = 0
    while time.time() < end_time:
        count += sum(y**2 for y in heavy_list)
        time.sleep(0.005)  # Micro-sleep artificially drops IPC below the High threshold
    return count

def mixed_work_worker(core_id):
    print(f"[Core {core_id}] Starting Time-Blocked Mixed Workload...")
    
    # 1. Setup massive array for the Memory Phase (80MB)
    massive_array = [i for i in range(10000000)]
    array_len = len(massive_array)
    stride = 1048583 
    
    total_count = 0

    # 2. Execute discrete 30-second phases to allow DVFS controller to stabilize
    # Total duration = (3 phases * 30s) * 3 iterations = ~270 seconds (4.5 minutes)
    for i in range(3):
        print(f"[Core {core_id}] Iteration {i+1}/3: Phase A (Intense CPU -> High)")
        total_count += heavy_cpu_phase(30)
        
        print(f"[Core {core_id}] Iteration {i+1}/3: Phase B (Memory Thrash -> Low)")
        total_count += memory_thrash_phase(30, massive_array, array_len, stride)
        
        print(f"[Core {core_id}] Iteration {i+1}/3: Phase C (Mild CPU -> Medium)")
        total_count += mild_cpu_phase(30)

    print(f"[Core {core_id}] Done. Checksum: {total_count}")

if __name__ == "__main__":
    cores_str = os.environ.get("OMP_NUM_THREADS", "1")
    try:
        num_cores = int(cores_str)
    except ValueError:
        num_cores = 1

    print(f"Starting multi-processed Mixed Workload on {num_cores} cores.")
    start_time = time.time()
    
    processes = []
    for i in range(num_cores):
        p = multiprocessing.Process(target=mixed_work_worker, args=(i,))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    elapsed = time.time() - start_time
    print(f"All cores completed block-mixed work in {elapsed:.2f} seconds.")