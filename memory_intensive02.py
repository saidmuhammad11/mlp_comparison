# memory_intensive.py
import time
import random

def memory_work(duration=120):
    print("Starting memory-intensive workload (random RAM access)...")
    # Allocate ~1 GB of data (adjust based on your system)
    size = 250_000_000  # ~1 byte per int → ~250MB
    arr = list(range(size))
    start = time.time()

    while time.time() - start < duration:
        # Random access to stress cache & memory
        _ = arr[random.randint(0, size - 1)]
    
    print("Memory-intensive workload done.")

if __name__ == "__main__":
    memory_work()