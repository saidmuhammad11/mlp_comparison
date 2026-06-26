import time
import numpy as np

def streaming_work(duration=120):
    print("Starting streaming workload...")
    size = 50_000_000  # ~200MB for float32
    data = np.random.rand(size).astype(np.float32)

    start = time.time()

    while time.time() - start < duration:
        _ = data.sum()   # Sequential scan

    print("Streaming workload done.")

if __name__ == "__main__":
    streaming_work()
