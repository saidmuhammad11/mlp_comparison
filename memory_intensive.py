import time
import numpy as np

def stream_work(duration=120):
    print("Starting STREAM-like memory-intensive workload...")

    # Allocate large arrays (adjust size based on RAM)
    N = 50_000_000  # ~200MB per array for float32
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32) * 2
    C = np.zeros(N, dtype=np.float32)

    start = time.time()

    while time.time() - start < duration:

        # COPY: C = A
        np.copyto(C, A)

        # SCALE: B = 3 * C
        B[:] = 3.0 * C

        # ADD: C = A + B
        C[:] = A + B

        # TRIAD: A = B + 3 * C
        A[:] = B + 3.0 * C

    print("STREAM-like workload done.")

if __name__ == "__main__":
    stream_work()
