# cpu_intensive.py
import time

def cpu_work(duration=120):
    print("Starting CPU-intensive workload (arithmetic loops)...")
    start = time.time()
    counter = 0
    while time.time() - start < duration:
        # Simulate heavy ALU usage
        for _ in range(1000):
            counter += 1
            counter *= 1.001
            counter %= 1000
    print("CPU-intensive workload done.")

if __name__ == "__main__":
    cpu_work()