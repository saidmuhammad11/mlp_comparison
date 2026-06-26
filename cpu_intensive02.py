import time
import math

def fpu_work(duration=120):
    print("Starting Floating-point workload...")
    start = time.time()
    x = 0.5

    while time.time() - start < duration:
        x = math.sin(x) * math.cos(x) + math.sqrt(x + 1.2345)

    print("Floating-point workload done.")

if __name__ == "__main__":
    fpu_work()
