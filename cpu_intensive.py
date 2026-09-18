import time
import math
import random

def branch_work(duration=180):
    print(f"Starting branch-heavy workload for {duration} seconds...")
    start = time.time()
    count = 0
    
    # Create a heavy list to force actual CPU computation
    heavy_list = [random.random() for _ in range(100000)]
    
    while time.time() - start < duration:
        x = random.randint(0, 1000)
        if x % 3 == 0:
            count += sum(math.sqrt(y) for y in heavy_list[:1000])
        elif x % 3 == 1:
            count -= sum(math.log1p(y) for y in heavy_list[:1000])
        else:
            count *= -1
            count += sum(y ** 2 for y in heavy_list[:1000])
            
    print(f"Branch-heavy workload done. Final count: {count}")

if __name__ == "__main__":
    branch_work(180)