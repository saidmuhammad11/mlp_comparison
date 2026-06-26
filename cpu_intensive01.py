import time
import random

def branch_work(duration=120):
    print("Starting branch-heavy workload...")
    start = time.time()
    count = 0

    while time.time() - start < duration:
        x = random.randint(0, 1000)
        if x % 3 == 0:
            count += 1
        elif x % 3 == 1:
            count -= 1
        else:
            count *= -1

    print("Branch-heavy workload done.")

if __name__ == "__main__":
    branch_work()
