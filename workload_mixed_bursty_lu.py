import subprocess, time
print('Running NAS LU (Bursty/Intermittent) - Targeting LOW/MEDIUM state')
end = time.time() + 350
while time.time() < end:
    subprocess.run([r'C:/Users/saidm/OneDrive/Desktop/Programming/Hardware_test/NPB3.0-omp-C/bin/lu.A.x'], stdout=subprocess.DEVNULL)
    time.sleep(0.5)
