import subprocess, time
print('Running NAS EP (Bursty CPU) - Targeting LOW state')
end = time.time() + 350
while time.time() < end:
    subprocess.run([r'C:/Users/saidm/Desktop/Programming/Hardware_test/NPB3.0-omp-C/bin/ep.A.x'], stdout=subprocess.DEVNULL)
    time.sleep(0.5)
