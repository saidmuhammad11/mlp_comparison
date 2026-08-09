import subprocess, time, os
print('Running NAS CG (Memory Intensive - Single-Threaded for LOW state)')
env = os.environ.copy()
env['OMP_NUM_THREADS'] = '1'
end = time.time() + 350
while time.time() < end:
    subprocess.run([r'C:/Users/saidm/OneDrive/Desktop/Programming/Hardware_test/NPB3.0-omp-C/bin/cg.C.x'], env=env, stdout=subprocess.DEVNULL)
