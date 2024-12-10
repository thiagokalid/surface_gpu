import numpy as np
import matplotlib.pyplot as plt

import time
from tqdm import tqdm
import src.surface_gpu.gpu as gpu
from src.surface_gpu.methods import parrilla_2007

# Speed:
c1 = 5.9
c2 = 3.2

# Emitters:
xA, zA = 2.5, 0

# Focuses:
xF = 2
zF = 25

maxiter = 100
epsilon = 2

# Profiles:
xmin = -12
xmax = 12
xstep = 1e-2
x = np.arange(xmin, xmax + xstep, xstep, dtype=np.float32)
xS = x
zS = x**2/10 + 5

N = len(zS)

# GPU:
t0 = time.time()
k_gpu = gpu.parrilla_2007(xA, zA, xF, zF, xS, zS, c1, c2)
elapsed_time_gpu = time.time() - t0

# CPU:
k_cpu = 0
t0 = time.time()
for i in range(64 * 40 * 60):
    k_cpu = parrilla_2007(xA, zA, xF, zF, xS, zS, c1, c2)
elapsed_time_cpu = time.time() - t0

# Result output:
print(f"""
GPU:
k = {k_gpu}
elapsed time = {elapsed_time_gpu * 1e3:.2f} ms
CPU:
k = {k_cpu}
elapsed time = {elapsed_time_cpu * 1e3:.2f} ms
Speed-up ≃ {elapsed_time_cpu / elapsed_time_gpu:.2f} x
""")

# Plot result:
plt.plot(xA, zA, "ks")
plt.plot(xF, zF, "sr")
plt.plot(xS, zS, "k")

# Plot rays:
if isinstance(k_cpu, int):
    k_cpu = [k_cpu]
    k_gpu = [k_gpu]

for k_c, k_g in zip(k_cpu, k_gpu):
    plt.plot([xA, xS[k_c], xF], [zA, zS[k_c], zF], 'r', linewidth=1, label=f'CPU ({elapsed_time_cpu * 1e3:.2f} ms)')
    plt.plot([xA, xS[k_g], xF], [zA, zS[k_g], zF], '--g', linewidth=2, label=f'GPU ({elapsed_time_gpu * 1e3:.2f} ms)')

    plt.plot(xS[k_c], zS[k_c], 'sr')
    plt.plot(xS[k_g], zS[k_g], 'sg')
plt.legend()