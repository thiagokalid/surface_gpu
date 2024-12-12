import numpy as np
import matplotlib.pyplot as plt

import time
from tqdm import tqdm
import src.surface_gpu.gpu as gpu
from src.surface_gpu.methods import parrilla_2007

# Speed:
c1 = 1.483
c2 = 5.9

# Emitters:
xA = np.array([0], dtype=np.float32)
zA = np.array([0], dtype=np.float32)

# Focuses:
XX, ZZ = np.meshgrid(
    np.linspace(5, 12, 5, dtype=np.float32),
    np.linspace(25, 30, 5, dtype=np.float32)
)

xF, zF = XX.ravel(), ZZ.ravel()

maxiter = 100
epsilon = 2

# Profiles:
xmin = 0
xmax = 15
xstep = 1e-3
x = np.arange(xmin, xmax + xstep, xstep, dtype=np.float32)
xS = x
zS = xS**2/10 + 5

Na = len(xA)
Nf = len(xF)

N = len(zS)

#GPU:
t0 = time.time()
k_gpu = gpu.parrilla_2007(xA, zA, xF, zF, xS, zS, c1, c2, maxiter, epsilon)
elapsed_time_gpu = time.time() - t0

# CPU:
k_cpu = np.zeros(shape=(Na, Nf), dtype=int)
t0 = time.time()
for a in range(Na):
    for f in range(Nf):
        k_cpu[a, f] = parrilla_2007(xA[a], zA[a], xF[f], zF[f], xS, zS, c1, c2, maxiter, epsilon)
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
plt.plot(xA, zA, "ks", label="Emitters")
plt.plot(xF, zF, "sr", label="Focuses")
plt.plot(xS, zS, "o-k", markersize=1, linewidth=1, alpha=.8)


# elapsed_time_gpu = elapsed_time_cpu
# k_gpu = k_cpu

# Plot rays:
for a in range(Na):
    for f in range(Nf):
        if a == 0 and f == 0:
            label_cpu = f'CPU ({elapsed_time_cpu * 1e3:.2f} ms)'
            label_gpu = f'GPU ({elapsed_time_gpu * 1e3:.2f} ms)'
        else:
            label_cpu = "_"
            label_gpu = "_"

        plt.plot([xA[a], xS[k_cpu[a, f]], xF[f]], [zA[a], zS[k_cpu[a, f]], zF[f]], 'r', linewidth=1, label=label_cpu)
        plt.plot([xA[a], xS[k_gpu[a, f]], xF[f]], [zA[a], zS[k_gpu[a, f]], zF[f]], '--g', linewidth=2, label=label_gpu)

        plt.plot(xS[k_cpu[a, f]], zS[k_cpu[a, f]], 'sr')
        plt.plot(xS[k_gpu[a, f]], zS[k_gpu[a, f]], 'sg')
plt.legend()