"""

 Trabalho final da disciplina "Computação em GPU" (GPU28EE), ministrada pelo professor Giovanni Alfredo Guarneri no
 último semestre de 2024.

 O autor do trabalho é o aluno **Thiago E. KALID.**.

 Para compilar as funções em GPU, usou-se o Cmake. Existem dois shell scripts, um chamado build e outro clean, onde o
 primeiro serve para compilar o projeto, e o segundo para limpar versões antigas de arquivos de compilação.
 Para usuários do linux, basta executar o comando:

 >> . build

 que o projeto deve ser compilado corretamente.

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count

import src.us_raytracing.gpu as gpu
import src.us_raytracing.cpu as cpu

if True:
    import matplotlib
    matplotlib.use("TkAgg") # by default

#%%
# User-input section:
# Speed:
c1 = 1.483
c2 = 5.9

# Emitters:
Nel = 64
xA = np.arange(0, Nel, dtype=np.float32) * 1e-2
xA -= np.mean(xA)
zA = np.zeros_like(xA, dtype=np.float32)

# Focuses:
XX, ZZ = np.meshgrid(
    np.linspace(2, 7, 1, dtype=np.float32),
    np.linspace(15, 20, 1, dtype=np.float32)
)

xF, zF = XX.ravel(), ZZ.ravel()

maxiter = 5
tolerance = 1e-3

# Medium profile:
xmin = -5
xmax = 5
xstep = 1e-2
x = np.arange(xmin, xmax + xstep, xstep, dtype=np.float32)
xS = x
zS = np.sin(xS) + 5  # this is just an example of an arbitrary surface.

# Others profiles examples (just uncomment to try):
# zS = xS * .5 + 5
# zS = xS**2 + 5


Na = len(xA)  # number of emitters
Nf = len(xF)  # number of focuses
N = len(zS)  # number of points of the discrete surface profile

#%%
# GPU:
t0 = time.time()
k_gpu = gpu.parrilla_2007(xA, zA, xF, zF, xS, zS, c1, c2, maxiter, tolerance)
elapsed_time_gpu = time.time() - t0

# %%
# CPU:
k_cpu = np.zeros(shape=(Na, Nf), dtype=int)
t0 = time.time()
for idx in range(Na * Nf):
    a = idx // Nf  # Compute row index
    f = idx % Nf   # Compute column index
    k_cpu[a, f] = cpu.parrilla_2007(xA[a], zA[a], xF[f], zF[f], xS, zS, c1, c2, maxiter, tolerance)
elapsed_time_cpu = time.time() - t0

# %%
# CPU with parallel programming:
k_cpu_parallel = np.zeros(shape=(Na, Nf), dtype=int)
t0 = time.time()
def compute_k(idx):
    a = idx // Nf  # Compute row index
    f = idx % Nf   # Compute column index
    return a, f, cpu.parrilla_2007(xA[a], zA[a], xF[f], zF[f], xS, zS, c1, c2, maxiter, tolerance)

with Pool(processes=cpu_count()) as pool:
    results = pool.map(compute_k, range(Na * Nf))

# Store results in the array
for a, f, value in results:
    k_cpu_parallel[a, f] = value
elapsed_time_cpu_parallel = time.time() - t0

# %%
# Print result:
print(f"""
GPU elapsed time = {elapsed_time_gpu * 1e3:.2f} ms
CPU-parallel elapsed time = {elapsed_time_cpu_parallel * 1e3:.2f} ms
CPU-serial elapsed time = {elapsed_time_cpu * 1e3:.2f} ms
========================================================
speed-up (CPU-serial / GPU): {elapsed_time_cpu/elapsed_time_gpu:.2f} x
speed-up (CPU-parallel / GPU): {elapsed_time_cpu_parallel/elapsed_time_gpu:.2f} x
""")

# %%
# Plot result:
if Na * Nf <= 100:
    plt.figure(figsize=(8, 4))
    plt.plot(xA, zA, "ks", label="Emitters", markersize=5)
    plt.plot(xF, zF, "or", label="Focuses", markersize=5)
    plt.plot(xS, zS, "o-k", markersize=1, linewidth=1, alpha=.8)

    for a in range(Na):
        for f in range(Nf):
            if a == 0 and f == 0:
                label_cpu = f'CPU ({elapsed_time_cpu * 1e3:.2f} ms)'
                label_gpu = f'GPU ({elapsed_time_gpu * 1e3:.2f} ms)'
            else:
                label_cpu = "_"
                label_gpu = "_"

            plt.plot([xA[a], xS[k_cpu[a, f]], xF[f]], [zA[a], zS[k_cpu[a, f]], zF[f]], 'r', linewidth=1, label=label_cpu)
            plt.plot([xA[a], xS[k_gpu[a, f]], xF[f]], [zA[a], zS[k_gpu[a, f]], zF[f]], '--g', linewidth=1, label=label_gpu)

            plt.plot(xS[k_cpu[a, f]], zS[k_cpu[a, f]], 'sr')
            plt.plot(xS[k_gpu[a, f]], zS[k_gpu[a, f]], 'sg')
    plt.legend()
    plt.xlabel("x-axis in mm")
    plt.ylabel("z-axis in mm")
    plt.show()