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

if __name__ == "__main__":

    #%%
    # User-input section:
    # Speed:
    c1 = 1.483
    c2 = 5.9

    # Emitters:
    xA = np.array([0], dtype=np.float32)
    zA = np.array([0], dtype=np.float32)

    num_focus = np.arange(2, 100, dtype=np.int32)
    Nrepeats = 5
    Nfocus = len(num_focus)

    elapsed_time_cpu = Nfocus * [0]
    elapsed_time_cpu_parallel = Nfocus * [0]
    elapsed_time_gpu = Nfocus * [0]

    # Numerical method:
    maxiter = 5
    tolerance = 1e-3

    # Medium profile:
    xmin = -5
    xmax = 5
    xstep = 1e-2
    x = np.arange(xmin, xmax + xstep, xstep, dtype=np.float32)
    xS = x
    zS = np.sin(xS) + 5  # this is just an example of an arbitrary surface.

    for j in range(Nrepeats):
        time.sleep(2)
        for m, n in enumerate(num_focus):
            # Focuses:
            XX, ZZ = np.meshgrid(
                np.linspace(2, 7, m, dtype=np.float32),
                np.linspace(15, 20, m, dtype=np.float32)
            )

            xF, zF = XX.ravel(), ZZ.ravel()


            Na = len(xA)  # number of emitters
            Nf = len(xF)  # number of focuses
            N = len(zS)  # number of points of the discrete surface profile

            #%%
            # GPU:
            t0 = time.time()
            k_gpu = gpu.parrilla_2007(xA, zA, xF, zF, xS, zS, c1, c2, maxiter, tolerance)
            elapsed_time_gpu[m] += (time.time() - t0) * 1/Nrepeats

            # %%
            # CPU:
            k_cpu = np.zeros(shape=(Na, Nf), dtype=int)
            t0 = time.time()
            for idx in range(Na * Nf):
                a = idx // Nf  # Compute row index
                f = idx % Nf   # Compute column index
                k_cpu[a, f] = cpu.parrilla_2007(xA[a], zA[a], xF[f], zF[f], xS, zS, c1, c2, maxiter, tolerance)
            elapsed_time_cpu[m] += (time.time() - t0) * 1/Nrepeats

            # %%
            # CPU with parallel programming:
            k_cpu_parallel = np.zeros(shape=(Na, Nf), dtype=int)
            t0 = time.time()
            def compute_k(idx):
                a = idx // Nf  # Compute row index
                f = idx % Nf  # Compute column index
                return a, f, cpu.parrilla_2007(xA[a], zA[a], xF[f], zF[f], xS, zS, c1, c2, maxiter, tolerance)


            with Pool(processes=cpu_count()) as pool:
                results = pool.map(compute_k, range(Na * Nf))

            # Store results in the array
            for a, f, value in results:
                k_cpu_parallel[a, f] = value
            elapsed_time_cpu_parallel[m] += (time.time() - t0) * 1/Nrepeats

    plt.figure(figsize=(8, 4))
    plt.plot(num_focus, elapsed_time_gpu, "o-", color='r', linewidth=.5, markersize=3, label="GPU parallel")
    plt.plot(num_focus, elapsed_time_cpu, "s-", color='b', linewidth=.5, markersize=3, label="CPU serial")
    plt.plot(num_focus, elapsed_time_cpu_parallel, "^-", color='g', linewidth=.5, markersize=3, label="CPU parallel")
    plt.gca().set_yscale('log')
    plt.xlabel("Number of focus in each axis")
    plt.ylabel("Elapsed-time in seconds")
    plt.title("Runtime of proposed ray-tracing algorithm using serial \n and parallel implementation for an increasing amount of focuses.")
    plt.grid()
    plt.legend()

    plt.savefig("../figures/two_media_speedup.png")