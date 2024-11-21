import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

from src.surface_gpu.methods import parrilla_generalized
matplotlib.use("QtAgg")

# Speed:
c1 = 5.900
c2 = 3.4
c3 = 4.8
c4 = 1.483

xmin = -10
xmax = 10
xstep = 1e-2
x = np.arange(xmin, xmax + xstep, xstep)

# Emitters:
xA, zA = 0, 0

# Focuses:
xF = 4
zF = 26

# Profiles:
x1 = x
z1 = np.sin(.25 * x) + 5

x2 = x
z2 = x2**2/20 + 10

x3 = x
z3 = np.sin(.25 * x) + 20


x = [x1, x2, x3]
z = [z1, z2, z3]
c = [c1, c2, c3, c4]
M = len(x)

k, elapsed_time = parrilla_generalized(x, z, xA, zA, xF, zF, c, tolerance=4)

# Profiles:
i = 0
# Plot surfaces:
for xi, zi in zip(x, z):
    plt.plot(x[i], z[i], 'o-', color="b", alpha=.5, label=fr"$S_{i + 1}$. $k_{i}={k[i]}$", markersize=1)
    plt.plot(x[i][k[i]], z[i][k[i]], '*', color="r")
    i += 1

# Plot rays:
for i in range(M+1):
    if i == 0:
        plt.plot([xA, x[i][k[0]]], [zA, z[i][k[0]]], color='lime')
    elif i == M:
        plt.plot([xF, x[i-1][k[i-1]]], [zF, z[i-1][k[i-1]]], color='lime')
    else:
        plt.plot([x[i][k[i]], x[i-1][k[i-1]]], [z[i][k[i]], z[i-1][k[i-1]]], color='lime')
# Emitter:
plt.plot(xA, zA, 'sk', label="Emitters")

# Focus:
plt.plot(xF, zF, 'xk', label="Focuses")

plt.legend()
plt.xlabel("x-axis in mm")
plt.ylabel("z-axis in mm")
plt.gca().set_aspect("equal")
plt.grid()
plt.title(f"Elapsed time = {elapsed_time * 1000:.2f} ms")
plt.show()
