import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from src.surface_gpu.methods import parrilla_generalized
matplotlib.use("TkAgg")

# Speed:
c1 = 5.9
c2 = 3.2
c3 = 1.483
c4 = 3.4

xmin = -12
xmax = 12
xstep = 1e-2
x = np.arange(xmin, xmax + xstep, xstep)

# Emitters:
xA, zA = 2.5, 0

# Focuses:
xF = 2
zF = 25

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

result = parrilla_generalized(x, z, xA, zA, xF, zF, c, tolerance=4, maxiter=200)

k = result['result']
elapsed_time = result['elapsed_time']

# Plot rays:
tof = 0
for i in range(M+1):
    if i == 0:
        plt.plot([xA, x[i][k[0]]], [zA, z[i][k[0]]], color='lime')
        tof += np.sqrt((xA - x[i][k[0]]) ** 2 + (zA - z[i][k[0]]) ** 2) / c[0]
    elif i == M:
        plt.plot([xF, x[i-1][k[i-1]]], [zF, z[i-1][k[i-1]]], color='lime')
        tof += np.sqrt((xF - x[i - 1][k[i - 1]]) ** 2 + (zF - z[i - 1][k[i - 1]]) ** 2) / c[-1]
    else:
        plt.plot([x[i][k[i]], x[i-1][k[i-1]]], [z[i][k[i]], z[i-1][k[i-1]]], color='lime')
        tof += np.sqrt((x[i][k[i]] - x[i - 1][k[i - 1]]) ** 2 + (x[i][k[i]] - z[i - 1][k[i - 1]]) ** 2) / c[i]

# Plot surfaces:
for i in range(M):
    plt.plot(x[i], z[i], 'o-', color="b", alpha=.5, label=fr"$S_{i + 1}$. $k_{i}={k[i]}$", markersize=2)
    plt.plot(x[i][k[i]], z[i][k[i]], '*', color="r")

# Emitter:
plt.plot(xA, zA, 'sk', label="Emitters")

# Focus:
plt.plot(xF, zF, 'xk', label="Focuses")

plt.legend()
plt.xlabel("x-axis in mm")
plt.ylabel("z-axis in mm")
plt.gca().set_aspect("equal")
plt.grid()
if result["converged"]:
    iterations = result["iter"]
    plt.title(f"Converged with {iterations} iterations or {elapsed_time * 1000:.2f} ms." + rf"$TOF = {tof:.2f}\mu s$")
else:
    plt.title("Has not converged.")
plt.show()
