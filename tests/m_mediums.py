import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from src.surface_gpu.methods import tof
matplotlib.use("TkAgg")

# Speed:
c1 = 5.725
c2 = 5.725
c3 = 5.725

xmin = -10
xmax = 10
xstep = 1e-2
x = np.arange(xmin, xmax + xstep, xstep)

# Emitters:
xt, zt = 0, 0

# Focuses:
xf = 6.86
zf = 15.66

# Profiles:
x1 = x
z1 = 5 * np.ones_like(x1)

x2 = x
z2 = (163 + 5 * x2)/16
#
# x3 = x
# z3 = np.sin(.25 * x) + 25

M = 2
Mk = k0 = Vk = Vk0 = np.zeros(M, dtype=int)
k = k0 + 1000

x = [x1, x2]
z = [z1, z2]
c = [c1, c2, c3]

N = [len(xi) for xi in x]
for i in range(100):
    if i == 7:
        abc = 1
    for m in range(M):
        if m == 0:
            km = int(k0[m])

            #
            xbeg, zbeg = xt, zt

            #
            xmid, zmid = x1, z1

            #
            xend, zend = float(x2[k0[m+1]]), float(z2[k0[m+1]])

            #
            cbeg = c[0]
            cend = c[1]
        elif m == M-1:
            km = int(k0[m])

            #
            xbeg, zbeg = float(x2[k0[M-2]]), float(z2[k0[M-2]])

            #
            xmid, zmid = x[-1], x[-1]

            #
            xend, zend = xf, zf

            cbeg = c[M-1]
            cend = c[M]
        else:
            km = int(k0[m])

            #
            xbeg, zbeg = float(x[m-1][k0[m-1]]), float(z[m-1][k0[m-1]])

            #
            xmid, zmid = x[m], z[m]

            #
            xend, zend = float(x[m+1][k0[m+1]]), float(z[m+1][k0[m+1]])

            cbeg = c[m]
            cend = c[m+1]

        #
        Vk0 = tof(km, xmid, zmid, xbeg, zbeg, cbeg) + tof(km, xmid, zmid, xend, zend, cend)
        Vk = tof(km + 1, xmid, zmid, xbeg, zbeg, cbeg) + tof(km + 1, xmid, zmid, xend, zend, cend)

        #
        k[m] = k0[m] - np.round(Vk0 / (Vk - Vk0))
    k = np.array([np.max([np.min([k[i], N[i]-4]), 0]) for i in range(M)])

    print("k=", k)
    if np.all(np.abs(k - k0) <= 1):
        print(f"Break before end. Iteration {i}")
        break
    else:
        k0 = np.copy(k)


# Profiles:
i = 0
colors = ['g', 'b', 'c']
for xi, zi in zip(x, z):
    plt.plot(x[i], z[i], 'o-', color=colors[i], alpha=.5, label=fr"$S_{i + 1}$")
    plt.plot(x[i][k[i]], z[i][k[i]], '*', color="r")
    i += 1

# Emitter:
plt.plot(xt, zt, 'sk', label="Emitters")

# Focus:
plt.plot(xf, zf, 'xk', label="Focuses")

plt.legend()
plt.xlabel("x-axis in mm")
plt.ylabel("z-axis in mm")
plt.grid()
plt.show()
