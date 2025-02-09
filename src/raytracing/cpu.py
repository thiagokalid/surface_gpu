import time
import numpy as np

__all__ = ["parrilla_2007", "parrilla_adapted", "parrilla_adapted_batch"]

def _lin_interp(x0, xf, y0, yf, x):
    if x > xf:
        return yf
    elif x < x0:
        return y0
    else:
        return  (yf - y0)/(x0 - xf) * x

def _compute_Mk(k: int, x2: np.ndarray, z2: np.ndarray):
    delta = .1
    xk_delta = x2[k] + delta
    zk_delta = _lin_interp(x0=x2[k], xf=x2[k + 1], y0=z2[k], yf=z2[k + 1], x=xk_delta)
    Mk = (zk_delta - z2[k]) / (xk_delta - x2[k])
    return Mk


def tof(k: int, x2: np.ndarray, z2: np.ndarray, x1: float, z1: float, c: float, Mk: float):
    return 1 / c * ((x2[k] - x1) + Mk * (z2[k] - z1)) / np.sqrt((x2[k] - x1) ** 2 + (z2[k] - z1) ** 2)


def parrilla_2007(xA: float, zA: float, xF: float, zF: float, xS: np.ndarray, zS: np.ndarray, c1: float, c2: float, maxiter: int=100, tolerance: float=1e-4):
    k0 = 0
    N = len(xS)

    for i in range(maxiter):
        k0 = int(np.round(k0))
        Mk = _compute_Mk(k0, xS, zS)
        Vk0 = tof(k0, xS, zS, xA, zA, c1, Mk) + tof(k0, xS, zS, xF, zF, c2, Mk)
        Vk = tof(k0 + 1, xS, zS, xA, zA, c1, Mk) + tof(k0 + 1, xS, zS, xF, zF, c2, Mk)
        istep = Vk0 / (Vk - Vk0)


        k = k0 - istep
        if k >= N-2:
            k = N-3
        elif k < 0:
            k = 0

        # print(f"k = {k:.4f}, k0 = {k0:.4f}; step = {istep:.4f}; np.abs(k - k0) = {np.abs(k - k0):.4f}")

        if np.abs(k - k0) <= tolerance:
            break
        else:
            k0 = k
    return int(k)

def parrilla_adapted(x: list, z: list, xA: float, zA: float, xF: float, zF: float, c: list, tolerance: float=1e-4, maxiter: int=100, delta=1e-3):
    M = len(c)
    k0 = np.zeros(M-1, dtype=float)
    step = np.copy(k0)
    k = k0 + 1000

    output = {
        "ki": [],
        "elapsed_time": -1,
        "converged": False,
        "result": None,
        "iter": 0
    }

    t0 = time.time()
    for i in range(maxiter):
        for m in range(M-1):  # 0 1 2
            if m == 0:
                p = k0[m]
                pn = k0[m + 1]
                Mk = (z[m](p + delta) - z[m](p)) / (x[m](p + delta) - x[m](p))

                Vk0 = \
                    1 / c[0] * ((x[m](p) - xA) + Mk * (z[m](p) - zA)) / np.sqrt(
                        (x[m](p) - xA) ** 2 + (z[m](p) - zA) ** 2) + \
                    1 / c[1] * ((x[m](p) - x[m + 1](pn)) + Mk * (z[m](p) - z[m + 1](pn))) / np.sqrt(
                        (x[m](p) - x[m + 1](pn)) ** 2 + (z[m](p) - z[m + 1](pn)) ** 2)

                Vk = \
                    1 / c[0] * ((x[m](p + 1) - xA) + Mk * (z[m](p + 1) - zA)) / np.sqrt(
                        (x[m](p + 1) - xA) ** 2 + (z[m](p + 1) - zA) ** 2) + \
                    1 / c[1] * ((x[m](p + 1) - x[m + 1](pn)) + Mk * (z[m](p + 1) - z[m + 1](pn))) / np.sqrt(
                        (x[m](p + 1) - x[m + 1](pn)) ** 2 + (z[m](p + 1) - z[m + 1](pn)) ** 2)

                step[0] = Vk0 / (Vk - Vk0)

            elif m == M - 2:
                p0 = k0[m - 1]
                p = k0[m]
                Mk = (z[m](p + delta) - z[m](p)) / (x[m](p + delta) - x[m](p))

                Vk0 = \
                    1 / c[-2] * ((x[m](p) - x[m - 1](p0)) + Mk * (z[m](p) - z[m - 1](p0))) / np.sqrt(
                        (x[m](p) - x[m - 1](p0)) ** 2 + (z[m](p) - z[m - 1](p0)) ** 2) + \
                    1 / c[-1] * ((x[m](p) - xF) + Mk * (z[m](p) - zF)) / np.sqrt(
                        (x[m](p) - xF) ** 2 + (z[m](p) - zF) ** 2)

                Vk = \
                    1 / c[-2] * ((x[m](p + 1) - x[m - 1](p0)) + Mk * (z[m](p + 1) - z[m - 1](p0))) / np.sqrt(
                        (x[m](p + 1) - x[m - 1](p0)) ** 2 + (z[m](p + 1) - z[m - 1](p0)) ** 2) + \
                    1 / c[-1] * ((x[m](p + 1) - xF) + Mk * (z[m](p + 1) - zF)) / np.sqrt(
                        (x[m](p + 1) - xF) ** 2 + (z[m](p + 1) - zF) ** 2)

                step[-1] = Vk0 / (Vk - Vk0)

            else:
                p0 = k0[m - 1]
                p = k0[m]
                pn = k0[m + 1]

                Mk = (z[m](p + delta) - z[m](p)) / (x[m](p + delta) - x[m](p))

                Vk0 = \
                    1 / c[m - 1] * ((x[m](p) - x[m - 1](p0)) + Mk * (z[m](p) - z[m - 1](p0))) / np.sqrt(
                        (x[m](p) - x[m - 1](p0)) ** 2 + (z[m](p) - z[m - 1](p0)) ** 2) + \
                    1 / c[m] * ((x[m](p) - x[m + 1](pn)) + Mk * (z[m](p) - z[m + 1](pn))) / np.sqrt(
                        (x[m](p) - x[m + 1](pn)) ** 2 + (z[m](p) - z[m + 1](pn)) ** 2)

                Vk = \
                    1 / c[m] * ((x[m](p + 1) - x[m - 1](p0)) + Mk * (z[m](p + 1) - z[m - 1](p0))) / np.sqrt(
                        (x[m](p + 1) - x[m - 1](p0)) ** 2 + (z[m](p + 1) - z[m - 1](p0)) ** 2) + \
                    1 / c[m + 1] * ((x[m](p + 1) - x[m + 1](pn)) + Mk * (z[m](p + 1) - z[m + 1](pn))) / np.sqrt(
                        (x[m](p + 1) - x[m + 1](pn)) ** 2 + (z[m](p + 1) - z[m + 1](pn)) ** 2)

                step[m] = Vk0 / (Vk - Vk0)

        k = k0 - step

        output['ki'].append(k)
        output['iter'] += 1
        #print(k)
        if np.linalg.norm(k - k0) <= tolerance:
            output['converged'] = True
            output['result'] = k
            break
        elif i == maxiter-1:
            output['converged'] = False
            output['result'] = k
        else:
            k0 = np.copy(k)
        output['elapsed_time'] = time.time() - t0
    return output

def parrilla_adapted_batch(x: list, z:list, xA_vec:list, zA_vec:list, xF_vec:list, zF_vec:list, c: list, tolerance: float=1e-4, maxiter: int=100, delta=1e-3):
    Nt, Nf = len(xA_vec), len(xF_vec)
    solutions = []

    t0 = time.time()
    for idx in range(Nt * Nf):
        t = idx // Nf  # Row index
        f = idx % Nf  # Column index

        xA, zA = xA_vec[t], zA_vec[t]
        xF, zF = xF_vec[f], zF_vec[f]

        solutions.append(parrilla_adapted(x, z, xA, zA, xF, zF, c, maxiter=5))
    elapsed_time = time.time() - t0
    return solutions, elapsed_time