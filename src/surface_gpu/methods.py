import time
import numpy as np

__all__ = ["parrilla_2007", "parrilla_generalized"]

def tof(k: int, x2: np.ndarray, z2: np.ndarray, x1: float, z1: float, c: float):
    Mk = (z2[k + 1] - z2[k]) / (x2[k + 1] - x2[k])
    return 1 / c * ((x2[k] - x1) + Mk * (z2[k] - z1)) / np.sqrt((x2[k] - x1) ** 2 + (z2[k] - z1) ** 2)


def parrilla_2007(surface, focus, transmitter, c1, c2):
    xf, zf = focus
    xt, zt = transmitter
    x1, z1 = surface

    k0 = 0

    for i in range(100):
        Vk0 = tof(k0, x1, z1, xt, zt, c1) + tof(k0, x1, z1, xf, zf, c2)
        Vk = tof(k0 + 1, x1, z1, xt, zt, c1) + tof(k0 + 1, x1, z1, xf, zf, c2)

        k = k0 - np.round(Vk0 / (Vk - Vk0))
        if np.abs(k - k0) <= 1:
            return k0
        else:
            k0 = int(k)


def parrilla_generalized_interpolated(x: list, z: list, xA: float, zA: float, xF: float, zF: float, c: list,
                                      tolerance: float=1e-4, maxiter: int=100, delta=1e-3):
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
                    1 / c[m] * ((x[m](p) - x[m - 1](p0)) + Mk * (z[m](p) - z[m - 1](p0))) / np.sqrt(
                        (x[m](p) - x[m - 1](p0)) ** 2 + (z[m](p) - z[m - 1](p0)) ** 2) + \
                    1 / c[m + 1] * ((x[m](p) - x[m + 1](pn)) + Mk * (z[m](p) - z[m + 1](pn))) / np.sqrt(
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

def parrilla_generalized(x: list, z: list, xA: float, zA: float, xF: float, zF: float, c: list, tolerance: int=2, maxiter: int=100):
    N = [len(xi) for xi in x]
    M = len(c)
    k0 = np.zeros(M-1, dtype=int)
    step = np.copy(k0)
    k = k0 + 1000
    elapsed_time = None

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
                Mk = (z[m][p + 1] - z[m][p]) / (x[m][p + 1] - x[m][p])

                Vk0 = \
                    1 / c[0] * ((x[m][p] - xA) + Mk * (z[m][p] - zA)) / np.sqrt(
                        (x[m][p] - xA) ** 2 + (z[m][p] - zA) ** 2) + \
                    1 / c[1] * ((x[m][p] - x[m + 1][pn]) + Mk * (z[m][p] - z[m + 1][pn])) / np.sqrt(
                        (x[m][p] - x[m + 1][pn]) ** 2 + (z[m][p] - z[m + 1][pn]) ** 2)

                Vk = \
                    1 / c[0] * ((x[m][p + 1] - xA) + Mk * (z[m][p + 1] - zA)) / np.sqrt(
                        (x[m][p + 1] - xA) ** 2 + (z[m][p + 1] - zA) ** 2) + \
                    1 / c[1] * ((x[m][p + 1] - x[m + 1][pn]) + Mk * (z[m][p + 1] - z[m + 1][pn])) / np.sqrt(
                        (x[m][p + 1] - x[m + 1][pn]) ** 2 + (z[m][p + 1] - z[m + 1][pn]) ** 2)

                step[0] = np.round(Vk0 / (Vk - Vk0))

            elif m == M - 2:
                p0 = k0[m - 1]
                p = k0[m]
                Mk = (z[m][p + 1] - z[m][p]) / (x[m][p + 1] - x[m][p])

                Vk0 = \
                    1 / c[-2] * ((x[m][p] - x[m - 1][p0]) + Mk * (z[m][p] - z[m - 1][p0])) / np.sqrt(
                        (x[m][p] - x[m - 1][p0]) ** 2 + (z[m][p] - z[m - 1][p0]) ** 2) + \
                    1 / c[-1] * ((x[m][p] - xF) + Mk * (z[m][p] - zF)) / np.sqrt(
                        (x[m][p] - xF) ** 2 + (z[m][p] - zF) ** 2)

                Vk = \
                    1 / c[-2] * ((x[m][p + 1] - x[m - 1][p0]) + Mk * (z[m][p + 1] - z[m - 1][p0])) / np.sqrt(
                        (x[m][p + 1] - x[m - 1][p0]) ** 2 + (z[m][p + 1] - z[m - 1][p0]) ** 2) + \
                    1 / c[-1] * ((x[m][p + 1] - xF) + Mk * (z[m][p + 1] - zF)) / np.sqrt(
                        (x[m][p + 1] - xF) ** 2 + (z[m][p + 1] - zF) ** 2)

                step[-1] = np.round(Vk0 / (Vk - Vk0))

            else:
                p0 = k0[m - 1]
                p = k0[m]
                pn = k0[m + 1]
                Mk = (z[m][p + 1] - z[m][p]) / (x[m][p + 1] - x[m][p])

                Vk0 = \
                    1 / c[m] * ((x[m][p] - x[m - 1][p0]) + Mk * (z[m][p] - z[m - 1][p0])) / np.sqrt(
                        (x[m][p] - x[m - 1][p0]) ** 2 + (z[m][p] - z[m - 1][p0]) ** 2) + \
                    1 / c[m + 1] * ((x[m][p] - x[m + 1][pn]) + Mk * (z[m][p] - z[m + 1][pn])) / np.sqrt(
                        (x[m][p] - x[m + 1][pn]) ** 2 + (z[m][p] - z[m + 1][pn]) ** 2)

                Vk = \
                    1 / c[m] * ((x[m][p + 1] - x[m - 1][p0]) + Mk * (z[m][p + 1] - z[m - 1][p0])) / np.sqrt(
                        (x[m][p + 1] - x[m - 1][p0]) ** 2 + (z[m][p + 1] - z[m - 1][p0]) ** 2) + \
                    1 / c[m + 1] * ((x[m][p + 1] - x[m + 1][pn]) + Mk * (z[m][p + 1] - z[m + 1][pn])) / np.sqrt(
                        (x[m][p + 1] - x[m + 1][pn]) ** 2 + (z[m][p + 1] - z[m + 1][pn]) ** 2)

                step[m] = np.round(Vk0 / (Vk - Vk0))

        k = k0 - step
        k = np.array([np.max([np.min([k[i], N[i] - 2]), 0]) for i in range(M-1)])

        output['ki'].append(k)
        output['iter'] += 1
        #print(k)
        if np.all(np.abs(k - k0) <= tolerance):
            output['elapsed_time'] = time.time() - t0
            output['converged'] = True
            output['result'] = k
            break
        elif i == maxiter-1:
            output['elapsed_time'] = -1
            output['converged'] = False
            output['result'] = k
        else:
            k0 = np.copy(k)
    return output