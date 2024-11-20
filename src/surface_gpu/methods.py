import numpy as np


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


def Vk_profile(k0: np.ndarray, p: int, profiles: list):
    k_past = int(k0[p-1])
    k_post = int(k0[p+1])
    k = int(k0[p])
    Vk0 = tof(k, profiles[p + 1]["x"], profiles[p + 1]["z"], profiles[p]["x"][k_past], profiles[p]["z"][k_past],
              profiles[p + 1]["c_inc"]) + \
          tof(k, profiles[p + 1]["x"], profiles[p + 1]["z"], profiles[p + 2]["x"][k_post], profiles[p + 2]["z"][k_post],
              profiles[p + 1]["c_ref"])
    return Vk0


def tof_profiles(k0: np.ndarray, profiles: list, emitter: list, focus: list):
    k = np.zeros_like(k0 + 2, dtype=int)
    elements = emitter + profiles + focus
    for p in range(1, len(elements) - 1):
        Vk0 = Vk_profile(k0, p, elements)
        post_k0 = k0 + 1
        post_k0[0] = 0
        post_k0[-1] = 0
        Vk = Vk_profile(post_k0, p, elements)
        k[p] = k0[p] - np.round(Vk0 / (Vk - Vk0))
    return k


def parrilla_2007_generalized(profiles: list, emitter: list, focus: list, maxiter=100):
    k0 = np.zeros(len(profiles) + 2, dtype=int)
    N_vector = [len(p["x"]) for p in profiles]
    for i in range(maxiter):
        k = tof_profiles(k0, profiles, emitter, focus)

        if np.all(np.abs(k - k0) <= 1):
            return k[1:-1]
        else:
            k0[1:-1] = np.array([np.min([k[j+1], N_vector[j]-2]) for j in range(len(k)-2)])
    return k[1:-1]