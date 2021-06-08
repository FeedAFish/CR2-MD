import numpy as np

import tp2_lib as tp


def methodeDF(K, tmax, L):
    # ---------------------------------------------------------------------------- #
    #                           Finite Difference Method                           #
    # ---------------------------------------------------------------------------- #

    # -------------------------- Continuous Problem Data ------------------------- #
    V0 = 1.0
    D = 0.2
    Tbas = 0.0  # (BC en y = 0)
    Thaut = 1.0  # (BC en y = L)

    # --------------------------- Numerical parameters --------------------------- #
    K2 = (K + 1) ** 2
    h = L / K
    dt = 0.8 * h * h / (4 * D)

    # ---------------------------- Initialisation of T --------------------------- #
    T = np.zeros(K2)

    # --------------------- Assembling of the Rigth Hand Side -------------------- #
    S = tp.secmb(K, dt, h, L, Tbas, Thaut, tp.indk, tp.source)

    # ---------------------- Assembling of the Scheme Matrix --------------------- #
    A = tp.MatA(L, V0, D, h, dt, K, tp.indk, tp.vit)

    # --------------------------------- Time Loop -------------------------------- #
    time = 0.0
    while time < tmax:
        T = A.dot(T) + S
        time = time + dt

    return T


def methodeMC(K, M, tmax, L):
    # ---------------------------------------------------------------------------- #
    #                              Monte Carlo Method                              #
    # ---------------------------------------------------------------------------- #

    # -------------------------- Continuous Problem Data ------------------------- #
    V0 = 1.0
    D = 0.2
    Tbas = 0.0  # (BC en y = 0)
    Thaut = 1.0  # (BC en y = L)

    # --------------------------- Numerical parameters --------------------------- #
    eps = L / M
    dt = 0.25 * eps * eps / D

    # ---------------------- Initialisation of Theta and X,Y --------------------- #
    X, Y = tp.posinit(K, L)
    Theta = np.zeros(K)

    # --------------------------------- Time Loop -------------------------------- #
    time = 0.0
    while time < tmax:
        time += dt
        X, Y, Theta = tp.evolution(
            X, Y, Theta, K, dt, L, D, V0, Tbas, Thaut, tp.source, tp.vit
        )

    # ------------------------- Compute Temperaure field ------------------------- #
    T = tp.tmoy(X, Y, Theta, K, M, L)

    return T
