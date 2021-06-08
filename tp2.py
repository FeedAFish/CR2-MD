# ---------------------------------------------------------------------------- #
#                          TP2 Modélisation MIC3 - GMM                         #
# ---------------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import tp2_lib as tp
import tp2_methodes_lib as meth


# ---------------------------------------------------------------------------- #
#                           Finite Difference Method                           #
# ---------------------------------------------------------------------------- #

# ------------------------------ Initialize Data ----------------------------- #
L = 1.0
List_K = [10, 20, 30]
List_tmax = [0.5, 1.0, 1.5]
num1 = 0
num2 = 0
T0DF = np.zeros([3, 11])
T1DF = np.zeros([3, 11])
T2DF = np.zeros([3, 11])
yDF = np.zeros(11)

for tmax in List_tmax:
    num1 = num1 + 1
    num3 = -1
    for K in List_K:
        num2 = num2 + 1
        num3 = num3 + 1

        # Compute T
        T = meth.methodeDF(K, tmax, L)

        # Convert vector T into a (K+1)x(K+1) Matrix
        T2D = tp.matT(T, K, tp.indk)
        # print(T2D)

        # Maillage
        x = np.linspace(0, L, K + 1)
        y = np.linspace(0, L, K + 1)
        X, Y = np.meshgrid(x, y)

        # Temperature profile extraction at x = L/4
        i0m = int(np.floor(K / 4.0))
        i0p = i0m + 1
        omega = K / 4.0 - i0m
        T0 = (1 - omega) * T2D[i0m, :] + omega * T2D[i0p, :]
        if num3 == 0:
            T00 = deepcopy(T0)
            y0 = deepcopy(y)
            # Save T0 in T0DF to serve as a reference solution for MMC
            T0DF[num1 - 1, :] = T0
            yDF = y0
        elif num3 == 1:
            T01 = deepcopy(T0)
            y1 = deepcopy(y)
        else:
            T02 = deepcopy(T0)
            y2 = deepcopy(y)

        # Temperature profile extraction at x = L/2
        i0m = int(np.floor(K / 2.0))
        i0p = i0m + 1
        omega = K / 2.0 - i0m
        T1 = (1 - omega) * T2D[i0m, :] + omega * T2D[i0p, :]
        if num3 == 0:
            T10 = deepcopy(T1)
            # Save T1 in T1DF to serve as a reference solution for MMC
            T1DF[num1 - 1, :] = T1
        elif num3 == 1:
            T11 = deepcopy(T1)
        else:
            T12 = deepcopy(T1)

        # Temperature profile extraction at x = 3L/4
        i0m = int(np.floor(3 * K / 4.0))
        i0p = i0m + 1
        omega = 3 * K / 4.0 - i0m
        T2 = (1 - omega) * T2D[i0m, :] + omega * T2D[i0p, :]
        if num3 == 0:
            T20 = deepcopy(T2)
            # Save T2 in T2DF to serve as a reference solution for MMC
            T2DF[num1 - 1, :] = T2
        elif num3 == 1:
            T21 = deepcopy(T2)
        else:
            T22 = deepcopy(T2)

        # Temperature field
        fig = plt.figure("ChampT_DF_" + str(num2))
        NbIso = 25
        h = L / K
        CF = plt.contourf(X, Y, np.transpose(T2D), NbIso)
        # plt.clabel(CF, colors = 'k', fmt = '%2.1f', fontsize=12)
        plt.colorbar(CF)
        plt.title("Temperature field at t = " + str(tmax) + " for h = " + str(h))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("ChampT_DF_" + str(num2) + ".png")

    fig = plt.figure("ProfilT_DF_025_" + str(num1))
    plt.plot(y0, T00, "-bo", y1, T01, "-rs", y2, T02, "-gv")
    plt.xlabel("y")
    plt.ylabel("T")
    plt.title("Temperature Profile at t = " + str(tmax) + " and x = L/4")
    plt.legend(["h=L/10", "h=L/20", "h=L/30"], loc="best")
    plt.savefig("ProfilT_DF_025_" + str(num1) + ".png")

    fig = plt.figure("ProfilT_DF_050_" + str(num1))
    plt.plot(y0, T10, "-bo", y1, T11, "-rs", y2, T12, "-gv")
    plt.xlabel("y")
    plt.ylabel("T")
    plt.title("Temperature Profile at t = " + str(tmax) + " and x = L/2")
    plt.legend(["h=L/10", "h=L/20", "h=L/30"], loc="best")
    plt.savefig("ProfilT_DF_050_" + str(num1) + ".png")

    fig = plt.figure("ProfilT_DF_075_" + str(num1))
    plt.plot(y0, T20, "-bo", y1, T21, "-rs", y2, T22, "-gv")
    plt.xlabel("y")
    plt.ylabel("T")
    plt.title("Temperature Profile at t = " + str(tmax) + " and x = 3L/4")
    plt.legend(["h=L/10", "h=L/20", "h=L/30"], loc="best")
    plt.savefig("ProfilT_DF_075_" + str(num1) + ".png")


# ---------------------------------------------------------------------------- #
#                              Monte Carlo Method                              #
# ---------------------------------------------------------------------------- #

# ------------------------------ Initialize Data ----------------------------- #
L = 1.0
List_K = [10000, 100000]
List_M = [10, 20]
List_tmax = [0.5, 1.0, 1.5]
num1 = 0
num2 = 0

for tmax in List_tmax:
    num1 = num1 + 1
    num3 = -1
    for K in List_K:
        for M in List_M:
            num2 = num2 + 1
            num3 = num3 + 1

            # Compute T
            T2D = meth.methodeMC(K, M, tmax, L)

            # Maillage
            eps = L / M
            x = np.linspace(eps / 2, L - eps / 2, M)
            y = np.linspace(eps / 2, L - eps / 2, M)
            X, Y = np.meshgrid(x, y)

            # Temperature profile extraction at x = L/4
            i0m = int(np.floor(M / 4.0))
            i0p = i0m + 1
            omega = M / 4.0 - i0m
            T0 = (1 - omega) * T2D[i0m, :] + omega * T2D[i0p, :]
            if num3 == 0:
                T00mc = deepcopy(T0)
                y0mc = deepcopy(y)
            elif num3 == 1:
                T01mc = deepcopy(T0)
                y1mc = deepcopy(y)
            elif num3 == 2:
                T02mc = deepcopy(T0)
                y2mc = deepcopy(y)
            else:
                T03mc = deepcopy(T0)
                y3mc = deepcopy(y)

            # Temperature profile extraction at x = L/2
            i0m = int(np.floor(M / 2.0))
            i0p = i0m + 1
            omega = M / 2.0 - i0m
            T1 = (1 - omega) * T2D[i0m, :] + omega * T2D[i0p, :]
            if num3 == 0:
                T10mc = deepcopy(T1)
            elif num3 == 1:
                T11mc = deepcopy(T1)
            elif num3 == 2:
                T12mc = deepcopy(T1)
            else:
                T13mc = deepcopy(T1)

            # Temperature profile extraction at x = 3L/4
            i0m = int(np.floor(3 * M / 4.0))
            i0p = i0m + 1
            omega = 3 * M / 4.0 - i0m
            T2 = (1 - omega) * T2D[i0m, :] + omega * T2D[i0p, :]
            if num3 == 0:
                T20mc = deepcopy(T2)
            elif num3 == 1:
                T21mc = deepcopy(T2)
            elif num3 == 2:
                T22mc = deepcopy(T2)
            else:
                T23mc = deepcopy(T2)

            # Temperature field
            fig = plt.figure("ChampT_MMC_" + str(num2))
            NbIso = 25
            h = L / M
            CF = plt.contourf(X, Y, np.transpose(T2D), NbIso)
            plt.colorbar(CF)
            plt.title(
                "Temperature field at t = "
                + str(tmax)
                + " for h = "
                + str(h)
                + " and K = "
                + str(K)
            )
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig("ChampT_MMC_" + str(num2) + ".png")

    fig = plt.figure("ProfilT_MMC_025" + str(num1))
    plt.plot(
        yDF,
        T0DF[num1 - 1, :],
        "--k",
        y0mc,
        T00mc,
        "-ro",
        y1mc,
        T01mc,
        "-rv",
        y2mc,
        T02mc,
        "-go",
        y3mc,
        T03mc,
        "-gv",
    )
    plt.xlabel("y")
    plt.ylabel("T")
    plt.title("Temperature Profile at t = " + str(tmax) + " and x = L/4")
    plt.legend(
        [
            "Finite Difference",
            "K=10000 , eps=1/10",
            "K=10000 , eps=1/20",
            "K=100000 , eps=1/10",
            "K=100000 , eps=1/20",
        ],
        loc="best",
    )
    plt.savefig("ProfilT_MMC_025_" + str(num1) + ".png")

    fig = plt.figure("ProfilT_MMC_050" + str(num1))
    plt.plot(
        yDF,
        T1DF[num1 - 1, :],
        "--k",
        y0mc,
        T10mc,
        "-ro",
        y1mc,
        T11mc,
        "-rv",
        y2mc,
        T12mc,
        "-go",
        y3mc,
        T13mc,
        "-gv",
    )
    plt.xlabel("y")
    plt.ylabel("T")
    plt.title("Temperature Profile at t = " + str(tmax) + " and x = L/2")
    plt.legend(
        [
            "Finite Difference",
            "K=10000 , eps=1/10",
            "K=10000 , eps=1/20",
            "K=100000 , eps=1/10",
            "K=100000 , eps=1/20",
        ],
        loc="best",
    )
    plt.savefig("ProfilT_MMC_050_" + str(num1) + ".png")

    fig = plt.figure("ProfilT_MMC_075" + str(num1))
    plt.plot(
        yDF,
        T2DF[num1 - 1, :],
        "--k",
        y0mc,
        T20mc,
        "-ro",
        y1mc,
        T21mc,
        "-rv",
        y2mc,
        T22mc,
        "-go",
        y3mc,
        T23mc,
        "-gv",
    )
    plt.xlabel("y")
    plt.ylabel("T")
    plt.title("Temperature Profile at t = " + str(tmax) + " and x = 3L/4")
    plt.legend(
        [
            "Finite Difference",
            "K=10000 , eps=1/10",
            "K=10000 , eps=1/20",
            "K=100000 , eps=1/10",
            "K=100000 , eps=1/20",
        ],
        loc="best",
    )
    plt.savefig("ProfilT_MMC_075_" + str(num1) + ".png")

plt.show()
