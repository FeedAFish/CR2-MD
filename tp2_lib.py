import numpy as np
import scipy.sparse as spsp  # # Sparse matrix


def indk(i, j, K):
    return i + (K + 1) * j


def vit(x, y, V0, L):
    V = np.zeros(2)
    V[0] = -V0 * np.sin((np.pi * x) / L) * np.cos((np.pi * y) / L)
    V[1] = V0 * np.cos((np.pi * x) / L) * np.sin((np.pi * y) / L)
    return V


def source(x, y, L):
    return 256 * (x / L) ** 2 * (y / L) ** 2 * (1 - x / L) ** 2 * (1 - y / L) ** 2


def MatA(L, V0, D, h, dt, K, indk, vit):

    K2 = (K + 1) ** 2
    A = spsp.lil_matrix((K2, K2))  # Declaration of A as a sparse matrix

    # Loop on the internal nodes
    for i in range(1, K):
        for j in range(1, K):
            k = indk(i, j, K)
            ke = indk(i + 1, j, K)
            ko = indk(i - 1, j, K)
            kn = indk(i, j + 1, K)
            ks = indk(i, j - 1, K)
            A[k, k] = 1 - (4 * dt * D) / h ** 2
            A[k, ke] = -(dt * vit(i * h, j * h, V0, L)[0]) / (2 * h) + dt * D / h ** 2
            A[k, ko] = (dt * vit(i * h, j * h, V0, L)[0]) / (2 * h) + dt * D / h ** 2
            A[k, kn] = -(dt * vit(i * h, j * h, V0, L)[1]) / (2 * h) + dt * D / h ** 2
            A[k, ks] = (dt * vit(i * h, j * h, V0, L)[1]) / (2 * h) + dt * D / h ** 2

    # Loop on the nodes located in x = 0 (Neumann Boundary Condition)
    for j in range(1, K):
        k = indk(0, j, K)
        ke = indk(1, j, K)
        kn = indk(0, j + 1, K)
        ks = indk(0, j - 1, K)
        A[k, k] = 1 - (3 * dt * D) / h ** 2
        A[k, ke] = dt * D / h ** 2
        A[k, kn] = -(dt * vit(i * h, j * h, V0, L)[1]) / (2 * h) + dt * D / h ** 2
        A[k, ks] = (dt * vit(i * h, j * h, V0, L)[1]) / (2 * h) + dt * D / h ** 2

    # Loop on the nodes located in x = L (Neumann Boundary Condition)
    for j in range(1, K):
        k = indk(K, j, K)
        ko = indk(K - 1, j, K)
        kn = indk(K, j + 1, K)
        ks = indk(K, j - 1, K)
        A[k, k] = 1 - (3 * dt * D) / h ** 2
        A[k, ko] = dt * D / h ** 2
        A[k, kn] = -(dt * vit(i * h, j * h, V0, L)[1]) / (2 * h) + dt * D / h ** 2
        A[k, ks] = (dt * vit(i * h, j * h, V0, L)[1]) / (2 * h) + dt * D / h ** 2

    return A


def secmb(K, dt, h, L, Tbas, Thaut, indk, source):
    S = np.zeros((K + 1) ** 2)
    for i in range(K + 1):
        for j in range(K + 1):
            S[indk(i, j, K)] = dt * source(i * h, j * h, L)
            if j == K:
                S[indk(i, j, K)] = Thaut
            if j == 0:
                S[indk(i, j, K)] = Tbas
    return S


def matT(T, K, indk):
    T2D = np.zeros((K + 1, K + 1))
    for i in range(K + 1):
        for j in range(K + 1):
            T2D[i, j] = T[indk(i, j, K)]
    return T2D


def evolution(X, Y, Theta, K, dt, L, D, V0, Tbas, Thaut, source, vit):

    Vit = np.zeros([K, 2])
    S = np.zeros(K)
    alpha = np.random.randn(K, 2)

    Y_temp = np.zeros(K)

    for k in range(0, K):
        V = vit(X[k], Y[k], V0, L)
        Vit[k, 0] = V[0]
        Vit[k, 1] = V[1]
        S[k] = source(X[k], Y[k], L)
        X[k] = X[k] + dt * Vit[k, 0] + np.sqrt(2 * D * dt) * alpha[k, 0]
        Y[k] = Y[k] + dt * Vit[k, 1] + np.sqrt(2 * D * dt) * alpha[k, 1]
        Y_temp[k] = Y[k]
        Theta[k] = Theta[k] + dt * S[k]
        X[k] = min(L, max(X[k], 0))
        Y[k] = min(L, max(Y[k], 0))
        if Y_temp[k] <= 0:
            Theta[k] = Tbas
        elif Y_temp[k] >= L:
            Theta[k] = Thaut
    return X, Y, Theta


def tmoy(X, Y, Theta, K, M, L):

    Tm = np.zeros([M, M])  # mean temperature in a cell
    nbp = np.zeros([M, M])  # number of particles in a cell

    eps = L / M

    for k in range(0, K):
        i = min(int(X[k] / eps), M - 1)
        j = min(int(Y[k] / eps), M - 1)
        nbp[i, j] += 1
        Tm[i, j] += Theta[k]

    Tm = Tm / nbp
    return Tm


def posinit(K, L):
    X = np.random.uniform(0, L, K)
    Y = np.random.uniform(0, L, K)
    return X, Y
