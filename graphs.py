import numpy as np
from collections import deque

# ===============================
# Ensure connectivity
# ===============================
def is_connected(A):
    n = A.shape[0]
    visited = [False] * n
    q = deque([0])
    visited[0] = True

    while q:
        u = q.popleft()
        for v in range(n):
            if A[u, v] and not visited[v]:
                visited[v] = True
                q.append(v)

    return all(visited)


# ===============================
# Random connected graph with RNG
# ===============================
def random_connected_graph(n, p=0.3, rng=None):
    """
    n   : number of nodes
    p   : edge probability
    rng : np.random.Generator instance for reproducibility
    """
    if rng is None:
        rng = np.random.default_rng()  # fallback: random generator

    while True:
        A = np.zeros((n, n), dtype=np.int8)

        # Random edges using rng
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    A[i, j] = 1
                    A[j, i] = 1

        if is_connected(A):
            break

    T = np.full(n, -1, dtype=np.int8)
    return A, T
