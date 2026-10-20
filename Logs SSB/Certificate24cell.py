import numpy as np
import time
from scipy.io import loadmat
from pathlib import Path

PATH       = Path(__file__).resolve().parent.parent / 'SlackMat.mat'
start_time = time.time()
X_data     = loadmat(PATH)['S24cell']
X          = X_data.astype(np.int64)

assert np.array_equal(X, X_data)
assert X.shape == (24, 24)
assert np.array_equal(np.bincount(X.ravel()), [144, 288, 144])
assert np.array_equal(np.bincount(X[0]), [6, 12, 6])

# Symmetries preserving X.
S = 1 - X
B = [0, 9, 10, 11]
V = (S @ S.T)[:, B] // 6
W = S[B, :].T
assert np.array_equal(V @ W.T, 2 * S)
rows = {tuple(v): i for i, v in enumerate(V)}
cols = {tuple(w): j for j, w in enumerate(W)}
assert len(rows) == len(cols) == 24

transformations = [(np.diag([-1, 1, 1, 1]), 1)]
for k in range(3):
    T = np.eye(4, dtype=np.int64)
    T[[k, k + 1]] = T[[k + 1, k]]
    transformations.append((T, 1))
transformations.append((np.array([[1, 1, 1, 1], [1, 1, -1, -1],
                                 [1, -1, 1, -1], [1, -1, -1, 1]]), 2))

generators = []
for T, q in transformations:
    assert np.all((V @ T.T) % q == 0) and np.all((W @ T.T) % q == 0)
    r = np.array([rows[tuple(v)] for v in (V @ T.T) // q])
    c = np.array([cols[tuple(w)] for w in (W @ T.T) // q])
    assert np.array_equal(np.sort(r), np.arange(24))
    assert np.array_equal(np.sort(c), np.arange(24))
    assert np.array_equal(X[np.ix_(r, c)], X)
    generators.append((r, c))

G    = [(np.arange(24), np.arange(24))]
seen = {tuple(range(24)) + tuple(range(24))}
for r, c in G:
    for rg, cg in generators:
        rn, cn = rg[r], cg[c]
        key    = tuple(rn) + tuple(cn)
        if key not in seen:
            seen.add(key)
            G.append((rn, cn))

assert len(G) == 1152
assert len({int(r[0]) for r, c in G}) == 24
for r, c in G: assert np.array_equal(X[np.ix_(r, c)], X)
print(f'Symmetries: {len(G)} - transitive on the rows')

# Lower bound: C = 108 L, with C = 3 on X = 1 and C = 1 on X = 2.
C = np.where(X == 1, 3, np.where(X == 2, 1, 0))
assert int((C * X).sum()) == 1152

positive = np.flatnonzero(X[0] > 0).tolist()
twos     = np.flatnonzero(X[0] == 2).tolist()
counts   = np.fromiter((k.bit_count() for k in range(1 << 18)), dtype=np.int64)
maximum  = 0
nbconfig = 0

for mask in range(1 << 6):
    J2 = [j for k, j in enumerate(twos) if mask & (1 << k)]
    J1 = [j for j in positive if j not in J2]
    subsets = np.arange(1 << len(J1), dtype=np.int64)
    scores  = np.zeros(len(subsets), dtype=np.int64)
    for i in range(24):
        if np.any(X[i, J2] == 0): continue
        u2 = 1 if np.any(X[i, J2] == 1) else 2
        m0 = sum(1 << k for k, j in enumerate(J1) if X[i, j] == 0)
        m1 = sum(1 << k for k, j in enumerate(J1) if X[i, j] == 1)
        m2 = sum(1 << k for k, j in enumerate(J1) if X[i, j] == 2)
        values = 2 * int(C[i, J2].sum()) + 3 * counts[subsets & m1] + counts[subsets & m2]
        scores += u2 * values * ((subsets & m0) == 0)
    maximum   = max(maximum, int(scores.max()))
    nbconfig += len(subsets)

assert nbconfig == 2985984
assert maximum == 216
print(f'Lower bound: {nbconfig} configurations - max <L,R> = {maximum}/216 = 1')

# Upper bound: X = sum_g (R_A^g / 144 + R_B^g / 432).
u_A2, v_A = np.zeros(24, dtype=np.int64), np.zeros(24, dtype=np.int64)
u_B2, v_B = np.zeros(24, dtype=np.int64), np.zeros(24, dtype=np.int64)
u_A2[0]   = 2
u_A2[1:9] = 1
v_A[[12, 13, 14, 18, 19, 20]] = 2
u_B2[[0, 4, 5, 18, 19, 23]]  = 2
v_B[[3, 4, 11, 17, 21, 22]]  = 1
A2, B2 = np.outer(u_A2, v_A), np.outer(u_B2, v_B)
assert np.all((0 <= A2) & (A2 <= 2 * X))
assert np.all((0 <= B2) & (B2 <= 2 * X))

total = np.zeros_like(X)
for r, c in G: total += 3 * A2[np.ix_(r, c)] + B2[np.ix_(r, c)]
assert np.array_equal(total, 864 * X)
assert 3 * len(G) * (3 + 1) == 32 * 432
print('Upper bound: exact decomposition verified - total weight = 32/3')

print(f'SSB = 32/3 - time: {time.time() - start_time:.2f} seconds')
