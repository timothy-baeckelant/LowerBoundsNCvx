import itertools
import time
import numpy as np

start_time = time.time()
# Exact geometry and symmetry generators, in the certificate ordering.
VA = np.array([
    [-2, -2, -2],
    [2, 2, 2],
    [-1, 0, -1],
    [-2, -2, 2],
    [1, -1, 0],
    [-1, 0, 1],
    [2, -2, -2],
    [0, -1, -1],
    [-2, 2, -2],
    [-1, -1, 0],
    [1, 1, 0],
    [2, -2, 2],
    [0, 1, -1],
    [1, 0, -1],
    [-1, 1, 0],
    [0, -1, 1],
    [0, 1, 1],
    [-2, 2, 2],
    [2, 2, -2],
    [1, 0, 1],
], dtype=int)
VB = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, -1],
    [0, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 0, 0],
    [0, -1, 1],
    [0, 0, 0],
    [-1, 1, 0],
    [1, -1, 0],
    [0, 0, 0],
    [0, 1, 1],
    [-1, 0, -1],
    [-1, -1, 0],
    [0, -1, -1],
    [0, 1, -1],
    [0, 0, 0],
    [0, 0, 0],
    [-1, 0, 1],
], dtype=int)
NA = np.array([
    [-1, 0, -2],
    [1, 0, 2],
    [1, 0, -2],
    [2, -1, 0],
    [0, -2, 1],
    [0, 2, 1],
    [2, 1, 0],
    [-1, 0, 2],
    [-2, 1, 0],
    [0, 2, -1],
    [0, -2, -1],
    [-2, -1, 0],
], dtype=int)
NB = np.array([
    [-1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, -1],
    [0, -1, 0],
], dtype=int)
gen_rows = np.array([
    [18, 3, 1, 13, 19, 14, 12, 8, 10, 6, 17, 9, 7, 4, 2, 16, 11, 15, 5, 0],
    [14, 4, 8, 7, 18, 10, 13, 15, 17, 19, 2, 6, 1, 9, 3, 0, 12, 5, 16, 11],
], dtype=int)
gen_cols = np.array([
    [2, 7, 5, 8, 0, 11, 4, 10, 3, 1, 6, 9],
    [7, 2, 9, 10, 3, 1, 6, 4, 5, 8, 0, 11],
], dtype=int)

# Recover X=A+B sqrt(5) from the exact vertices and facet normals.
p, q = VA@NA.T + 5*VB@NB.T, VA@NB.T + VB@NA.T
assert np.all((8-3*p+5*q) % 8 == 0) and np.all((p-3*q) % 8 == 0)
A, B = (8-3*p+5*q)//8, (p-3*q)//8
assert A.shape == B.shape == (20, 12)


def nonnegative(a, b):
    # Exact sign of a + b sqrt(5), with integer coefficients.
    a, b = int(a), int(b)
    if a >= 0 and b >= 0: return True
    if a < 0 and b <= 0: return False
    return a*a >= 5*b*b if a >= 0 else 5*b*b >= a*a


# Exact regular geometry: vertices and normals are (P + Q sqrt(5))/2.
VP, VQ, NP, NQ = [], [], [], []
for signs in itertools.product([-1, 1], repeat=3):
    VP.append(2*np.array(signs)); VQ.append(np.zeros(3, dtype=int))
for k in range(3):
    for s, t in itertools.product([-1, 1], repeat=2):
        p, q = np.zeros(3, dtype=int), np.zeros(3, dtype=int)
        p[(k+1)%3], p[(k+2)%3], q[(k+1)%3], q[(k+2)%3] = s, -t, s, t
        VP.append(p); VQ.append(q)
        p, q = np.zeros(3, dtype=int), np.zeros(3, dtype=int)
        p[(k+1)%3], p[(k+2)%3], q[(k+2)%3] = 2*s, t, t
        NP.append(p); NQ.append(q)
assert {tuple(p)+tuple(q) for p, q in zip(VP, VQ)} == {
    tuple(p)+tuple(q) for p, q in zip(VA, VB)}
assert {tuple(p)+tuple(q) for p, q in zip(NP, NQ)} == {
    tuple(p)+tuple(q) for p, q in zip(NA, NB)}
p, q = VA@NA.T + 5*VB@NB.T, VA@NB.T + VB@NA.T
assert np.array_equal(p, 2*(3*(1-A)-5*B))
assert np.array_equal(q, 2*((1-A)-3*B))

levels = np.full(A.shape, -1, dtype=int)
for k, (a, b) in enumerate([(3, -1), (-1, 1), (2, 0)]):
    mask = (A == a) & (B == b)
    assert mask.sum() == 60
    levels[mask] = k
assert np.all((levels >= 0) | ((A == 0) & (B == 0)))
zero = levels < 0
assert np.all(zero.sum(axis=0) == 5) and np.all(zero.sum(axis=1) == 3)
edges = [(i, j) for i, j in itertools.combinations(range(20), 2) if (zero[i] & zero[j]).sum() == 2]
assert len(edges) == 30
for i, j in edges:
    p, q = VA[i]-VA[j], VB[i]-VB[j]
    assert p@p + 5*(q@q) == 24 and 2*(p@q) == -8
print('Exact regular dodecahedron slack matrix verified')

# Two generators give 120 simultaneous row and column symmetries.
G = [(np.arange(20), np.arange(12))]
seen = {tuple(range(20))+tuple(range(12))}
for r, c in G:
    for rg, cg in zip(gen_rows, gen_cols):
        rn, cn = rg[r], cg[c]
        key = tuple(rn)+tuple(cn)
        if key not in seen:
            seen.add(key); G.append((rn, cn))
assert len(G) == 120 and len({int(r[0]) for r, c in G}) == 20
for r, c in G:
    assert np.array_equal(np.sort(r), np.arange(20))
    assert np.array_equal(np.sort(c), np.arange(12))
    assert np.array_equal(A[np.ix_(r, c)], A)
    assert np.array_equal(B[np.ix_(r, c)], B)

# Lower bound: L=1/50 at X=2, 1/(10(3-sqrt(5))) at X=3-sqrt(5), zero otherwise.
# Put phi=(1+sqrt(5))/2 and b=3-sqrt(5). Enumerate v_j in {0,b,b*phi,b*phi**2}.
choices = [range(int(k)+2) for k in levels[0]]
states = np.array(list(itertools.product(*choices)), dtype=int)
assert len(states) == 13824
active = states > 0
exponents = states-1
allowed = ~np.any(active[:, None, :] & zero[None, :, :], axis=2)
row_exponents = np.maximum(0, np.where(active[:, None, :],
    exponents[:, None, :]-levels[None, :, :], 0).max(axis=2))
assert np.all(row_exponents[allowed] <= 2)
d = np.clip(exponents[:, None, :]-row_exponents[:, :, None], -2, 2)
support = allowed[:, :, None] & active[:, None, :]
small = support & (levels[None, :, :] == 0)
large = support & (levels[None, :, :] == 2)

# Powers phi**k = powers[k+4,0] + powers[k+4,1]*phi, for -4<=k<=2.
powers = np.array([[5,-3], [-3,2], [2,-1], [-1,1], [1,0], [0,1], [1,1]])
numerator = 5*np.where(small[:, :, :, None], powers[d+4], 0).sum(axis=(1, 2))
numerator += 2*np.where(large[:, :, :, None], powers[d+2], 0).sum(axis=(1, 2))
# 50<L,uv^T> = numerator[:,0] + numerator[:,1]*phi.
assert all(nonnegative(100-2*a-b, -b) for a, b in numerator)
assert np.any((numerator[:, 0] == 50) & (numerator[:, 1] == 0))
print('Exact separation: 13824 configurations, maximum <L,R> = 1')
assert 60*2 + 60*5 == 42*10  # Objective 60*2/50 + 60/10 = 42/5.

# Upper bound: X=sum_g (R_A^g/50 + R_B^g/20).
# u=(UP+UQ*sqrt(5))/2 and v=VP+VQ*sqrt(5).
UP, UQ = np.zeros((2, 20), dtype=int), np.zeros((2, 20), dtype=int)
VP, VQ = np.zeros((2, 12), dtype=int), np.zeros((2, 12), dtype=int)
UP[0, [0,1,5,7,8,11,12,13,15,16]] = 2
VP[0, [2,7]], VQ[0, [2,7]] = 3, -1
UP[1, [0,14]] = 2
UP[1, [3,9,13,15]], UQ[1, [3,9,13,15]] = -1, 1
UP[1, [7,8]], UQ[1, [7,8]] = 3, -1
VP[1, [1,6]] = 2
VP[1, [2,5]], VQ[1, [2,5]] = -1, 1
total_p, total_q = np.zeros_like(A), np.zeros_like(B)
for weight, up, uq, vp, vq in zip([2, 5], UP, UQ, VP, VQ):
    assert all(nonnegative(a, b) for a, b in zip(up, uq))
    assert all(nonnegative(a, b) for a, b in zip(vp, vq))
    p = np.outer(up, vp)+5*np.outer(uq, vq)
    q = np.outer(up, vq)+np.outer(uq, vp)
    assert all(nonnegative(a, b) for a, b in zip((2*A-p).ravel(), (2*B-q).ravel()))
    for r, c in G:
        total_p += weight*p[np.ix_(r, c)]
        total_q += weight*q[np.ix_(r, c)]
assert np.array_equal(total_p, 200*A) and np.array_equal(total_q, 200*B)
assert len(G)*(2+5) == 42*20  # 120*(1/50+1/20) = 42/5.
print('Exact rank-one decomposition verified')
print(f'SSB(X) = 42/5 = 8.4 - all decisive checks use integer arithmetic')
print(f'Time: {time.time()-start_time:.2f} seconds')
