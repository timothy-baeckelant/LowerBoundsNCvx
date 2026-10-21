import itertools
import time
import numpy as np

start_time = time.time()

# Exact regular geometry, in the ordering of Scuboctahedron in SlackMat.mat.
vertices = np.array([
    [0,-1,-1], [-1,0,-1], [1,0,-1], [-1,-1,0], [1,-1,0], [0,1,-1],
    [0,-1,1], [-1,1,0], [1,1,0], [-1,0,1], [1,0,1], [0,1,1]])
normals = np.array([
    [-1,0,0], [1,0,0], [0,1,0], [0,0,-1], [0,-1,0], [0,0,1],
    [1,1,1], [1,-1,-1], [1,1,-1], [1,-1,1],
    [-1,1,1], [-1,-1,-1], [-1,1,-1], [-1,-1,1]])
standard_vertices = set()
standard_normals = set(itertools.product([-1, 1], repeat=3))
for k in range(3):
    for s, t in itertools.product([-1, 1], repeat=2):
        v = [0, 0, 0]; v[(k+1)%3] = s; v[(k+2)%3] = t
        standard_vertices.add(tuple(v))
    for s in [-1, 1]:
        v = [0, 0, 0]; v[k] = s; standard_normals.add(tuple(v))
assert set(map(tuple, vertices)) == standard_vertices
assert set(map(tuple, normals)) == standard_normals
offsets = np.where(np.count_nonzero(normals, axis=1) == 1, 1, 2)
slack = offsets - vertices@normals.T
assert np.all(slack % offsets == 0)
X = slack // offsets
assert X.shape == (12, 14) and np.all(np.isin(X, [0, 1, 2]))
assert np.array_equal(np.bincount(X.ravel()), [48, 72, 48])
assert np.all((X[:, offsets == 1] == 0).sum(axis=0) == 4)
assert np.all((X[:, offsets == 2] == 0).sum(axis=0) == 3)
print('Exact regular cuboctahedron slack matrix verified')

# Signed coordinate permutations give the 48 symmetries.
row_index = {tuple(v):i for i, v in enumerate(vertices)}
col_index = {tuple(v):j for j, v in enumerate(normals)}
G = []
for permutation in itertools.permutations(range(3)):
    for signs in itertools.product([-1, 1], repeat=3):
        r = np.array([row_index[tuple(v)] for v in vertices[:, permutation]*signs])
        c = np.array([col_index[tuple(v)] for v in normals[:, permutation]*signs])
        assert np.array_equal(X[np.ix_(r, c)], X)
        assert np.array_equal(offsets[c], offsets)
        G.append((r, c))
assert len({tuple(r)+tuple(c) for r, c in G}) == 48
assert len({int(r[0]) for r, c in G}) == 12

# L=C/120: 3/40 at X=1; 1/24 or 1/120 at X=2 for square or triangle columns.
C = 9*(X == 1) + (X == 2)*np.where(offsets == 1, 5, 1)
assert np.all(C >= 0)
assert 5*int((C*X).sum()) == 39*120

# After normalization and rounding, u_i is in {0,1/2,1}, v_j in {0,1,2}.
# Transitivity allows u_0=1, hence 0<=v_j<=X[0,j].
V = np.array(list(itertools.product(*[range(int(x)+1) for x in X[0]])))
assert len(V) == 5184
U = np.full((len(V), 12), 2, dtype=int)  # U=2u, chosen as large as possible.
for j in range(14):
    active = V[:, j] > 0
    U[active] = np.minimum(U[active], 2*X[None, :, j]//V[active, j, None])
assert np.all(U[:, 0] == 2)
scores = np.einsum('ti,ij,tj->t', U, C, V)
assert scores.max() == 240  # <L,uv^T> = scores/240.
print('Exact separation: 5184 configurations, maximum <L,R> = 1')

# Three atoms, with u=U/2 and integer v.
U, V = np.zeros((3, 12), dtype=int), np.zeros((3, 14), dtype=int)
U[0, [0,1,2,5]] = 2
V[0, 5] = 2; V[0, [6,9,10,13]] = 1
U[1, 0] = 2; U[1, [1,2,3,4]] = 1
V[1, [2,5,6,10]] = 2
U[2, [0,2,8,11]] = 2
V[2, [0,9,12,13]] = 1
total = np.zeros_like(X)
for weight, u, v in zip([1, 8, 4], U, V):
    R = np.outer(u, v)  # R=2 times the actual atom.
    assert np.all(R >= 0) and np.all(R <= 2*X)
    assert int((C*R).sum()) == 240
    for r, c in G: total += weight*R[np.ix_(r, c)]
assert np.array_equal(total, 160*X)
# X=sum_g(R_A^g/80 + R_B^g/10 + R_C^g/20).
assert 5*len(G)*(1+8+4) == 39*80
print('Exact rank-one decomposition verified')
print('SSB(X) = 39/5 = 7.8 - all decisive checks use integer arithmetic')
print(f'Time: {time.time()-start_time:.2f} seconds')
