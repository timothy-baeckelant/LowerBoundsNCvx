"""Verify the four nonnegative factorizations in Appendix A.3.
An optional argument selects a different path to SlackMat.mat.
Requires SciPy and SymPy. No optimization solver is used.

The JSON stores W and H in multiplication order, transposing the W^T
printed in the appendix where necessary. Letters are local to each example.
For the two golden-ratio matrices, the tables below specify which exact
slack value each distinct double in SlackMat.mat represents. Lookup uses
the stored values directly, with no tolerance or nearest-value rounding.
"""

import argparse
import json
from pathlib import Path

from scipy.io import loadmat
from sympy import Integer, Matrix, expand, sqrt


HERE = Path(__file__).resolve().parent
PHI = (1 + sqrt(5)) / 2
ONE_OVER_PHI = PHI - 1
ONE_OVER_PHI_SQUARED = 2 - PHI

# W and H use the symbols defined separately for each example in the paper.
DODECA = {
    "a": 2 * ONE_OVER_PHI_SQUARED,
    "b": ONE_OVER_PHI,
    "c": PHI,
    "d": 2 * ONE_OVER_PHI,
}
ICOSIDODECA = {
    "a": ONE_OVER_PHI_SQUARED,
    "b": ONE_OVER_PHI,
    "c": PHI,
    "d": 2 * ONE_OVER_PHI,
    "e": 1 + ONE_OVER_PHI_SQUARED,
    "f": ONE_OVER_PHI - ONE_OVER_PHI_SQUARED,
}

# The target is read independently of W and H, preserving the benchmark's
# row and column ordering. The keys are the numeric levels in SlackMat.mat.
INTEGER_SLACKS = {i: Integer(i) for i in range(3)}
DODECA_SLACKS = {
    0: Integer(0), 0.76393202250021: 2 * ONE_OVER_PHI_SQUARED,
    1.23606797749979: 2 * ONE_OVER_PHI, 2: Integer(2),
}
ICOSIDODECA_SLACKS = {
    0: Integer(0), 0.381966011250105: ONE_OVER_PHI_SQUARED,
    0.618033988749895: ONE_OVER_PHI, 1: Integer(1),
    1.38196601125011: 1 + ONE_OVER_PHI_SQUARED,
    1.61803398874989: PHI, 2: Integer(2),
}

# Each shape is (number of rows, factorization size, number of columns).
CASES = {
    "Sdodecahedron": ((20, 9, 12), DODECA, DODECA_SLACKS),
    "S24cell": ((24, 12, 24), {}, INTEGER_SLACKS),
    "Sicosidodecahedron": ((30, 12, 32), ICOSIDODECA, ICOSIDODECA_SLACKS),
    "Scuboctahedron": ((12, 8, 14), {}, INTEGER_SLACKS),
}


def read_factor(rows, symbols):
    return Matrix([[Integer(v) if type(v) is int else symbols[v] for v in row]
                   for row in rows])


def verify(name, factors, stored):
    (m, r, n), symbols, slack_values = CASES[name]
    w = read_factor(factors["W"], symbols)
    h = read_factor(factors["H"], symbols)
    if (w.shape, h.shape, stored.shape) != ((m, r), (r, n), (m, n)):
        raise ValueError(f"{name}: unexpected dimensions {w.shape}, {h.shape}, {stored.shape}")
    for value in set(w) | set(h):
        if value.is_nonnegative is not True:
            raise ValueError(f"{name}: factor entry {value} is not certified nonnegative")

    product = w * h
    for i in range(m):
        for j in range(n):
            level = stored[i, j]
            if level not in slack_values:
                raise ValueError(f"{name}: unknown slack value {level!r} at ({i+1}, {j+1})")
            residual = expand(product[i, j] - slack_values[level])
            if residual != 0:
                raise ValueError(f"{name}: nonzero exact residual {residual} at ({i+1}, {j+1})")
    print(f"{name}: W and H nonnegative, W*H = S exactly ({m} x {n}, size {r})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mat_file", nargs="?", type=Path, default=HERE.parent / "SlackMat.mat")
    args = parser.parse_args()
    factors = json.loads((HERE / "factors.json").read_text(encoding="utf-8"))
    stored = loadmat(args.mat_file, variable_names=list(CASES))
    for name in CASES:
        verify(name, factors[name], stored[name])


if __name__ == "__main__":
    main()
