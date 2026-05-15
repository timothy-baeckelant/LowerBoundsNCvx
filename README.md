# Computing Lower Bounds on the Nonnegative Rank via Non-Convex Optimization Solvers

This repository contains Python code accompanying the paper:

**Computing Lower Bounds on the Nonnegative Rank via Non-Convex Optimization Solvers**  
Timothy Baeckelant, Arnaud Vandaele, Nicolas Gillis, 2026.

The code implements four bounds considered in the paper:

- **FSB**: fooling set bound;
- **HSB**: hyperplane separation bound;
- **SSB**: self-scaled bound;
- **RCB**: rectangle covering bound.

The code supports both fully specified nonnegative matrices and partial matrices.  
In the Python implementation, entries equal to `-1` represent unspecified entries.

## Repository content

The main file is:

```text
LowerBounds.py
```

It contains:

- definitions of several benchmark matrices;
- implementations of the four bounds `FSB`, `HSB`, `SSB`, and `RCB`;
- test functions for computing the fooling set bound and the rectangle covering bound;
- a running example based on the matrix `LEDM(5)`.

The file

```text
SlackMat.mat
```

contains additional slack matrices used in the experiments:

- `Sdodecahedron`;
- `Scuboctahedron`;
- `Sicosidodecahedron`;
- `S24cell`.

## Requirements

The code requires:

- Python 3;
- NumPy;
- SciPy;
- Gurobi;
- gurobipy.

A valid Gurobi license is required.  
Information about Gurobi installation and licensing is available from the official Gurobi website.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The file `requirements.txt` should contain:

```text
numpy
scipy
gurobipy
```

## Running the example

To run the example from the paper, execute:

```bash
python LowerBounds.py
```

This computes the four bounds on the running example of the paper, namely the linear Euclidean distance matrix of size 5 x 5, denoted by `LEDM(5)`.

This matrix is defined by

```text
M_ij = (i-j)^2.
```

With zero-based indexing, this gives:

```text
[[ 0.  1.  4.  9. 16.]
 [ 1.  0.  1.  4.  9.]
 [ 4.  1.  0.  1.  4.]
 [ 9.  4.  1.  0.  1.]
 [16.  9.  4.  1.  0.]]
```

The expected values are:

```text
FSB = 3
HSB = 2.0
SSB ≈ 4.18569
RCB = 4
```

The running times may vary depending on the machine, the Gurobi version, and the available license.

## Partial matrices

Partial matrices are encoded using the following convention:

- positive entries are treated as known positive entries;
- zero entries are treated as known zeros;
- entries equal to `-1` are treated as unspecified entries.

For such matrices, the functions automatically detect the presence of unspecified entries and adapt the corresponding combinatorial formulations.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{BaeckelantVandaeleGillis2026,
  title   = {Computing Lower Bounds on the Nonnegative Rank via Non-Convex Optimization Solvers},
  author  = {Baeckelant, Timothy and Vandaele, Arnaud and Gillis, Nicolas},
journal={arXiv preprint arXiv:2605.14058},
 year    = {2026}
}
```

## License

Please see the `LICENSE` file for licensing information.
