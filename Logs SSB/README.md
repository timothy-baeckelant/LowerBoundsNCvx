# SSB numerical outputs and exact certificate

This directory collects the available numerical matrices $L$ used in the SSB computations reported in the paper. The files are grouped by matrix family:

- [correl.txt](correl.txt): slack submatrices of the correlation polytope;
- [ledms.txt](ledms.txt): linear Euclidean distance matrices;
- [ngons.txt](ngons.txt): slack matrices of regular polygons;
- [polyhedral.txt](polyhedral.txt): slack matrices of selected polytopes.

These are numerical outputs, subject to the solver tolerances described in the paper and the precision displayed in the files.

## Exact certificate for the 24-cell

For the matrix `S24cell` in `SlackMat.mat`, the [proof](Certificate24cell.pdf) gives an exact optimal matrix $L$ and a matching decomposition establishing $\mathrm{SSB}(X)=32/3$.

To verify the certificate, run the following command from the repository root:

```bash
python "Logs SSB/Certificate24cell.py"
```

The [script](Certificate24cell.py) requires NumPy and SciPy and performs the decisive checks in integer arithmetic. It reads the existing `SlackMat.mat` file from the repository root and requires no optimization solver.
