# SSB outputs and exact certificates

This directory collects the available numerical matrices $L$ used in the SSB computations reported in the paper. The files are grouped by matrix family:

- [correl.txt](correl.txt): slack submatrices of the correlation polytope;
- [ledms.txt](ledms.txt): linear Euclidean distance matrices;
- [ngons.txt](ngons.txt): slack matrices of regular polygons;
- [polyhedral.txt](polyhedral.txt): slack matrices of selected polytopes.

The matrices use the row and column ordering of the corresponding matrices in `SlackMat.mat`. Numerical outputs are subject to the solver tolerances described in the paper and the precision displayed in the files. In `polyhedral.txt`, the original numerical output is followed by exact optimal witnesses for the three matrices below. Their exact entrywise definitions are given before the rounded decimal displays.

## Exact certificates

For the following matrices in `SlackMat.mat`, an exact optimal matrix $L$ and a matching rank-one decomposition establish:

- `S24cell`: $\mathrm{SSB}=32/3$ ([proof](Certificate24cell.pdf), [verification](Certificate24cell.py));
- `Sdodecahedron`: $\mathrm{SSB}=42/5$ ([proof](CertificateDodecahedron.pdf), [verification](CertificateDodecahedron.py));
- `Scuboctahedron`: $\mathrm{SSB}=39/5$ ([proof](CertificateCuboctahedron.pdf), [verification](CertificateCuboctahedron.py)).

To verify the certificates, run the following commands from the repository root:

```bash
python "Logs SSB/Certificate24cell.py"
python "Logs SSB/CertificateDodecahedron.py"
python "Logs SSB/CertificateCuboctahedron.py"
```

The scripts perform the decisive checks in integer arithmetic and require no optimization solver. All three require NumPy; the 24-cell script also requires SciPy and reads `SlackMat.mat` from the repository root. The other two scripts are self-contained and require no external data file. Run Python without the `-O` option, since verification uses assertions.
