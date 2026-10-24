# Environment and portability

environment.yml and requirements.txt pin the actual current project versions captured by standard-library package metadata. No dependency was upgraded or installed for candidate preparation. Core versions: Python 3.10.18, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, scikit-learn 1.7.2, Matplotlib 3.10.9. Python scientific imports occur only when running the corresponding commands. Configs are JSON-compatible YAML and use the standard-library JSON parser, so no YAML package is required.

The development/candidate-validation host is macOS 26.6.2, Darwin 25.6.0, x86_64; the project history includes a Mac/Rosetta environment. The release does not require macOS, Rosetta, an x86 CPU or a fixed local Conda installation path. There are no explicit architecture-specific binaries or OS-only imports. Compatible native platforms may use the same package versions; wheel/Conda availability is platform dependent.

```
conda env create -f environment.yml
conda activate sensing-drift-release
```

Alternatively, use an isolated Python 3.10 environment and install the exact pins:

```
python -m pip install -r requirements.txt
```

The completed smoke check used the existing pinned environment in a fresh temporary checkout, not a newly provisioned dependency environment. No cross-platform execution is claimed. Current OS/version metadata must not be fabricated into a historical inventory: the external experiment lacked its own complete package/OS record. Historical UCI/core and reconstruction versions were recorded; current Matplotlib/presentation versions do not retroactively identify the experiment's entire environment.

No random seed is required for the canonical BDA pipeline under the released implementation. Neighbor/eigenvalue ties depend on saved row order and numerical libraries. Cross-BLAS differences may affect distances, eigenvectors or hard 1NN decisions; exact output identity across architectures is not guaranteed. Numerical checks are eigenvalue imaginary magnitude≤1e−8×max(1, abs(real)), vector imaginary magnitude≤1e−8, and aggregate relative Frobenius eigen-residual≤1e−6. They are fail-stop diagnostics, not accuracy-tuned repair thresholds. Feature overlap verification used atol 1e−9, rtol 0. A reconstructed CSV byte-hash mismatch must be reviewed independently; full external Phase A does not silently accept a different benchmark representation.
