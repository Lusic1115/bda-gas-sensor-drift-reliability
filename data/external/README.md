# External public recordings and local reconstruction

Official dataset: https://doi.org/10.5281/zenodo.15681119 (record https://zenodo.org/records/15681119 ), version v1. Dataset description paper: https://doi.org/10.1038/s41597-025-05993-8 . Obtain Dataset.zip from the official record, review its terms, and extract locally. The original archive is not included in this candidate. Its expected SHA256 is recorded in manifests/external_reconstruction_freeze.json.

Expected user-supplied structure:

```
data/external/Dataset/
  feature_extraction.ipynb
  extracted_features.csv
  Day 1/*.csv
  ...
  Day 40/*.csv
```

The archive has 39 retained acquisition dates (nominal Day 14 absent),700 recordings and a distributed 624-row processed file. Do not fabricate a blank day or samples. All 700 official recordings are processed in sorted nominal-day/filename order using unchanged supplied functions loaded from the user's notebook after notebook/cell/source hashes pass. Only the first three function-definition cells execute; the original export cell does not. Outputs must be outside the raw tree and must not already exist.

```
python scripts/reconstruct_external_features.py --raw-dir data/external/Dataset --notebook data/external/Dataset/feature_extraction.ipynb --out-dir out/external_reconstruction
```

This is Level 3, not a default smoke test. It must process 700/700 or fail; it does not impute, skip failed records or select observations by target labels/performance. Class/date strings are retained as literal metadata by upstream code, not numerical inputs to feature functions or an inclusion rule. See docs/EXTERNAL_RECONSTRUCTION.md for hash/schema gates and the distinction between byte replay and numerical equivalence.

The 700-row values and upstream notebook implementations are NOT INCLUDED. DataCite metadata declares CC BY 4.0 for the dataset; the notebook is not given a separate MIT software license by this project. This does not assert that redistribution is forbidden; it records a deliberately conservative candidate scope. The recipe and hashes allow licensed local reproduction without redistributing the inputs.
