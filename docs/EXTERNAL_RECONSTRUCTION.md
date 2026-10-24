# External reconstruction

The derived 700-row feature table is NOT INCLUDED in this release. The candidate contains this project's wrapper, ordered feature schema, positional chronology and hashes; it contains neither the raw recordings, upstream notebook code nor the derived 700-row values. Download the official version identified in data/external/README.md subject to its terms.

The production command checks the exact notebook bytes, three cell hashes and concatenated function-source hash before executing the supplied first three definition cells. It uses their original loader and FeatureExtractor without formula edits. A controlled per-file walk presents each existing raw file to the unchanged loader, matching the original reconstruction's order: nominal-day numeric order then filename. All 700 distinct files must be available; there is no label-/performance-based inclusion. An error/nonfinite feature/schema mismatch stops the operation rather than skipping or repairing a recording.

Four supplied feature families generate 254 processed columns: 248 gas-sensor values, four environmental features and Class/date metadata. The classification pipeline uses only the 248 ordered names in data/external/feature_schema.csv. The source block of 90 observations and target chronology follow data/external/domain_assignments.csv. Original chemical/concentration strings are literal metadata, not numerical extraction inputs. The public mapper recognizes only the frozen five conditions for the three-analyte task; it is not a concentration classifier.

```
python scripts/reconstruct_external_features.py --raw-dir data/external/Dataset --notebook data/external/Dataset/feature_extraction.ipynb --out-dir out/external_reconstruction
```

The output directory must be new and outside the raw dataset. Outputs include reconstructed_features_700.csv and reconstruction_manifest.json with row/schema/raw-file/definition hashes and actual output SHA256. No original file is overwritten. The wrapper records whether the output matches the frozen byte hash. Exact byte replay depends on numeric/CSV library versions; a false match flag is not silently called equivalent. The full external pipeline requires the exact frozen table bytes and stops on mismatch; request scientific/data-provenance review rather than weakening that gate to improve results. The historic overlap tolerance 1e−9, rtol 0 is a reported audit tolerance, not an automatic repair rule.

The supplied notebook is executable code. Hash pinning identifies the reviewed version; it is not a general sandbox for arbitrary notebooks. The original export cell is never executed. Production is fixed at 700 recordings; internal two-record synthetic fixtures exercise ordering/schema/hash/no-overwrite behavior without accessing the real dataset. The candidate preparation did not re-extract 700 files.
