# Gas-sensor drift adaptation reliability

Code and frozen-result package prepared for **How Reliable Is Balanced Distribution Adaptation for Gas Sensor Drift? A Strict Label-Free Evaluation Across Two Long-Term Benchmarks**.

Authors: Yexin Xie; Qiuyu Zhou. Affiliation: College of Photonic and Electronic Engineering, Fujian Normal University, Fuzhou 350117, China. Correspondence contact: 108122023045@student.fjnu.edu.cn.

Software version: 1.0.0. The repository is hosted at https://github.com/Lusic1115/bda-gas-sensor-drift-reliability. A permanent archive DOI will be added after archival.

## Scope and findings

The study evaluates established primal BDA with fixed balance candidates across two chronological gas-sensor feature benchmarks. Sensitivity and selector tradeoffs depend on the benchmark; the observed UCI pseudo-class collapse does not replicate in the external canonical runs. No new adaptation algorithm, universal selector guarantee or hardware result is claimed. Oracle scores are retrospective; deployable decisions use source labels and unlabeled target features only. See docs/CLAIM_BOUNDARIES.md.

## Structure

- `src/`: canonical mathematics, data loaders, selectors and the gated pipeline.
- `scripts/`: command-line entrypoints; `configs/`: frozen parameters and protocols.
- `data/`: download instructions, reconstruction schema and chronology metadata.
- `results/`: frozen summaries and paper tables; `figures/`: PDF and PNG artwork.
- `docs/`: protocol and provenance; `manifests/`: source mappings and checksums.

Raw archives, the derived 700-row feature values, prediction vectors, environment binaries and internal audit history are excluded.

## Installation

Use environment.yml or requirements.txt; see docs/ENVIRONMENT.md for actual pins and platform caveats. No macOS, Rosetta or x86 requirement is imposed. Existing-environment metadata is not substituted for missing historical environment records.

```
conda env create -f environment.yml
conda activate sensing-drift-release
```

## Level 1 — paper results from frozen summaries

This is the default. It validates summary hashes and writes tables/figures to a new directory without fitting a classifier or loading target truth.

```
python scripts/reproduce_paper.py --level summaries --output out/paper
python scripts/make_tables.py --output out/tables
python scripts/make_figures.py --figure 5 --output out/gap_figure
```

The repository retains research Figure 1/2/3/5/6/S1. Research Figure 5 is the reviewed 180 mm grayscale-friendly gap plot (submission Figure 4). All five frozen paper tables are included. Level 1 is lightweight relative to full adaptation; no standardized runtime benchmark is available.

## Dataset setup

Obtain UCI dataset 224 from https://doi.org/10.24432/C5RP6W and external raw recordings/notebook from https://doi.org/10.5281/zenodo.15681119 . Follow data/uci/README.md and data/external/README.md; review official terms. No automatic downloader bypasses license checks. The derived 700-row feature table is not distributed in this release. Users can reconstruct it using the documented workflow; provenance, expected hashes and schema are supplied. Raw datasets and the original external notebook must be obtained separately.

## Level 2 — canonical UCI experiments

Full runs require explicit opt-in and locally obtained inputs. This command stops after all 99 candidates/controls/decisions are frozen; it does not score target labels.

```
python scripts/reproduce_paper.py --level full --dataset uci --data data/uci/Dataset --output out/uci_phase_a
python scripts/evaluate_frozen.py --dataset uci --data data/uci/Dataset --output out/uci_phase_a
```

The second command is the separate retrospective boundary. The pipeline never chooses mu, iterations, preprocessing or parameters from these metrics. See docs/PROTOCOL.md. Source-only SVM and historical UCI 1NN are contextual controls, not isolated estimates of adaptation benefit.

## Level 3 — external reconstruction and canonical validation

```
python scripts/reconstruct_external_features.py --raw-dir data/external/Dataset --notebook data/external/Dataset/feature_extraction.ipynb --out-dir out/external_reconstruction
python scripts/run_canonical.py --dataset external --data out/external_reconstruction/reconstructed_features_700.csv --output out/external_phase_a
python scripts/evaluate_frozen.py --dataset external --data out/external_reconstruction/reconstructed_features_700.csv --output out/external_phase_a
```

The notebook must match the reviewed hashes; unchanged supplied feature definitions process all 700 files. The pipeline requires the frozen reconstructed CSV bytes, ordered 248 features and source/target block assignment. No failed records are silently omitted. See docs/EXTERNAL_RECONSTRUCTION.md. Do not mistake the derived representation for the official 624-row processed file.

## Runtime and determinism

Canonical experiments are CPU-feasible; actual runtime depends on hardware, numerical libraries and domain size. No precise minute estimate is invented. No random seed is required for the canonical BDA pipeline under the released implementation. Row order, exact-distance ties and cross-BLAS floating-point behavior can affect hard predictions. Synthetic tests have their own explicit fixture seeds and reduced dimensions; they do not change paper parameters.

## Tests and data availability

```
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python scripts/smoke_test.py --output out/synthetic_smoke
```

Complete grids and necessary paper summaries are bundled; prediction files can be regenerated only by an explicit full run. No full internal audit history is distributed. Source/output hashes and provenance are in manifests/. See docs/DATA_PROVENANCE.md.

## Citation

CITATION.cff describes this repository and its two paper authors. It records software version 1.0.0 and the repository URL. No archive DOI or Git tag is claimed. Cite the original datasets separately using [docs/DATA_ATTRIBUTION.md](docs/DATA_ATTRIBUTION.md). BDA and EnsV algorithm references are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); these citations do not assert that upstream source code is bundled.

## License

### Project source code and documentation

The [MIT License](LICENSE) covers **Original source code and original documentation contained in this repository, unless otherwise noted.** Copyright (c) 2026 Yexin Xie. This copyright statement follows project provenance; the repository citation records both paper authors. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and the complete [file-level scope inventory](manifests/license_scope.csv).

**Third-party datasets, external notebooks, fonts, and other third-party materials are not covered by the MIT License and remain subject to their respective original terms.**

### Third-party datasets

Raw UCI and external datasets are not included. Obtain them from the original DOI/source links above. Their licenses and attribution remain separate from MIT; the UCI page's current license field and historical restriction wording are both documented without resolving their priority. See [data provenance](docs/DATA_PROVENANCE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

### External feature-extraction notebook

The original feature_extraction.ipynb and its original or modified function blocks are not redistributed. Obtain the hash-pinned notebook from the original Zenodo record. The release provides its own wrapper and instructions; record-level dataset licensing is not a separate MIT software license for that notebook.

### Derived 700-row table

The reconstructed feature table is not distributed in this release, despite the data-license audit's attributed-redistribution finding. The workflow, expected output hash, ordered schema and provenance support local reconstruction without bundling the feature values.

### Generated research outputs

Frozen summaries, tables and figures are analytical outputs generated by this study. They do not relicense their underlying third-party data. These outputs are outside the MIT source/documentation grant; dataset attribution and provenance still apply. No separate output license is granted by the software license. DejaVu font material embedded in PDFs retains the notices in THIRD_PARTY_NOTICES.md; standalone font files are not distributed.

No Git tag, GitHub Release or permanent archive DOI has been created.
