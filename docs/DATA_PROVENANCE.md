# Data provenance and redistribution scope

## UCI

Official dataset 224: https://archive.ics.uci.edu/dataset/224/gas%2Bsensor%2Barray%2Bdrift%2Bdataset ; citation: Vergara, A.(2012), DOI https://doi.org/10.24432/C5RP6W . The original 128-feature/class-token format and frozen 10 file hashes are documented in data/uci/README.md and manifests/uci_input_hashes.json. The official page currently contains both CC BY 4.0 license information and a research-only/noncommercial statement; no raw redistribution is attempted here. Obtain files from the official source and review terms.

## External

Public raw dataset DOI https://doi.org/10.5281/zenodo.15681119 ; description https://doi.org/10.1038/s41597-025-05993-8 . The official archive has 700 raw recordings, 39 retained dates and a 624-row distributed feature file. 76 omitted distributed rows concentrate on nominal days 3, 27, 28, 29, 30, 31: source coverage 72/90 and late target T5 coverage 32/90. Reconstruction processed all 700 with unchanged supplied functions before external classification, not to improve classification performance. It was not imputation/interpolation or a performance-selected subset.

The 624 overlaps matched within atol 1e−9, rtol 0, median absolute difference 0. Serialized comparison maximum 2.1827872842550278 e−11 differs from the earlier in-memory 1.4551915228366852 e−11 because they are different audit stages. Their historical coverage cause remains unresolved. This candidate publishes no feature-value table/raw recordings/upstream implementation. The exact frozen output/notebook/function/schema hashes are in manifests/external_reconstruction_freeze.json.

The DOI-specific DataCite record (https://api.datacite.org/dois/10.5281/zenodo.15681119) declares CC BY 4.0 for the external dataset. This does not establish a separate MIT software license for the supplied notebook. Both raw recordings and the original notebook remain excluded and must be obtained from Zenodo. See DATA_ATTRIBUTION.md and ../THIRD_PARTY_NOTICES.md.

The derived 700-row feature table is not distributed in this release. Users can reconstruct it using the documented workflow. Provenance, the expected output hash and the ordered schema are provided. The record-level data grant permits attributed derivatives; this release deliberately retains link-and-reconstruct handling. Original notebook function bodies are not vendored.

Raw-to-feature coverage is 700 recordings to 700 reconstructed feature rows, with no target-label or performance-based sample inclusion/exclusion. The existing 624-row overlap checks above are frozen historical validations, not recomputed results. No interpolation, imputation or outcome-dependent recovery is introduced.

## Project outputs

Only allowlisted frozen metrics, selection/collapse summaries, paper tables, figure data and vector artwork are copied. Sources and transforms are traced in manifests/frozen_result_map.csv and asset_sources.json. No complete internal audit dump, prediction vectors, unrelated coursework or author submission declarations are included. MIT covers original source/documentation only. Project-generated summaries/artwork remain distinct from raw data and are outside that software grant; dataset attribution remains applicable.

Historical status flags inside copied frozen manifests are retained as provenance snapshots; they do not override the current release policy described here.
