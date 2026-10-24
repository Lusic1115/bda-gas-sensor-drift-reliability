# UCI data acquisition

Dataset: Gas Sensor Array Drift Dataset, UCI identifier 224. DOI: https://doi.org/10.24432/C5RP6W. Official record and download: https://archive.ics.uci.edu/dataset/224/gas%2Bsensor%2Barray%2Bdrift%2Bdataset. Download manually; no archive is bundled or fetched automatically.

Place the original LIBSVM-like files in data/uci/Dataset/, named batch1.dat through batch10.dat. Each row has an integer class token and indexed features 1–128. The concentration-extended dataset is a different input format. Expected file hashes are in manifests/uci_input_hashes.json. Input hashing does not parse target truth.

Batch 1 is the only source; Batches 2–10 are targets. Preserve all rows and feature order, including the 11 historical Batch 2 special-pattern observations. Fit StandardScaler on Batch 1 only. Canonical BDA then applies sample-wise L2 normalization. The target loader skips label tokens; target truth is parsed only in independent Phase B, after all 99 candidates and decisions are frozen.

The official page contains both a CC BY 4.0 license block and research-only/noncommercial wording in the description. This candidate does not resolve their legal priority and redistributes no UCI measurements. Review official terms and attribution requirements before use or redistribution.
