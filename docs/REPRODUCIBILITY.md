# Reproducibility levels and boundaries

Level 1 reproduces five tables and six figures from the released frozen summaries, with integrity checks. It never imports adaptation code or evaluates target truth. Table CSV bytes are preserved. Figure numeric payloads are validated against frozen records; the collapse plot uses frozen counts and state/cohort sources without recomputing scientific statistics. Research figure IDs 1, 2, 3, 5, 6 and S1 are retained; submission preparation renumbered research 5 and 6 to submission 4 and 5.

```
python scripts/reproduce_paper.py --level summaries --output out/paper
python scripts/make_tables.py --output out/tables
python scripts/make_figures.py --figure 2 --output out/figure2
```

Level 2 explicitly runs the canonical grids using separately obtained inputs. Phase A freezes all predictions and decisions; Phase B validates the freeze before parsing target truth. Level 3 reconstructs external features using the user's hash-pinned original notebook. Neither runs by default. Output directories must be new.

The historical UCI source-only 1NN is a contextual control; SVM context is included in the paper tables. Complete grids retain unfavorable results and secondary UCI cohorts. Historical prediction_path fields identify intentionally omitted artifacts, not bundled resources. The prediction hash manifest preserves historical identities; explicit full reproduction writes new vectors, indexes and freezes.

Four core, normalization and classifier modules are copied unchanged. The old base class retains its reference-initialization method as a dependency; all public adaptation entrypoints use the paper-initialized canonical class. There is no public algorithm-version selector.

Synthetic tests run from the candidate root:

```
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python scripts/smoke_test.py --output out/synthetic_smoke
```

The smoke fixture uses 18 source and 9 target synthetic observations, dimension 3 and one iteration. These test parameters do not change the paper protocol. Candidate validation uses a fresh temporary copy and the existing pinned environment. No real adaptation or reconstruction was executed. Fresh dependency installation, cross-platform execution and bit-identical predictions across numerical libraries are not claimed.
