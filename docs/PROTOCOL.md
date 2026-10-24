# Frozen protocol

## Datasets and canonical BDA

The UCI protocol uses Batch 1 as source and Batches 2–10 as targets, retaining all 128 features and all original observations. The primary targets with matching label support are Batches 2, 6, 7, 8, 9 and 10. Batches 3, 4 and 5 lack true class 6 and are reported separately. The external benchmark uses 248 sensor features: the first five retained dates provide 90 source observations; seven chronological target blocks contain 90, 89, 90, 90, 90, 90 and 71 observations. Class, date, concentration and environmental measurements are excluded from feature inputs.

The machine-readable configs freeze source-fitted StandardScaler, sample-wise L2 normalization before and after projection, primal representation, dimension 30, lambda 1, exactly 10 iterations, Euclidean 1NN and 11 mu values from 0.0 to 1.0. Initial pseudo labels come from source-trained 1NN in the normalized, unprojected representation before the first eigendecomposition. The marginal term includes the source class count. Missing target pseudo classes retain their source-only conditional contribution. MMD is normalized by its Frobenius norm before adding lambda I. The solver is scipy.linalg.eig; returned right eigenvectors retain Euclidean unit normalization without B-whitening. Iteration 10 is returned, with no early stopping.

## Phase A — freeze before evaluation

**Target truth is inaccessible to adaptation and selection.** The core interface is fit_predict(Xs, ys, Xt). UCI target loading skips label tokens. External loading reads only feature columns and the first 90 source labels. Positional domain metadata contains no target class labels. The scaler is fitted on source observations only.

All 99 UCI or 77 external candidates, source-only controls, selector decisions and uniform ensemble predictions must be completed and saved before prediction_freeze.json is written. The freeze hashes inputs, code, configs, predictions and decision/index/diagnostic metadata. Hashing input bytes does not parse target truth. No target metric, oracle or iteration selection enters this phase.

Source-only SVM uses RBF, C=1, gamma=scale and balanced class weights after source standardization. Historical UCI 1NN uses standardized features without sample L2 normalization and is a contextual control. External 1NN matches the normalized canonical bootstrap representation. Mu=0 is marginal-only adaptation.

## Label-free selectors and ensemble

For each target observation, distance margin is the distance to the nearest source observation of another class minus the distance to the nearest source observation of the predicted class, measured in the final normalized embedding. margin_mean and margin_median maximize the mean or median raw distance difference. Equal scores select the smaller mu.

Hard-onehot EnsV averages the 11 hard class votes. A class tie selects the smallest class ID. The candidate with the highest agreement with this role prediction is selected; a candidate-score tie selects the smaller mu. **This is an applicability baseline outside the original probabilistic EnsV setting. No probability/NLL theoretical guarantee is inherited.**

Uniform multi-mu voting directly returns the same equally weighted role prediction, with the same class tie rule. No learned, weighted or subset ensemble is introduced.

## Phase B — retrospective evaluation

The independent evaluator validates the complete freeze, required metadata, prediction references, input/code/config hashes and cohort counts before parsing any target truth. It then computes Accuracy, Balanced Accuracy, default sklearn Macro-F1, fixed-class Macro-F1 and target-present Macro-F1. Fixed classes are [1,2,3,4,5,6] for UCI and [1,2,3] for the external benchmark. Metrics never feed back into Phase A.

Released oracle, gap and collapse summaries are frozen retrospective artifacts. Level 1 renders these saved values without adaptation or retuning. The canonical pipeline has no random stage; numerical portability caveats and diagnostic tolerances are documented in ENVIRONMENT.md.
