# How Reliable Is Balanced Distribution Adaptation for Gas Sensor Drift? A Strict Label-Free Evaluation Across Two Long-Term Benchmarks

**Article type:** Article

**Authors:** Yexin Xie; Qiuyu Zhou

**Affiliations:** College of Photonic and Electronic Engineering, Fujian Normal University, Fuzhou 350117, China

**Corresponding author:** Yexin Xie; email: 108122023045@student.fjnu.edu.cn

## Abstract

Gas-sensor adaptation studies such as gas-specific Balanced Distribution Adaptation (BDA) motivate a deployment question: how reliable is a balance choice made without target labels? We evaluate canonical BDA across two long-term feature benchmarks under a strict target-label-free protocol that freezes candidate predictions, selectors and voting before retrospective scoring. UCI same-label-support targets and an external chronological benchmark reconstructed with unchanged supplied extraction exhibit mean target-wise Target-present Macro-F1 ranges of 0.2565 and 0.0696 across the fixed eleven-value balance grid. Selector rankings and average-versus-worst-gap tradeoffs depend on the benchmark. Present-class pseudo-label collapse occurs in 33/66 UCI primary runs and 0/77 external runs in outputs 1–10, partially replicating balance sensitivity and selection tradeoffs, but not the UCI collapse pattern. Uniform multi-balance voting mitigates some extreme selections and remains a risk-mitigation baseline with benchmark-dependent tradeoffs. The external 700-row table is a derived reconstruction, not the distributed 624-row feature release. The contribution is a reproducible reliability evaluation that separates oracle tuning from deployable selection; it introduces no improved BDA algorithm.

**Keywords:** gas sensor drift; balanced distribution adaptation; unsupervised domain adaptation; label-free model selection; sensor arrays; electronic nose; reliability

## 1. Introduction

Gas-sensor drift is a long-term reliability problem: a classifier trained from an earlier labeled acquisition period may encounter a later feature distribution that differs from the one on which it was fitted. For electronic-nose classification, reporting performance on a random partition does not directly address that chronological transfer problem. A deployment-facing evaluation must instead examine what can be decided from an early source period and later unlabeled observations, while retaining the temporal structure and class support of those observations.

Unsupervised domain adaptation offers a way to use target features without requiring target labels during adaptation. That promise is conditional on the full deployment procedure, not merely the adaptation update. The preprocessing fit, representation, candidate schedule, model selector, and stopping rule all determine which predictions are available before labels are observed. A protocol can be label free during its adaptation loop yet still provide an optimistic impression if the reported setting is chosen retrospectively from target outcomes. This motivates separating the adaptation information boundary from the evaluation information boundary.

Balanced Distribution Adaptation provides a direct setting in which to study the distinction [1]. Its balance factor controls the mixture of marginal and pseudo-label-based conditional discrepancy. Gas-sensor applications already investigate BDA and balance sweeps [2], while related work provides dynamic alignment and unsupervised validation [3] [4]. The remaining empirical issue addressed here is not whether the balance factor can be varied, but whether a fixed or label-free rule provides reliable outcomes when the best balance is unavailable at decision time.

Published accuracy gains and hindsight-optimal factors do not by themselves resolve this model-selection question. A candidate may have a favorable mean while failing on a particular chronological target. An ensemble may moderate an extreme choice while providing little average advantage. A pseudo-label support failure prominent on one benchmark may fail to appear on another. Treating these possibilities as evaluation objects requires preserving the candidate predictions, freezing decisions before scoring, and reporting sensitivity and oracle gaps alongside conventional accuracy summaries.

For example, the frozen UCI Batch8 record favors different actions depending on the information boundary: a target-label oracle can choose its best grid member, while a mean-margin rule must commit from unlabeled geometry and direct voting commits to an aggregate. Comparing those actual commitments reveals a risk that an oracle-only result cannot expose. We therefore ask: in a realistic target-label-free setting, how reliable is canonical BDA for long-term gas-sensor drift when its balance factor must be fixed or selected without target labels? The study compares frozen temporal protocols on UCI and an external benchmark reconstructed from public recordings using unchanged supplied feature extraction. Its purpose is an empirical reliability assessment with explicit boundaries, rather than a new adaptation mechanism. In sensor-system evaluation, unavailable target annotations make oracle balance tuning an unattainable decision rule. We therefore evaluate sensitivity, label-free selection, worst observed oracle gaps, voting risk mitigation and pseudo-class collapse together, asking which findings persist under an external chronological protocol. Their benchmark-dependent strengths and failure modes define a boundary-aware reliability contribution. The contributions are:

- We document an auditable target-label-free protocol that freezes adaptation, candidate selection, and voting before target evaluation, separating deployable decisions from retrospective oracles.
- We characterize cross-benchmark balance sensitivity using domain-level Macro-F1 and Balanced Accuracy across the fixed grid.
- We compare deployable model selection and worst observed oracle gaps, using matched and contextual controls with explicit comparability boundaries.
- We diagnose present-class pseudo-label collapse on UCI while documenting its nonreplication in the external canonical runs, distinguishing actually absent target classes and bootstrap states from output-state collapse.

## 2. Related Work

Gas-sensor drift motivates transferring a classifier from an earlier labeled acquisition period to later periods without repeatedly obtaining target annotations. The verified gas-specific BDA study already investigates balanced marginal and conditional alignment for metal-oxide gas arrays and examines balance-factor effects [2]. The gas-specific study also uses particle-swarm optimization for BDA parameter tuning. Its optimization objective differs from evaluating a fixed candidate pool with frozen label-free selection; the available source does not establish enough detail about fitness-data access to characterize that tuning as deployable under the present information boundary. No target-label leakage accusation follows. Its existence rules out treating the use of BDA for gas drift, or a sweep of its balance factor, as a new algorithmic idea. The present question concerns what a frozen label-free choice can deliver and how reliability conclusions change across temporal benchmarks, rather than whether favorable target-labeled tuning can produce a useful result.

Original BDA introduces the balance between marginal and conditional distributions, while its weighted extension addresses class-prior imbalance [1]. Manifold Embedded Distribution Alignment provides dynamic distribution weighting using discrepancy estimates [3]. These methods motivate the adaptive and weighted stress tests retained in the supplement, but their original objectives are not equivalent to the current missing-pseudo-class audit. Deep Subdomain Adaptation Network uses soft class membership in classwise alignment [5]. Consequently, soft conditional weighting is established prior art; the supplementary soft mechanism experiment here is used to test a local reliability hypothesis, not to claim a new classwise adaptation principle.

Unsupervised validation is itself a developed research problem. Soft Neighborhood Density evaluates target structure without target labels and considers hyperparameter or iteration selection [4]. We did not implement the original SND probability outputs and neighborhood construction in the present nearest-neighbor pipelines. An arbitrary nearest-neighbor entropy proxy would not be a faithful reproduction. This is a nonimplementation boundary, not a claim that neighborhood-based validation cannot be constructed from nearest-neighbor features. The closest reliability positioning is ensemble-validation work that explicitly studies adverse model-selection outcomes and a certified baseline [6]. That overlap narrows the present contribution to an auditable sensor-drift application and bounded diagnosis. The hard one-hot applicability baseline is labeled throughout to prevent its deterministic implementation from being mistaken for the original full probability setting.

Recent gas-domain literature includes prototype-guided Transformer adaptation and domain-adversarial mixed-gas regression [7] [8]. The verified publisher records establish their subject matter but do not fully establish the target-label treatment of their model-selection procedures. The manuscript therefore does not infer that their reported tuning is strictly label free, accuse it of leakage, or claim classification superiority over methods not run here. The regression task also differs from the retained analyte-identity classification problem. These works provide context for the broader field while the numerical evaluation remains confined to frozen baselines and canonical candidates.

The distinction among algorithm design, model validation, and evaluation reliability determines the scope of this paper. Prior work already supplies adaptation mechanisms and validation concepts. Here, their presence motivates explicit control of when target outcomes become available, scrutiny of adverse selection outcomes, and support-aware comparisons across datasets. The verified literature matrix is targeted rather than systematic. The bibliography is verified against primary records; this targeted review does not establish exhaustive coverage or priority.

## 3. Materials and Methods

### 3.1 Problem definition and evaluation unit

Let a labeled source domain be \(\mathcal D_s=\{(x_i^s,y_i^s)\}_{i=1}^{n_s}\) and an unlabeled target domain be \(\mathcal D_t=\{x_j^t\}_{j=1}^{n_t}\). The task is analyte identity classification after a chronological shift in acquisition conditions. A target domain is a temporal batch or block, rather than a random held-out subset. Source labels define the candidate class set. Each source–target pair resets to the original fixed early source and is adapted separately using the same frozen parameter schedule. The complete unlabeled target batch must be available jointly before adaptation or selection; this is transductive batch adaptation, not online streaming. Separate tasks are not statistically independent replicates: temporal domains share a source and acquisition history. The study concerns the reliability of that schedule and of label-free selection among its outputs, rather than the discovery of a better classifier.

The main cross-dataset comparison uses same-label-support targets. On UCI, these are B2 and B6–B10. B3–B5 lack true class 6, contain a subset of source classes, and form a secondary analysis. Their outcomes remain available, but they are not silently incorporated into the primary mean. The external targets retain complete analyte support. Means are calculated over target domains within each analysis cohort; neither dataset size nor the number of domains is used to construct a pooled cross-dataset result. This preserves the distinction between replication across benchmarks and aggregation across observations.

### 3.2 Strict target-label-free execution and evaluation boundary

The deployable information set contains source features, source labels, target features, and diagnostics computed without target outcomes. Target features enter source-transformed representations and the transductive discrepancy calculation. Target labels do not enter standardizer fitting, adaptation, pseudo-label updates, candidate ranking, voting, hyperparameter adjustment, or early stopping. The iteration count and candidate grid are fixed in advance. Saved predictions and selector decisions are frozen before retrospective target scoring. The paper-writing stage reads frozen scalar metrics and diagnostic counts; it does not rerun adaptation, select new candidates, or reopen raw target labels.

Retrospective target labels support final metrics, the oracle reference, support-aware failure interpretation, and descriptive comparisons of already-frozen decisions. These uses do not make an oracle deployable. In particular, scoring a candidate pool after freezing is different from using its target scores to choose a model for deployment. The same distinction applies to reporting a domain's best balance factor: it identifies what happened under hindsight, not an available unlabeled decision rule. Figure 1 specifies this boundary and excludes any feedback from target evaluation to deployment decisions.

![Figure 1](figures/Figure1.png)

*Figure 1. Strict target-label-free computational workflow. Source-only scaling precedes transductive adaptation of the whole unlabeled target block. Candidate runs use the fixed 11-value μ grid and 10 outputs. Label-free selection or voting is frozen and hashed before retrospective target evaluation. Target labels enter only the terminal evaluation block; oracle selection and target-support attribution are retrospective and have no feedback path.*

### 3.3 Canonical primal BDA and implementation-specific normalization

Canonical BDA balances marginal and conditional discrepancy [1]. A StandardScaler is fitted on source observations only and applied unchanged to the target. Each standardized observation is then normalized to unit Euclidean norm. Write the resulting source and target columns as \(X\in\mathbb R^{d\times(n_s+n_t)}\). Initial target pseudo-labels are generated by Euclidean one-nearest-neighbor classification in this normalized, unprojected representation before the first eigendecomposition. Conditional discrepancy is therefore initialized from the first iteration whenever its weight is nonzero. This initialization order is part of the evaluated protocol and should not be conflated with an implementation that omits conditional alignment in its initial projection.

For the marginal discrepancy, define \(e_i=1/n_s\) on source columns and \(e_i=-1/n_t\) on target columns. The actual frozen core uses \(M_0=Cee^\top\), where \(C\) is the number of source classes. The associated projected marginal term is
\[
\operatorname{tr}(P^\top X M_0X^\top P)=C\left\|n_s^{-1}\sum_iP^\top x_i^s-n_t^{-1}\sum_jP^\top x_j^t\right\|_2^2.
\]
This class-count factor is retained because it appears in the executed implementation; dropping it would change the relative weighting before normalization.

For a source class \(c\), let \(n_s^c\) count its labeled source observations and \(\widehat n_t^c\) count its current target pseudo-labels. Define a conditional vector \(e_c\) with source entries \(1/n_s^c\) for class \(c\), target entries \(-1/\widehat n_t^c\) for pseudo-class \(c\) when that pseudo-class exists, and zeros elsewhere. If \(\widehat n_t^c=0\), all target entries remain zero while the source entries are retained. Thus the actual conditional matrix is
\[
M_c=e_ce_c^\top,\qquad M_{\mathrm{cond}}=\sum_{c\in\mathcal C_s}M_c.
\]
When both supports exist, its trace is the squared distance between projected source and pseudo-target class means. When a target pseudo-class is missing, it is a source-only factor; the implementation does not divide by zero, delete the source class, or substitute a true target class mean. This detail is essential to interpreting collapse diagnostics without implying that canonical BDA itself necessarily throws an exception.

At each iteration the mixed discrepancy and its normalization are
\[
M(\mu)=(1-\mu)M_0+\mu M_{\mathrm{cond}},\qquad \widetilde M=M/\|M\|_F.
\]
The primal projection follows the generalized objective
\[
\min_{P}\operatorname{tr}\!\left[P^\top(X\widetilde M X^\top+\lambda I)P\right]
\quad\text{subject to}\quad P^\top XHX^\top P=I,
\qquad H=I-\frac{\mathbf1\mathbf1^\top}{n_s+n_t}.
\]
This constrained objective motivates the eigensystem; it does not describe a constraint enforced on the returned projection. Define \(A=X\widetilde M X^\top+\lambda I\) and \(B=XHX^\top\). The executed generalized eigenproblem is \(AP=BP\Lambda\). The implementation uses a low-rank factor \(F\) satisfying \(FF^\top=M\); \(\|F^\top F\|_F=\|M\|_F\) supplies the same Frobenius normalization without constructing the full sample matrix. The executed solver is `scipy.linalg.eig(A,B)`, including singular B handling through finite-pair filtering. It takes the smallest finite real eigenvalues and the corresponding returned right eigenvector columns. SciPy returns Euclidean-unit columns; the code takes their real parts after its imaginary-component checks and does not B-whiten or otherwise rescale them. Thus \(P^\top BP=I\) is not enforced. Rescaling individual projection coordinates could change Euclidean distances even after sample normalization; the manuscript therefore preserves the actual solver convention rather than substituting an equivalent-looking constrained solution. It then applies sample-wise L2 normalization to the projected observations. Euclidean one-nearest-neighbor predictions update the pseudo-labels.

The fixed projection dimension is 30, regularization is \(\lambda=1\), and the iteration count is 10. The balance grid is \(\{0.0,0.1,\ldots,1.0\}\). A zero balance weight removes the conditional factor but retains marginal adaptation and normalization; it is not a source-only classifier. The midpoint assigns equal pre-normalization coefficients to the marginal and conditional matrices as implemented. No target outcome determines dimensionality, regularization, iteration count, candidate-grid resolution, or numerical recovery. The grid and dimension were fixed in the earlier UCI experiment; the external benchmark freeze retained them before external performance evaluation. They were not optimized here, and the study provides no parameter-robustness or optimality claim. Matrix Frobenius scaling occurs before adding the identity penalty; there is no trace normalization or hidden eigenvector whitening.

### 3.4 Label-free selectors and uniform voting

The main single-candidate selectors use the mean or median raw distance margin in the final embedding. For target observation \(z_j^t\) with candidate prediction \(\widehat y_j\), the raw-distance diagnostic is
\[
m_j=\min_{i:y_i^s\ne\widehat y_j}\|z_j^t-z_i^s\|_2-\min_{i:y_i^s=\widehat y_j}\|z_j^t-z_i^s\|_2.
\]
The margin is an unscaled difference of nearest-source distances. It is not a distance ratio, softmax confidence, calibrated probability, or distance between class centroids. `margin_mean` and `margin_median` maximize the corresponding aggregate across unlabeled target observations. Exact score ties use the smaller balance value. Their potential usefulness rests on a geometric confidence heuristic; neither score has access to correctness.

The EnsV-inspired hard one-hot applicability baseline is motivated by ensemble-validation model selection [6]. Each candidate contributes one hard class vote per target observation. A role prediction is obtained by uniform majority voting; a class-count tie uses the smallest class identifier. The single candidate with maximum agreement with this role model is selected, with the smaller balance value resolving exact candidate-score ties. This is an applicability baseline for hard one-nearest-neighbor outputs. It does not reproduce the full original probabilistic setting, and no probability-based or certified guarantee is transferred to this implementation. Its observed performance cannot establish a general failure or success of the original EnsV method.

Direct multi-balance voting uses the role predictions themselves as the final output, rather than selecting the candidate closest to them. The pool consists of all frozen balance candidates, with equal weights and the same deterministic class-tie convention. Candidate validation and direct voting therefore share a role model but have different outputs. Voting is included as a risk-mitigation baseline, not an algorithmic contribution or a promised default. Its potential protection against an isolated poor candidate must be assessed alongside average performance and failures shared by multiple candidates.

### 3.5 Metrics, oracle gaps, and support-aware failure definitions

Target-present Macro-F1 averages classwise F1 over classes actually represented in a target domain. Balanced Accuracy (BA) averages recall over that target-present support. UCI fixed-six Macro-F1 remains a consistency metric in the supplement; for the primary same-label-support cohort it coincides with Target-present Macro-F1. External fixed-three and Target-present Macro-F1 likewise coincide because all analytes are present. On target-support-subset UCI targets, those definitions need not coincide, and legacy default macro averaging over the union of true and predicted labels must not be relabeled as a guaranteed target-present score.

For each domain, the retrospective single-balance oracle is the maximum saved target-present F1 over the fixed grid. The descriptive oracle gap of a strategy is
\[
g_d=\max_{\mu\in\mathcal G}F1_{d,\mu}-F1_{d,\mathrm{strategy}}.
\]
A gap of at least 0.15 is called a catastrophic oracle gap, using the convention fixed in the earlier frozen selection analysis before this cross-benchmark synthesis. It is prespecified for the present reporting, rather than newly selected to improve its apparent result. This threshold is descriptive, not a theoretical boundary, safety certificate, or estimated deployment failure probability. Direct voting may exceed every single candidate and can therefore have a negative oracle gap. Such values must remain visible. The mean over balance candidates is another retrospective summary: it is not the score of a deployable model or an ensemble prediction.

Sensitivity is summarized by the range and population standard deviation of saved F1 over the grid for each domain. The primary summaries average these domain-level values separately by benchmark. Present-class pseudo-label collapse means that a class known retrospectively to be present in a target has no pseudo-labeled observations in an inspected state. The main definition covers output states after each projection; bootstrap pseudo-labels are analyzed separately. A run is counted once if any inspected state meets the condition. This supports an audit of class support, not a causal test that collapse produced the eventual F1 loss. Truly absent classes on target-support-subset domains do not count as present-class collapse.

### 3.6 Datasets, temporal construction, and source-only controls

Table 1 specifies the frozen cohort boundaries. UCI uses Batch1 as the source and later batches as targets [9]. The retained gas-array feature representation and source class vocabulary remain fixed. The secondary target-support-subset cohort is retained to disclose behavior outside the primary same-label-support setting; reporting all UCI targets together is supplementary and explicitly labeled. No random repartitioning or target-label stratification is introduced for this draft.

The external benchmark is a reconstructed derived feature table from public electronic-nose recordings, rather than a newly released official feature dataset. The public archive and supplied extraction code are identified by the frozen provenance record and official dataset sources [10] [11]. Unchanged author-supplied extraction functions were used to recover the full raw-recording coverage. Gas-sensor features are retained while temperature, humidity, and metadata are excluded from classification features. Concentration remains a recorded covariate; the prediction task is analyte identity rather than concentration estimation. The chronological source comprises the earliest retained days, and subsequent chronological blocks form separate target adaptation tasks. The last target block is shorter. Full counts and the mandatory distributed-versus-reconstructed distinction are given below and in the provenance supplement.

The external official five-condition vocabulary is mapped deterministically as follows: `EtOH` → Ethanol; `Diacetyl 0.1ppm` and `Diacetyl 1ppm` → Diacetyl; `Phenylethanol 200ppm` and `Phenylethanol 1000ppm` → 2-Phenylethanol. The classification target is analyte identity; concentration is metadata/covariate only and is never a prediction label or model input. This three-class task was selected for conceptual agreement with UCI gas-identity classification, not from classification performance. Ethanol denotes the reference-solution condition; the official description also identifies ethanol diluent in the other solutions. We do not interpret the task as pure-gas discrimination or assign an unsupported ethanol ppm value.

UCI uses a metal-oxide sensor array over 36 months, with six gases (ethanol, ethylene, ammonia, acetaldehyde, acetone and toluene) and 128 steady-state/transient response features [9] [2]. The external archive uses 62 metal-oxide sensors plus temperature/humidity channels over approximately one year [10]. Its gas-only features are exposure mean, relative baseline change, pre/post baseline ratio and initial exposure slope, in the supplied extraction order. Its 39 retained acquisition dates span 2024-02-16 to 2025-02-05. Retained-day indices are chronological ranks, not consecutive calendar days: nominal Day14 is absent in the supplied archive, and no blank day or synthetic samples are inserted. The source is retained dates 1–5; T1–T7 contain 6–10, 11–15, 16–20, 21–25, 26–30, 31–35 and 36–39. Date membership was frozen without target performance; acquisition gaps and concentration structure remain part of the benchmark rather than controlled causes of drift.

The UCI contextual source-only control is the frozen RBF support-vector machine trained with source-fitted standardization and balanced class weighting. Its saved primary-cohort fixed-six F1 is valid as target-present F1 because all source classes are present. Accuracy and BA are also retained. The frozen UCI standardized one-nearest-neighbor baseline is shown separately: it omits the sample normalization used by canonical initialization. Consequently, matching classifier names do not make it a representation-matched adaptation control. External controls include both RBF-SVM for continuity and a normalized source-only one-nearest-neighbor classifier that matches canonical initialization. Only the latter supports the reported within-external adaptation-gain comparison.

The public archive contains 700 raw recordings, but the distributed `extracted_features.csv` contains 624 observations, leaving 76 recordings without a distributed feature row. This is temporally selective missing coverage concentrated on nominal Days 3, 27, 28, 29, 30 and 31, not uniform chronological coverage: the distributed table covers only 72/90 source observations and 32/90 observations in late target block T5. Reconstruction was performed to restore complete chronological coverage of the publicly released raw dataset, not to improve classification performance. It re-extracted the originally defined sensor features from all publicly released raw recordings using the authors' supplied feature-extraction implementation without altering feature definitions. All 700 recordings were attempted, 700 succeeded and 0 failed; this was not imputation, interpolation, synthetic augmentation, manual repair or a performance-selected subset. All overlapping numerical feature values passed `atol=1e-09, rtol=0`; median absolute difference was 0, and post-serialization maximum difference was 2.1828e-11, limited to numerical precision. All 700 recordings were reconstructed before any classification experiment on this external benchmark; no observation was included or excluded based on classification accuracy, target labels, BDA performance or downstream model behavior. Domain construction remained chronological, predefined and independent of model performance. The resulting reconstructed feature table is a derived benchmark, not an official processed release. Detailed mapping, missing-day counts and overlap evidence are in the Supplementary Materials.

**Table 1. Dataset and protocol specification.** Primary cohorts remain separate from the UCI target-support-subset cohort. Target counts are chronological, not randomly sampled. External source: earliest five retained days; subsequent five-day target blocks, with a final four-day block.

| Cohort | Features | Class support | Source | Source N | Target domains | Target N |
| --- | --- | --- | --- | --- | --- | --- |
| UCI same-label-support (n=6) | 128 | 6 source; 6 target-present | Batch1 | 445 | B2, B6, B7, B8, B9, B10 | 1244, 2300, 3613, 294, 470, 3600 |
| UCI target-support-subset (n=3) | 128 | 6 source; 5 target-present | Batch1 | 445 | B3, B4, B5 | 1586, 161, 197 |
| External derived (n=7) | 248 | 3 source; 3 target-present | Retained dates 1–5 | 90 | T1-T7 | 90,89,90,90,90,90,71 |

### 3.7 Computational protocol and reproducibility

Both source-only RBF-SVM controls use kernel `rbf`, C=1, gamma=`scale` and class_weight=`balanced`, with source-fitted StandardScaler. The UCI native labels remain 1–6; the external fixed encoding is Ethanol=1, Diacetyl=2, 2-Phenylethanol=3. One-nearest-neighbor uses Euclidean distance (the installed default Minkowski metric with p=2) and uniform weights. The historical environment records Python 3.10.18, NumPy 2.2.6, SciPy 1.15.3 and scikit-learn 1.7.2; the writing environment does not define the experimental versions. The executable entrypoints, historical commands, exact feature order, hashes and external acquisition-date memberships are supplied in the [Supplementary Materials](Supplementary_Materials_v1.0.md).

All eleven candidates complete exactly ten iterations. There is no cross-validation, tuning, early stopping or stochastic initialization in the canonical pipeline. Its behavior is deterministic conditional on the saved data order, dependency versions and numerical environment; no random seed is needed for this pipeline. Exact-distance neighbor ties inherit scikit-learn's data-order behavior. Equal selector scores favor smaller μ, and equal vote counts favor the smallest encoded class. Equal numerical eigenvalues follow NumPy's recorded sorting/solver order; this is another reason to retain the environment and row order. Zero/nonfinite sample norms, inadequate finite eigenpairs, imaginary eigenvalue components above 1e−8 × max(1, |real eigenvalue|), eigenvector imaginary components above 1e−8, or relative eigen-residuals above 1e−6 stop the run; no epsilon normalization, dimension reduction or numerical repair is introduced.

The external execution record contains a process deviation: the explicit comparison against frozen UCI core hashes completed after 33 attempts had begun, although current-core hashes had been recorded before attempts. Both core hashes matched; no code or scientific protocol changed and no adaptation was retried. This timing distinction is retained in the appendix. Candidate predictions, diagnostic selectors and voting were frozen before retrospective target-label scoring. 

### 3.8. Generative-AI Assistance

ChatGPT and Codex (OpenAI) assisted research planning, code development and debugging, organization of diagnostic analyses, interpretation and checking of computational results, and manuscript drafting and editing. The authors manually verified experimental protocols, dataset definitions, numerical results, statistical summaries, scientific claims, references, and final manuscript content against code, frozen result files, and original sources. Historical model/version identifiers cannot be reliably reconstructed; no specific model version is asserted.

## 4. Results

### 4.1 Source-Only Performance and Canonical BDA

Table 2 summarizes the primary domain-level means and retains deployable fixed settings separately from hindsight references. On the UCI same-label-support cohort, the frozen RBF-SVM has mean target-present F1 0.3327, BA 0.4199, and Accuracy 0.4055. Canonical BDA at the marginal-only endpoint has mean F1 0.6023 and BA 0.6664; at the midpoint the corresponding means are 0.5608 and 0.6091. These are descriptive comparisons under distinct frozen pipelines. The standardized UCI one-nearest-neighbor reference is included for transparency, but its representation mismatch precludes treating its difference from BDA as an isolated adaptation effect.

Externally, the source-only SVM mean F1 is 0.5384, compared with 0.5213 for the method-matched source-only one-nearest-neighbor control. Canonical BDA yields mean F1 0.5661 at the marginal-only endpoint and 0.5574 at the midpoint. Across the frozen external domain-by-balance combinations, 45/77 F1 differences relative to the matched control are positive and 32/77 are negative. Adaptation benefits are therefore heterogeneous within the external experiment: the average of a selected fixed setting should not be read as improvement for every target or every balance value.

The single-balance oracle means are 0.6673 on UCI and 0.5936 externally. Their status is explicitly retrospective. They summarize the best observed candidate on each domain using target outcomes and cannot be offered as label-free deployment results. Likewise, the mean-over-grid rows describe the candidate pool and do not describe a voting predictor. This distinction matters when a broad grid contains both useful and detrimental candidates.

**Table 2. Source-only controls and canonical BDA performance.** F1 is Target-present Macro-F1 and BA is Balanced Accuracy; means are over the primary target domains within each benchmark. Oracle-single-balance and mean-over-grid rows are retrospective summaries. The UCI RBF-SVM is a contextual baseline; the UCI standardized one-nearest-neighbor representation omits sample L2 normalization. The external normalized one-nearest-neighbor baseline is method matched. No pooled mean or causal adaptation effect is inferred from unmatched baselines.

| Cohort | Method / reference | Mean F1 | Mean BA | Deployable |
| --- | --- | --- | --- | --- |
| UCI same-label-support (n=6) | Source-only RBF-SVM (context) | 0.3327 | 0.4199 | Yes |
| UCI same-label-support (n=6) | Source-only 1NN (standardized; unmatched) | 0.3016 | 0.3896 | Yes |
| UCI same-label-support (n=6) | Fixed μ=0.0 | 0.6023 | 0.6664 | Yes |
| UCI same-label-support (n=6) | Fixed μ=0.5 | 0.5608 | 0.6091 | Yes |
| UCI same-label-support (n=6) | Oracle single μ (retrospective) | 0.6673 | 0.7183 | No |
| UCI same-label-support (n=6) | Candidate-grid mean (retrospective) | 0.5674 | 0.6309 | No |
| External derived (n=7) | Source-only 1NN (matched) | 0.5213 | 0.5535 | Yes |
| External derived (n=7) | Source-only RBF-SVM (context) | 0.5384 | 0.5541 | Yes |
| External derived (n=7) | Fixed μ=0.0 | 0.5661 | 0.5668 | Yes |
| External derived (n=7) | Fixed μ=0.5 | 0.5574 | 0.5581 | Yes |
| External derived (n=7) | Oracle single μ (retrospective) | 0.5936 | 0.5963 | No |
| External derived (n=7) | Candidate-grid mean (retrospective) | 0.5568 | 0.5572 | No |

### 4.2 Sensitivity to the BDA Balance Factor

The mean target-wise F1 range over the frozen balance grid is 0.2565 for the UCI primary cohort and 0.0696 externally. The respective median ranges are 0.2729 and 0.0609. The contrast is also visible in the population-standard-deviation summaries in Table 3. Thus sensitivity is observed in both benchmarks, but its magnitude is substantially larger under the UCI protocol. This is descriptive partial replication, rather than evidence of equal sensitivity or stochastic instability.

Variation also differs across domains. UCI target-wise ranges extend from 0.0338 to 0.5943; external ranges extend from 0.0224 to 0.1654. Figures 2 and 3 show the domain curves alongside the separate benchmark summaries. A single pooled curve or a best-domain example would obscure this variation. Best and worst balance locations are hindsight descriptors of the fixed grid, not recommended deployment settings.

**Table 3. Balance sensitivity on primary target cohorts.** F1 is Target-present Macro-F1, BA is Balanced Accuracy, and SD is population standard deviation; scores and differences use absolute 0–1 units. Range and population SD are computed over each domain's frozen candidate scores, then summarized across domains. The secondary UCI cohort and all-target summary are supplementary. No distributional significance test is implied.

| Cohort | Domains | Mean F1 range | Median range | Minimum range | Maximum range | Mean F1 SD | Mean BA range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| External derived (n=7) | 7 | 0.0696 | 0.0609 | 0.0224 | 0.1654 | 0.0208 | 0.0710 |
| UCI same-label-support (n=6) | 6 | 0.2565 | 0.2729 | 0.0338 | 0.5943 | 0.0839 | 0.2335 |

![Figure 2](figures/Figure2.png)

*Figure 2. UCI primary same-label-support target batches B2 and B6–B10 (6 domains; 66 saved candidate runs). Target-present macro-F1 is plotted for each fixed μ value from 0 to 1 in increments of 0.1. Lines join saved metrics without fitting or aggregation. Both benchmark figures use identical 0–1 axes. These target-label-based curves and their maxima are retrospective evaluation, not deployable tuning.*

![Figure 3](figures/Figure3.png)

*Figure 3. External chronological domains T1–T7 (7 domains; 77 saved candidate runs) from the reconstructed 700-row derived benchmark; the distributed feature table contains 624 rows. Target-present macro-F1 is plotted for each fixed μ value from 0 to 1 in increments of 0.1. Lines join saved metrics without fitting or aggregation. Both benchmark figures use identical 0–1 axes. These target-label-based curves and their maxima are retrospective evaluation, not deployable tuning.*

The domain-level range and SD companion is provided as Supplementary Figure S1; complete candidate tables, selector records, support diagnostics and provenance are provided in Sections S1–S11 of the Supplementary Materials.

### 4.3 Reliability of Label-Free Balance Selection

Table 4 compares the fixed settings, margin aggregations, hard one-hot ensemble-validation applicability baseline, and direct voting. On UCI, `margin_mean` reaches mean F1 0.6054 and mean BA 0.6797. Its worst oracle gap is 0.2051, with 1/6 domains crossing the descriptive threshold. The frozen per-domain record identifies Batch8 as that extreme failure. A relatively strong average does not remove the need to inspect the tail of selection outcomes.

Externally, `margin_mean` has mean F1 0.5562 and worst gap 0.0837, with 0/7 threshold crossings. The UCI Batch8 pattern therefore does not recur for this selector under the external protocol. Conversely, the external `margin_median` worst gap is 0.1654, showing that changing a simple aggregation does not yield a uniformly better reliability profile. The data support a benchmark-dependent tradeoff, not a universal ranking of confidence heuristics.

The EnsV-inspired hard one-hot applicability baseline attains mean F1 0.5902 on UCI and 0.5619 externally. These outcomes concern the frozen hard-prediction implementation described in Methods. Agreement with a role model is label free, but agreement alone does not establish that the role model or its closest candidate is accurate. No evaluated deployable selector consistently dominates across both benchmarks when mean F1, worst gap, and threshold counts are considered together. This statement is restricted to the tested candidate pool and frozen strategies; it is not a claim that label-free selection is impossible.

**Table 4. Frozen deployable strategy reliability.** F1 is Target-present Macro-F1 and BA is Balanced Accuracy, in absolute 0–1 score units. Means are unweighted across target domains; worst F1 is the minimum domain F1 and worst gap is the maximum domain gap. Gap≥0.15 reports a domain count. All decisions precede target evaluation. Oracle gaps use the retrospective best single candidate and may be negative for direct voting. The gap threshold is descriptive. `EnsV-onehot` means the EnsV-inspired hard one-hot applicability baseline, not a complete reproduction of the original probabilistic setting. Complete per-domain records and top-rank tie definitions are supplementary.

| Cohort | Strategy | Mean F1 | Mean BA | Worst F1 | Mean gap | Worst gap | Gap≥0.15 | Domains |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UCI same-label-support (n=6) | Fixed μ=0.0 | 0.6023 | 0.6664 | 0.4829 | 0.0650 | 0.1528 | 1 | 6 |
| UCI same-label-support (n=6) | Fixed μ=0.5 | 0.5608 | 0.6091 | 0.2088 | 0.1065 | 0.2740 | 2 | 6 |
| UCI same-label-support (n=6) | Mean margin | 0.6054 | 0.6797 | 0.2778 | 0.0620 | 0.2051 | 1 | 6 |
| UCI same-label-support (n=6) | Median margin | 0.5953 | 0.6610 | 0.2778 | 0.0720 | 0.2051 | 1 | 6 |
| UCI same-label-support (n=6) | EnsV hard one-hot applicability | 0.5902 | 0.6403 | 0.3877 | 0.0772 | 0.1433 | 0 | 6 |
| UCI same-label-support (n=6) | Uniform multi-μ voting | 0.5940 | 0.6450 | 0.3831 | 0.0733 | 0.1278 | 0 | 6 |
| External derived (n=7) | Fixed μ=0.0 | 0.5661 | 0.5668 | 0.4008 | 0.0275 | 0.0851 | 0 | 7 |
| External derived (n=7) | Fixed μ=0.5 | 0.5574 | 0.5581 | 0.3429 | 0.0362 | 0.1429 | 0 | 7 |
| External derived (n=7) | Mean margin | 0.5562 | 0.5540 | 0.4124 | 0.0374 | 0.0837 | 0 | 7 |
| External derived (n=7) | Median margin | 0.5519 | 0.5517 | 0.3205 | 0.0417 | 0.1654 | 1 | 7 |
| External derived (n=7) | EnsV hard one-hot applicability | 0.5619 | 0.5628 | 0.3543 | 0.0317 | 0.1315 | 0 | 7 |
| External derived (n=7) | Uniform multi-μ voting | 0.5588 | 0.5581 | 0.3483 | 0.0348 | 0.1375 | 0 | 7 |

![Figure 4](figures/Figure4.png)

*Figure 4. Per-domain oracle gaps for six deployable strategies, shown separately for UCI primary same-label-support (6 domains) and external primary (7 domains). Gap equals the retrospective best single-μ Target-present Macro-F1 minus strategy macro-F1; zero is parity and negative values favor voting over the single-μ oracle. Distinct marker shapes and colors identify target domains; the same shape key remains readable in grayscale. Horizontal offsets separate points. The 0.15 line is a pre-fixed descriptive catastrophic-gap threshold, not a statistical test. Hard-onehot EnsV is an applicability baseline. No pooling across datasets or significance inference is used.*

### 4.4 Multi-Balance Voting as a Risk-Mitigation Baseline

Direct voting has primary UCI mean F1 0.5940 and worst oracle gap 0.1278. Its threshold count is 0/6. In the separate all-target UCI supplementary cohort, the corresponding count is 0/9. The primary and supplementary denominators are different analysis populations and must not be interchanged. The frozen Batch8 comparison shows that voting avoids the extreme margin-mean selection outcome there; this case motivates its role as a risk-mitigation baseline.

External voting has mean F1 0.5588, worst oracle gap 0.1375, and threshold count 0/7. Its mean does not exceed the fixed marginal-only setting in Table 2, and its worst gap is larger than that of external `margin_mean`. Voting therefore reduces some extreme model-selection failures without becoming consistently performance-optimal. Zero observed threshold crossings under these finite grids do not guarantee safety on future acquisitions, a different candidate pool, or a shifted label vocabulary.

### 4.5 Pseudo-Label Collapse as a Dataset-Dependent Failure Mode

Using the harmonized output-state definition, present-class collapse occurs in 33/66 UCI primary runs and 0/77 external runs (Table 5). Bootstrap pseudo-labels are a separate window: the corresponding counts are 55/66 and 0/77. Reporting the bootstrap-inclusive count as an output-only result would conflate initialization support with support after projection.

The supplementary all-target UCI output-only count is 53/99; the bootstrap-inclusive count is 78/99. True-absent class membership in B3–B5 is excluded from present-class collapse. These counts are canonical saved diagnostic summaries and are not replaced by failure totals from weighted or adaptive variants. Present-class collapse is frequent in the UCI primary analysis and absent from the external canonical runs inspected here. This is a dataset-dependent failure pattern, not evidence that collapse is inherent to BDA or impossible externally.

**Table 5. Comparable canonical present-class collapse.** A run is affected if an actually present target class is absent from pseudo-labels in any output state. Bootstrap states are separately disclosed in the text. Target-present support is retrospective diagnostic information and does not enter adaptation. Native estimator failures of prior-art stress tests have different definitions and are not merged into these denominators.

| Cohort | State window | Affected runs | Total runs | Affected states | Total states |
| --- | --- | --- | --- | --- | --- |
| UCI same-label-support (n=6) | Outputs 1–10 | 33 | 66 | 156 | 660 |
| External derived (n=7) | Outputs 1–10 | 0 | 77 | 0 | 770 |

![Figure 5](figures/Figure5.png)

*Figure 5. Present-class pseudo-label support collapse in comparable output states 1–10. Each left cell counts output states with at least one retrospectively true-present class missing from pseudo-label support (0–10 states) for a target domain and fixed μ. UCI primary has 33/66 affected runs and 156/660 affected states; external has 0/77 and 0/770. Separate right cells count affected bootstrap-state-0 runs across 11 μ values per domain (UCI 55/66; external 0/77). Output panels share a 0–10 color scale; bootstrap panels use a separate 0–11 scale because their unit is candidate runs rather than output states. Every cell prints its exact count, so color is redundant. Repeated states and μ runs are not independent trials. True-present support is retrospectively frozen metadata, not a fitting input; absent true classes are excluded. Zero observed external collapse does not establish its absence in other settings.*

### 4.6 Common Findings, Partial Replication, and Unreplicated Stress Tests

The common result is the existence of balance-dependent target performance and nontrivial label-free selection tradeoffs. The magnitude of sensitivity, selector rankings, and pseudo-label support behavior differ across the two feature benchmarks. The external experiment therefore partially replicates the balance-sensitivity and selection-tradeoff findings while narrowing the reach of UCI-specific conclusions. It supports neither an identical failure mechanism nor a general claim of out-of-benchmark robustness. Domain counts remain separate throughout this comparison.

UCI-only prior-art stress tests provide context rather than cross-dataset validation. The adaptive A-distance estimator encountered undefined states under the frozen UCI protocol, and weighted BDA also encountered missing-pseudo-class estimator failures. Initialization variants and censored iterations are preserved in the supplement. These native failures are not canonical collapse counts, and they do not establish that the underlying algorithms are generally ineffective. The soft conditional mechanism study is likewise supplementary: it removed undefined states in the tested mechanism but reduced primary average F1 relative to canonical BDA. This negative result constrains the suggested mechanism without creating a successful replacement method.

## 5. Discussion

### 5.1 Oracle Tuning versus Deployable Selection

The central evaluation distinction is between finding a favorable balance factor after observing target outcomes and selecting one before those outcomes are known. An oracle can choose a different candidate for every chronological target, including different choices among otherwise difficult domains. A deployable rule must instead use the information available at the decision boundary. Reporting only the oracle maximum would suppress uncertainty about which candidate should be used and could overstate the performance available in a realistic unlabeled deployment. It is useful as a hindsight reference, provided its role remains explicit.

This distinction also changes the interpretation of a method comparison. A large oracle improvement does not by itself show that a label-free selector can realize that improvement. Conversely, a selector's average score can conceal a large isolated gap. Mean performance, per-domain outcomes, and worst observed gap answer different questions and should be reported together. The present study does not estimate future risk probabilities from the temporal domains. It supplies an auditable account of how average and adverse selection outcomes differ under the frozen candidate grid.

### 5.2 Dataset-Dependent Reliability and the Scope of Replication

The two benchmarks agree that balance selection deserves explicit evaluation, but they differ in the strength and form of the observed difficulty. UCI exhibits larger target-wise variation and frequent present-class collapse. The external protocol exhibits weaker balance variation and no observed canonical present-class collapse. This contrast limits conclusions drawn from either benchmark alone. A failure pattern that is prominent under one temporal feature representation is not automatically an invariant property of the adaptation method.

The comparison does not isolate a physical explanation for the difference. Feature construction, class composition, concentration structure, chronological segmentation, and acquisition conditions vary together between the benchmarks. None is independently manipulated in the frozen evidence. Differences in sensitivity or pseudo-label support therefore cannot be attributed to a particular sensor mechanism, feature family, or dataset size. The defensible finding is heterogeneity under the two recorded protocols, with a clearly bounded absence of the UCI collapse pattern in the external runs.

The derived external table improves coverage transparency but introduces its own interpretive boundary. Reconstruction reproduces supplied feature definitions and checks overlapping observations; it does not turn the derived table into an official feature release or supply independent hardware validation. The provenance evidence makes the external comparison inspectable. It does not eliminate all possible dependencies between extraction, temporal coverage, and classification difficulty. These limits belong alongside the replication statement rather than being deferred to a generic caveat.

### 5.3 Model-Selection Uncertainty and Ensemble Applicability

The frozen strategies illustrate several distinct sources of uncertainty. Distance margins measure geometric separation relative to labeled source examples, but confidently separated target observations can still be assigned incorrectly under a domain shift. Mean and median aggregation emphasize different parts of the target margin distribution. Their disagreement across benchmarks shows that a simple aggregation choice can change the average-versus-tail tradeoff without resolving the underlying absence of target correctness information. This is consistent with the need to evaluate label-free validation itself, rather than assume that confidence is a reliable substitute for accuracy.

Ensemble agreement adds a different reference: a candidate is judged against the role model formed by the pool. This can reduce dependence on any one model's diagnostic, but it inherits the pool's shared errors. The hard one-hot applicability baseline also discards the probabilistic structure considered in the original reliability literature [6]. Its results are informative about the current candidate family and output representation. They cannot be used to accept or reject the original method's assumptions or transfer its certification arguments to deterministic nearest-neighbor predictions.

Direct voting avoids the second step of choosing one candidate, which explains its role in the evaluation design, not a proven mechanism for its observed successes. It may moderate an isolated bad selection, as in the frozen UCI example, while offering no protection when many candidates make the same error. The external comparison shows why a conditional risk-mitigation baseline should not become an unconditional recommendation. Its observed worst gap and mean are both necessary to judge whether the tradeoff is attractive within a specified deployment objective.

### 5.4 Practical Implications for Sensor-Drift Evaluation

An evaluation intended to inform deployment should declare the source–target chronology, preprocessing fit boundary, class support, and candidate schedule before presenting adaptation performance. It should distinguish fixed settings from label-free selection and both from retrospective target-label tuning. Publishing the candidate sensitivity profile is useful because a favorable fixed setting on one domain may coexist with large deterioration elsewhere in the same grid. Publishing the decision-freeze record makes it possible to assess whether target evaluation influenced the reported deployment choice. For periodic sensor recalibration, these records distinguish decisions available before new annotations from hindsight performance summaries; they do not demonstrate online or hardware deployment.

Matched controls are equally important. The external normalized one-nearest-neighbor control shares the canonical pre-projection representation, supporting a bounded adaptation-gain comparison. The UCI standardized one-nearest-neighbor control does not share sample normalization, and the RBF-SVM differs in classifier family. Both remain useful context, but neither difference isolates the effect of distribution alignment. Explicitly disclosing this mismatch is preferable to presenting a large numerical difference as an unexplained adaptation gain. Future controlled comparisons could address it, but no such experiment is introduced into this frozen draft.

Support audits should accompany outcome metrics. A classifier may omit a class from its pseudo-labels even while all source classes are present, and a target may genuinely omit a source class. These situations have different interpretations. Separating bootstrap from post-projection states and actually present from truly absent classes prevents initialization deficits, label-space mismatch, and output collapse from being counted as the same event. The practical implication is a reporting discipline with visible limitations, rather than a guarantee that a particular diagnostic will prevent failure.

### 5.5 Negative Results and Prior-Art Boundaries

Supplementary diagnostics did not establish reliable stability gating or support promoting a combined margin/agreement selector; neither was implemented as an improved selector. UCI-only A-distance and weighted-BDA stress tests encountered native estimator undefinedness under their frozen no-fallback rules, a different endpoint from canonical output collapse. The soft conditional mechanism removed its tested undefined construction but reduced primary average performance, and the saved soft-evidence associations did not consistently support the reliability hypothesis. These scoped negative findings constrain follow-up interpretations rather than establish general algorithm rankings; detailed cohorts, counts, correlations and paired comparisons remain in the Supplementary Materials.

## 6. Limitations

The empirical scope is limited to two temporally structured gas-sensor benchmarks. Both use pre-extracted feature classification, and only the canonical BDA family is evaluated across both. The evidence does not cover end-to-end raw-signal models, other sensor modalities, broad unsupervised adaptation families, or physical deployment. The temporal targets are related acquisition blocks rather than independent population samples. Domain-level means, extrema, and counts are descriptive; they do not establish population-level uncertainty or future failure rates.

The external representation is a reconstructed derived feature table, with overlapping numerical equivalence and unchanged supplied extraction documented in the frozen record. Although reconstruction was reproducible and numerically consistent with the distributed feature table on the overlapping observations, the historical reason why that table contained only 624 of the 700 raw recordings remains unresolved. The reconstruction is not an official full-coverage feature release. The source and some external targets are relatively small. Concentration and acquisition conditions can vary across chronological blocks, but the study does not identify their causal contribution to adaptation behavior. The retained analyte-identity task should not be generalized to concentration regression or mixed-gas inference.

The hard one-hot ensemble-validation applicability baseline differs from the original probabilistic setting. Its deterministic role model does not inherit the original guarantees. UCI target-support-subset targets require separate metric and collapse interpretation; all-target summaries cannot replace the primary same-label-support analysis. The UCI source-only one-nearest-neighbor representation differs from canonical initialization. Finally, finite candidate grids, fixed stopping, and the tested selectors leave many possible deployment strategies unexplored. An observed zero threshold count or absent collapse pattern cannot support a universal UDA claim.

## 7. Conclusions

Reliable evaluation of BDA for sensor drift requires separating oracle tuning from deployable target-label-free model selection. Under the frozen protocols, changing the marginal–conditional balance affects target performance in both benchmarks, with a larger sensitivity magnitude on UCI. The observed tradeoffs among fixed settings, margin diagnostics, ensemble-validation applicability, and direct voting do not yield a consistently dominant deployable strategy across the two benchmarks. A favorable average or oracle outcome alone is insufficient to describe that reliability profile.

Present-class pseudo-label collapse is frequent in the UCI primary analysis but does not appear in the inspected external canonical runs. This contrast makes collapse a dataset-dependent empirical finding within the present scope, rather than an inherent property of every BDA application. Direct multi-balance voting can reduce some extreme selection failures, yet its average and worst-gap tradeoffs remain benchmark dependent. The resulting contribution is an auditable reliability evaluation, with explicit preprocessing, temporal support, reconstruction provenance, and decision-freeze boundaries, rather than an improved adaptation algorithm.

## Supplementary Materials

Supplementary Materials accompany this manuscript: Sections S1–S11, numbered supplementary tables and complete CSV companions, and Figure S1. They contain full balance-grid results, source/control and class-support records, historical Batch2 inspection, stability/complementarity diagnostics, initialization-specific adaptive and weighted stress tests, negative soft-mechanism results, external reconstruction provenance, and detailed selector/collapse results. See [Supplementary Materials](Supplementary_Materials_v1.0.md) and its file index for exact Table S numbering.

## Author Contributions

Yexin Xie: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing—original draft, Writing—review and editing, Visualization, Project administration. Qiuyu Zhou: Validation.

## Funding

This research received no external funding.

## Institutional Review Board Statement

Not applicable.

## Informed Consent Statement

Not applicable.

## Data Availability Statement

The UCI Gas Sensor Array Drift Dataset is available from the UCI Machine Learning Repository (https://doi.org/10.24432/C5RP6W). The external long-term electronic-nose dataset, including the raw recordings and supplied feature-extraction notebook, is available from Zenodo (https://doi.org/10.5281/zenodo.15681119). This study generated no new raw sensor acquisition data. The study code, frozen result summaries, figures, and reproduction materials are publicly available at https://github.com/Lusic1115/bda-gas-sensor-drift-reliability; software release v1.0.0 is archived on Zenodo (https://doi.org/10.5281/zenodo.23610757) [12]. The reconstructed 700-row feature table is not directly redistributed in the code repository. It can be reconstructed locally from the original raw data using the public code and documented workflow. It is a study-derived reconstruction, not an official 700-row processed table released by the dataset authors; the authors' distributed processed table contains 624 rows. Original source code and documentation are provided under the MIT License; third-party data, notebook, and embedded font terms remain separate, and the software license does not automatically relicense generated research outputs.

## Acknowledgments

During the preparation of this study and manuscript, the authors used ChatGPT and Codex (OpenAI) for the research and writing assistance described in Section 3.8. Historical model/version identifiers cannot be reliably reconstructed. The authors reviewed and edited the outputs, performed the checks described in Section 3.8, and take full responsibility for the content of this publication.

## Conflicts of Interest

The authors declare no conflicts of interest.

## References

1. Wang, J.; Chen, Y.; Hao, S.; Feng, W.; Shen, Z. Balanced Distribution Adaptation for Transfer Learning. *2017 IEEE International Conference on Data Mining (ICDM)* **2017**, 1129–1134. https://doi.org/10.1109/icdm.2017.150.

2. Jiang, Z.; Xu, P.; Du, Y.; Yuan, F.; Song, K. Balanced Distribution Adaptation for Metal Oxide Semiconductor Gas Sensor Array Drift Compensation. *Sensors* **2021**, 21(10), 3403. https://doi.org/10.3390/s21103403.

3. Wang, J.; Feng, W.; Chen, Y.; Yu, H.; Huang, M.; Yu, P.S. Visual Domain Adaptation with Manifold Embedded Distribution Alignment. *Proceedings of the 26th ACM international conference on Multimedia* **2018**, 402–410. https://doi.org/10.1145/3240508.3240512.

4. Saito, K.; Kim, D.; Teterwak, P.; Sclaroff, S.; Darrell, T.; Saenko, K. Tune it the Right Way: Unsupervised Validation of Domain Adaptation via Soft Neighborhood Density. *2021 IEEE/CVF International Conference on Computer Vision (ICCV)* **2021**, 9164–9173. https://doi.org/10.1109/iccv48922.2021.00905.

5. Zhu, Y.; Zhuang, F.; Wang, J.; Ke, G.; Chen, J.; Bian, J.; Xiong, H.; He, Q. Deep Subdomain Adaptation Network for Image Classification. *IEEE Transactions on Neural Networks and Learning Systems* **2021**, 32(4), 1713–1722. https://doi.org/10.1109/tnnls.2020.2988928.

6. Hu, D.; Luo, M.; Liang, J.; Foo, C.-S. Towards Reliable Model Selection for Unsupervised Domain Adaptation: An Empirical Study and A Certified Baseline. *Advances in Neural Information Processing Systems* **2024**, 37, 135883–135903. https://doi.org/10.52202/079017-4316.

7. Sun, J.; Zheng, H.; Diao, W.; Sun, Z.; Qi, Z.; Wang, X. Prototype-Optimized unsupervised domain adaptation via dynamic Transformer encoder for sensor drift compensation in electronic nose systems. *Expert Systems with Applications* **2025**, 260, 125444. https://doi.org/10.1016/j.eswa.2024.125444.

8. Cui, X.; Huang, X.; Zhang, X.; Feng, P.; Wang, L.; Duan, S.; Peng, X. PTADA: An unsupervised domain-adversarial regression algorithm for sensor drift in mixed gas scenarios. *Sensors and Actuators B: Chemical* **2026**, 447, Part 2, 138855. https://doi.org/10.1016/j.snb.2025.138855.

9. Vergara, A. Gas Sensor Array Drift Dataset. *UCI Machine Learning Repository [Dataset]* **2012**. https://doi.org/10.24432/C5RP6W.

10. Wörner, J.; Eimler, J.; Pein-Hackelbusch, M. Long-term drift behavior in metal oxide gas sensor arrays: a one-year dataset from an electronic nose. *Scientific Data 12, 1628* **2025**. https://doi.org/10.1038/s41597-025-05993-8.

11. Wörner, J.; Eimler, J.; Pein-Hackelbusch, M. Long-Term Drift Behavior of Electronic Nose (version v1). *Zenodo [Dataset]* **2025**. https://doi.org/10.5281/zenodo.15681119.

12. Xie, Yexin; Zhou, Qiuyu. How Reliable Is Balanced Distribution Adaptation for Gas Sensor Drift? A Strict Label-Free Evaluation Across Two Long-Term Benchmarks. Software version v1.0.0. Zenodo. https://doi.org/10.5281/zenodo.23610757.
