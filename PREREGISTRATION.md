# Pre-Registered Analysis Plan

**Seed Stability of Grad-CAM Localization Rankings in Chest X-ray Pneumonia Detection**

This is the pre-registration reproduced in Appendix H of "Seed Sensitivity of Grad-CAM Localization Rankings in Chest X-ray Pneumonia Detection" (ML4H 2026, Findings Track). References to sections, tables and appendices (for example Table 5, Appendix A, Appendix B, Appendix C) refer to the paper.

**Date written:** August 31, 2026.

**Status:** This document was written and timestamped in the shared project Drive prior to the completion of any of the 21 experimental training runs. All confirmatory analyses reported in the paper follow this plan. Analyses not specified here are labeled exploratory in the paper.

**Version 2 (Aug 31, 2026).** Supersedes v1. Change: frozen configuration re-specified after a pipeline-native hyperparameter search (torchxrayvision `mimic_nb` initialization; 25-trial Optuna on the confirmatory pipeline). Hypotheses, reference values, metrics, and analysis plan unchanged from v1.

**Version 3 (Sept 1–2, 2026).** All v2 text retained; amended passages are marked below with the superseded v2 wording followed by the dated replacement. Each amendment is listed in the log with its reason and its timing relative to the results it could affect.

## Amendments log

- **2026-09-01 · §5 · Map rectification added.** The scorer rectifies the raw 7×7 Grad-CAM map (ReLU) before max-normalization, per Grad-CAM Eq. 2 (Selvaraju et al., 2017) and consistent with Arun et al. (2021). Reason: v2 specified max-normalization "to [0,1]," which presupposes a non-negative map, but omitted the rectification step. Timing: amendment drafted ~4:40 PM CDT from the run notebook, before any epoch-25 score had been seen by the authors; an epoch-25 scoring run under the original text completed ~5:10 PM, before the amendment posted; the confirmatory arm was re-scored under the amended protocol and the unamended output is retained as a sensitivity result (Appendix B).
- **2026-09-01 · §4, §5 · Checkpoint [N] = 25.** Reason: the training code retained only the best-validation and final-epoch checkpoints per seed; intermediate epochs were not saved, so the fixed-epoch analysis uses the final epoch of the pre-specified 25-epoch cosine schedule. Per-seed validation AUC at epoch 25 was 0.846 ± 0.002, roughly 0.02 below each seed's peak (epochs 3 to 8, median 5). The best-checkpoint arm remains exploratory (§7) because the selected epoch varies by seed.
- **2026-09-01 · §3.2, §3.3 · Comparator table corrected.** Guided backpropagation (GBP, 0.17 ± 0.11) is stated in Arun et al. (2021) Table 3 (RSNA pneumonia); the v2 note claiming it was unstated was incorrect, and GBP was also omitted from the analysis code's hard-coded copy of the table. Restored: eight methods, seven adjacent gaps (0.11, 0.07, 0.02, 0.00, 0.15, 0.00, 0.01). No per-image scores affected. C1's gap count changes from 3 of 6 to 4 of 7. Published values are for InceptionV3; DenseNet-121 detection values were not retrievable from the published figure, so per §3.5 the InceptionV3 gaps serve as the reference and absolute levels are not compared, only spread against gaps.
- **2026-09-01 · H2 · Analysis note, registered text unchanged.** The two-flip threshold is unreachable under single-method substitution: a substituted score can displace at most one adjacent neighbor. H2 is reported as registered. A post-hoc analysis centring each seed's deviation on the published Grad-CAM value is reported in Appendix C and labeled post-hoc.
- **2026-09-02 · §5, §7 · M1 added.** Pre-registered secondary measurement of vanilla-gradient seed spread on the same 21 epoch-25 checkpoints, entered before any GRAD map was computed. See §5, M1. §7 amended to carve out M1 from the "additional saliency methods" exclusion.

## 1. Purpose

We measure whether saliency-method localization performance, as ranked by published single-training-run evaluations (Arun et al., 2021), is stable under retraining with different random seeds. We retrain one fixed pipeline 21 times, changing only the random seed, generate Grad-CAM maps from each resulting model, score them against radiologist ground truth using Arun et al. (2021)'s protocol, and compare the seed-induced variability to the published between-method score gaps.

An exploratory 5-seed pilot, conducted prior to this document on a different hyperparameter configuration, used the same initialization family (torchxrayvision `mimic_nb`) later adopted here. That adoption was made before any confirmatory run, for pipeline maturity and compute feasibility, and hyperparameters were re-searched from scratch on the confirmatory pipeline. No pilot result feeds any confirmatory analysis.

## 2. Hypotheses (confirmatory)

**H1 (Claim C1: variability vs. gaps):** The distribution of Grad-CAM localization utility scores across the 21 measurement seeds will have a spread (defined in §6.2) comparable to or exceeding the published score gaps between adjacently-ranked saliency methods in Arun et al. (2021)'s pneumonia results. Falsifiable form: if seed-induced spread is small relative to every adjacent gap, H1 is unsupported and we report that.

**H2 (Claim C2: ranking flips):** Seed-induced score variation is large enough that, if each seed's Grad-CAM score were substituted into Arun et al. (2021)'s published ranking, the adjacent-pair ordering would flip for at least 2 of 7 adjacent pairs in at least 3 of 21 seeds. Falsifiable form: fewer flips = H2 unsupported; we report the observed flip count either way.

## 3. Fixed reference values (from Arun et al. (2021))

**3.1 Scoring protocol used:** Area under the precision-recall curve (AUPRC), computed per image, threshold-free. Pixels inside the ground-truth bounding boxes are treated as positive labels and pixels outside as negative; each pixel of the (upsampled, normalized) saliency map is treated as a continuous output from a binary classifier, and AUPRC is computed over the resulting precision-recall curve. No binarization or thresholding of the map occurs anywhere in the protocol. Detection utility is scored on the box-bearing (positive) test images.

**3.2 Per-method pneumonia detection scores** (Arun et al. (2021), Table/Fig 2B; InceptionV3-generated maps, mean ± SD AUPRC). Amended 2026-09-01: GBP row added; ranks 6–8 renumbered. See Table 5 (Appendix A) for the full table. Baselines: AVG mask 0.47 ± 0.27; RetinaNet 0.59 ± 0.26.

v2 note: "GBP's detection AUPRC is not stated in their text; it appears only in their Fig 2B/supplement. C2 is computed over the 7 methods with stated values; if the GBP value is retrieved from the figure before analysis, it is added and noted." Amended 2026-09-01: GBP 0.17 ± 0.11 is stated in Arun et al. (2021) Table 3 (verified against the paper). Eight methods are used throughout.

**3.3 Adjacent gaps (computed from 3.2):** gap(XRAI→GCAM) = 0.11 · gap(GCAM→GRAD) = 0.07 · gap(GRAD→IG) = 0.02 · gap(IG→GGCAM) = 0.00 · gap(GGCAM→SG) = 0.15 · gap(SG→GBP) = 0.00 · gap(GBP→SIG) = 0.01. v2: "Three of the six adjacent gaps are ≤0.02." v3: Four of the seven adjacent gaps are ≤0.02; two of those are exact ties at reported precision.

**3.4 Their repeatability testing:** Arun et al. (2021) assessed repeatability by training two separately initialized models of the same architecture and comparing the resulting saliency maps via SSIM (n = 2). Grad-CAM passed their repeatability criterion on the pneumonia dataset. Whether localization utility scores, and the method rankings built on them, survive retraining was not tested and n = 2 cannot characterize the variance of any score. That untested quantity is precisely what this study measures.

**3.5 Architecture comparability caveat:** the reference scores above are from InceptionV3-generated maps (their stated values); our pipeline retrains DenseNet-121, an architecture they also evaluated. If DenseNet-specific detection AUPRC values are retrievable from their Fig 2B/supplement before analysis, those are used as the primary ruler with InceptionV3 as secondary; otherwise the InceptionV3 gaps serve as the reference with this caveat stated in the paper. (Outcome, 2026-09-01: not retrievable; InceptionV3 gaps are the reference.)

## 4. Experimental pipeline (frozen)

**Model/config:** DenseNet-121 initialized from torchxrayvision `densenet121-res224-mimic_nb`, trained end-to-end (no frozen layers).

**Hyperparameters:** lr 4.448862857672661 × 10⁻⁴, weight decay 1.3560236266648827 × 10⁻⁵, dropout 0.3366734543720239, batch size 128, AdamW, cosine annealing (T_max = 25). v2: "[N] epochs fixed for every seed." v3: 25 epochs fixed for every seed (see Amendments log, 2026-09-01). Loss: 2-class cross-entropy with inverse-frequency class weights. Only the seed differs between runs.

**Seeds:** [42, 739, 2026, 7, 13, 21, 99, 123, 256, 314, 500, 777, 1000, 1234, 2024, 2025, 3407, 4096, 5555, 8080, 9999]

**Hardware:** All 21 training runs and all Grad-CAM generation execute on a single Colab account (one author, NVIDIA A100 class), which must pass the bit-identical same-seed rehearsal (frozen configurations, seed 42, 5 epochs, run twice, byte-identical final weights) before any confirmatory run launches. No run or map generation occurs on any other account or machine. The hyperparameter search ran on the same account.

**Data:** RSNA Pneumonia Detection Challenge dataset, frozen train/val/test splits (files: `train_split.csv` / `val_split.csv` / `test_split.csv`). Test set: 601 images with radiologist bounding boxes as ground truth.

**Determinism:** deterministic algorithms enabled; identical library versions across all runs; images served from the pre-built .npy cache and resized identically to 224 × 224 in every run.

**Hyperparameters:** selected by a 25-trial Optuna search on this exact pipeline (DenseNet-121, `mimic_nb` initialization, search seed 42) completed before any confirmatory run; best trial validation ROC-AUC 0.8684. The optimum was flat: all completed trials fell within ±0.02 ROC-AUC.

**Seed inclusion:** All 21 seeds are analyzed. No seed is excluded for any reason, including anomalous scores; anomalies are the object of study.

## 5. Map generation and scoring (pinned choices)

**Grad-CAM:** computed on each run's final trained model (v2: "epoch [N]"; v3: epoch 25 final checkpoint, see Amendments log 2026-09-01), from the post-ReLU final DenseNet feature block, backpropagating the single pneumonia logit. Raw 7 × 7 float maps saved per (seed, image).

**Rectification (amended 2026-09-01):** the raw 7 × 7 map is rectified (ReLU; negative values set to zero) at scoring time, before max-normalization, per Grad-CAM Eq. 2 (Selvaraju et al., 2017) and consistent with Arun et al. (2021). Rectification is part of the map's definition, not a scoring threshold; the rectified continuous map is scored directly. The exploratory best-checkpoint arm is re-scored under the amended protocol so the paired comparison (§7) uses one scorer.

**Normalization:** per-image max-normalization of the rectified map to [0,1] (max clamped at 10⁻⁸ minimum).

**Upsampling:** bilinear, 7 × 7 → 1024 × 1024, applied at scoring time.

**Threshold:** none. The protocol is threshold-free (per §3.1): the continuous normalized map is scored directly.

**Primary score:** per-image AUPRC of the continuous normalized upsampled map against the binary box mask (pixels in any ground-truth box = positive), per Arun et al. (2021). Implemented as `sklearn average_precision_score(mask.flatten(), map.flatten())`.

**Secondary score (pre-registered):** hit rate (pointing game): the fraction of images whose maximum-intensity map location falls inside any ground-truth box.

**Negative (no-pneumonia) test images:** excluded from localization scoring; scoring is performed on the 601 box-bearing test images, consistent with Arun et al. (2021).

**Checkpoint [N] = 25 (amended 2026-09-01):** the final epoch of the pre-specified 25-epoch cosine schedule. The training code retained only the best-validation and final-epoch checkpoints per seed; intermediate epochs were not saved, so the fixed-epoch analysis uses the final epoch. Per-seed validation AUC at epoch 25 was 0.846 ± 0.002, roughly 0.02 below each seed's peak (epochs 3–8, median 5). The best-checkpoint arm is exploratory (§7) because the selected epoch varies by seed.

**M1 (pre-registered secondary measurement, added in v3.1, 2026-09-02, before any GRAD map was computed):** Vanilla gradient saliency (absolute input gradient of the pneumonia logit, 1×224×224, no smoothing) on the same 21 epoch-25 checkpoints and 601 images, scored under the identical §5 protocol. We report GRAD's seed spread (SD of per-seed mean AUPRC) with 95% bootstrap CI, and the number of seeds in which GRAD ≥ Grad-CAM.

- **Pre-specified interpretation.** CI overlapping Grad-CAM's [0.0144, 0.0309]: the cross-method transfer assumed in C1 is empirically supported for this pair. CI entirely above: seed variability is at least as large for a second method and C1's conclusion extends. CI entirely below: seed variability is method-specific and C1 is restricted to gaps involving methods of measured spread. Any seed with GRAD ≥ Grad-CAM is a directly observed seed-induced rank reversal between adjacent published methods.
- **Expectation, stated for the record:** GRAD spread ≥ Grad-CAM spread, on the basis of lower GRAD cross-model repeatability in Arun et al. (2021) (SSIM 0.30 vs 0.68, Table 4).
- **Motivation:** replaces the mis-specified H2 substitution test with a direct measurement on outcomes unobserved at registration.
- **Determinism check:** one seed's GRAD maps computed twice and hashed before the full run.

## 6. Confirmatory analyses (exact statistics)

**6.1 Per-image variability:** for each of the 601 test images, the standard deviation of its AUPRC across the 21 seeds, summarized by median and IQR across images.

**6.2 C1 test:** the seed-induced spread statistic = the standard deviation of per-seed mean AUPRC across the 21 seeds. Compared against each adjacent gap from §3.3. Reported with a bootstrap 95% CI (1,000 resamples over seeds).

**6.3 C2 test:** for each seed, substitute that seed's mean Grad-CAM score into the published ranking; count adjacent-pair inversions. Report the flip matrix (seed × pair), total flip rate, and bootstrap CI. H2 criterion: ≥2/7 pairs flipped in ≥3/21 seeds.

**6.4 Model-equivalence control:** test AUC of all 21 models, reported as mean ± SD, establishing that classification performance is stable while (or whether) maps move.

**6.5 Exclusion rules:** all-zero maps scored as zero-overlap and counted/reported. A crashed run is re-executed once with the same seed; if it fails again, the failure is documented and the seed excluded (reported).

Everything else (per-image instability correlates, stable/unstable exemplars, qualitative figures) is exploratory and labeled as such.

## 7. What this plan does NOT include

External validation on other datasets; additional saliency methods beyond the pre-registered M1 (vanilla gradient; see §5), including integrated gradients, occlusion, and XRAI (v2 listed these without the M1 carve-out; amended 2026-09-02); alternative preprocessing pipelines; intermediate-epoch checkpoint analysis; and augmentation-axis comparisons are explicitly out of scope for this study and are designated follow-up work.

---

Signed by all three student authors, 08/31/26. Written prior to completion of any experimental run. Amendments of 2026-09-01 and 2026-09-02 entered by one author; the other two notified in the team channel 1:20 AM 09/02/26.