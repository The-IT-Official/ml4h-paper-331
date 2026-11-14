# Seed Sensitivity of Grad-CAM Localization Rankings in Chest X-ray Pneumonia Detection

Code for **"Seed Sensitivity of Grad-CAM Localization Rankings in Chest X-ray Pneumonia Detection"**, Machine Learning for Health (ML4H) 2026, Findings Track, Paper 331.

Elaine Wynette Wijaya\*, Nathan Peter\*, Ryan Varghese\*, Elizabeth Anne Krupinski

\* equal contribution

## Summary

Saliency benchmarks usually rank methods such as Grad-CAM from a single training run per method. We retrain one fixed DenseNet-121 pneumonia classifier on the RSNA Pneumonia Detection Challenge data 21 times, changing only the random seed, and score Grad-CAM against radiologist bounding boxes on the 601 box-bearing test patients using per-image AUPRC and the pointing-game hit rate.

| Quantity | Value |
|---|---|
| Test ROC-AUC across seeds | 0.858 +/- 0.003 |
| Grad-CAM (7x7) mean AUPRC, seed SD (95% CI) | 0.290, 0.0243 (0.0144 to 0.0309) |
| Grad-CAM per-seed range | 0.267 to 0.351 |
| Grad-CAM pointing-game hit rate | 0.369 +/- 0.054 |
| Vanilla gradient mean AUPRC, seed SD (95% CI) | 0.369, 0.0080 (0.0060 to 0.0093) |
| Vanilla gradient pointing-game hit rate | 0.599 +/- 0.024 |
| Grad-CAM from dense block 3 (14x14), mean, seed SD | 0.185, 0.0196 |
| Grad-CAM 7x7, nearest-neighbor upsampling, mean, seed SD | 0.2916, 0.0193 |
| Grad-CAM on best-validation checkpoints, mean, seed SD | 0.328, 0.033 |
| Vanilla gradient on best-validation checkpoints, mean, seed SD | 0.420, 0.0083 |
| Center-Gaussian prior / average-box prior / random | 0.281 / 0.450 / 0.117 |

Seed choice alone decides whether Grad-CAM beats the center prior (11 seeds above, 10 below), and 52% of the 210 seed pairs differ in mean AUPRC by more than 0.02.

## Repo layout

```
model/
  01_Data_Splits.ipynb                       Kaggle notebook: patient-level 70/20/10 splits, DICOM sharding
  02_Hyperparameter_Search.ipynb             Optuna hyperparameter search (25 trials, search seed 42)
  03_Training_and_Scoring_Pipeline.ipynb     determinism rehearsal, 21-seed training, unamended (pre-ReLU) scoring
notebooks/
  04_Figures_and_Tables.ipynb                Grad-CAM and vanilla gradient export, confirmatory scoring, baselines,
                                             best-vs-epoch-25 comparison, all paper figures and tables
  05_Gradcam_Paper_and_Rebuttal_Layers.ipynb Grad-CAM re-export and scoring at 7x7 (final) or 14x14 (dense block 3)
  06_B3_Upsampling_B4_Best_Checkpoint.ipynb  bilinear vs nearest upsampling, best-validation arm, reproduction gates
PREREGISTRATION.md                           pre-registered analysis plan and amendments log (paper Appendix H)
LICENSE
```

## Which notebook produces which result

Cell numbers count code and text cells from 1 at the top of each notebook. Section headings are given too, since cell numbers shift if cells are added.

| Paper result | Paper section | Notebook | Where | Output (under `xrv_pipeline_nb/`) |
|---|---|---|---|---|
| Patient-level splits | 3.1 | 01 | only cell | `train_split.csv`, `val_split.csv`, `test_split.csv` |
| Hyperparameters (Table 2) | 3.2 | 02 | sections 15 to 17 | `trial_*_hyperparams.json` |
| Bit-identical training rehearsal | 3.3, App. G | 03 | cell 23 | printed SHA-256 of both runs |
| 21 trained checkpoints | 3.2 | 03 | cell 24 (`train_one_run`, cell 22) | `frozen_xrv_seed{SEED}_{best,epoch25}.pth` |
| Unamended (no-ReLU) sensitivity run, mean 0.2853, SD 0.0242 | App. B | 03 | cells 29 and 30, "epoch 25 grad" | `gradcam_npy_25/`, `gradcam_metrics_25/` |
| Epoch-25 Grad-CAM export | 3.4 | 04 | cell 48, "part 2: Epoch 25" | `gradcam_npy_25_fixedrelu/` |
| Main Grad-CAM result, H1, H2, ROC-AUC, random and center baselines | 4.1 to 4.4, 4.6 | 04 | cell 50, "AUPRC analysis: CONFIRMATORY" | `gradcam_metrics_25_withGBP/_aggregate/` |
| Center-prior Wilcoxon tests, per-seed table | 4.4 | 04 | cell 18, "run discussions f1 - f4" | `paper_outputs/vs_centre_baseline.csv`, `paper_outputs/T1_per_seed.csv` |
| Average-box prior (0.450) | 4.4 | 04 | cell 20, "box baseline" | `paper_outputs/avg_box_baseline.json` |
| Figures 1 to 4 | 4 | 04 | cell 18 | `paper_outputs/F1_*.pdf` to `F4_*.pdf` |
| Figure 7 (Patients A and B) | 4.6, App. E | 04 | cell 22, "additional f5 test qualitative" | `images_output/worst_sd.pdf`, `median_sd.pdf` |
| Vanilla gradient export and determinism hash | 3.7, App. G | 04 | cell 35 (rehearsal), then cell 37 (full run) | `grad_vanilla_npy_25/` |
| Vanilla gradient scoring (mean 0.369, SD 0.0080) | 4.8 | 06 | `vg_final` condition | `paper_outputs/B3_B4/B3_B4_summary.json` |
| Best-validation Grad-CAM export and scoring | 4.7 | 04 | cells 42 and 44, "part 1: Best checkpoint" | `gradcam_npy_best_fixedrelu/`, `gradcam_metrics_best_withGBP/` |
| Best vs epoch-25 paired comparison | 4.7 | 04 | cells 54 and 58, "part 3: comparison" | `checkpoint_comparison_withGBP/` |
| Grad-CAM 7x7 re-export (matches paper to 1e-4) | 4.9 | 05 | `LAYER = "final"` | `gradcam_final_npy_25/`, `gradcam_metrics_25_final/aggregate.json` |
| Grad-CAM from dense block 3, 14x14 | 4.9 | 05 | `LAYER = "denseblock3"` | `gradcam_denseblock3_npy_25/`, `gradcam_metrics_25_denseblock3/aggregate.json` |
| Table 4 and Figure 5 (layer comparison) | 4.9 | 04 | cells 61 and 62, "B1" | `paper_outputs/T2_resolution.csv`, `F6_resolution.pdf` |
| Bilinear vs nearest upsampling | 4.9 | 06 | | `paper_outputs/B3_B4/B3_B4_summary.json` |
| Table 6 (both methods on both checkpoint arms) | App. D | 06 | | `paper_outputs/B3_B4/B3_B4_summary.json` |
| Number of seeds needed, Figure 6 | 4.10 | 04 | cell 64, "B2" | `paper_outputs/T3_seeds_needed.csv`, `F7_seeds.pdf` |

The figure file names in `paper_outputs/` do not follow the paper's figure numbers: `F1_per_seed_auprc.pdf` is paper Figure 2, `F2_spread_vs_gaps.pdf` is Figure 1, `F3_per_image_sd.pdf` is Figure 4, `F4_loc_vs_class.pdf` is Figure 3, `F6_resolution.pdf` is Figure 5, and `F7_seeds.pdf` is Figure 6.

## Setup

Model and training configuration (identical for every seed; only the seed changes):

| Setting | Value |
|---|---|
| Backbone | DenseNet-121, initialized from torchxrayvision `densenet121-res224-mimic_nb`, trained end-to-end (no frozen layers) |
| Input | 1 x 224 x 224 grayscale, scaled with `xrv.datasets.normalize(img, 255)` |
| Head | Dropout then Linear(1024, 2) on the global average of the post-ReLU final features |
| Optimizer | AdamW, lr 4.448862857672661e-4, weight decay 1.3560236266648827e-5 |
| Dropout | 0.3366734543720239 |
| Batch size | 128 |
| Schedule | Cosine annealing, T_max = 25, 25 epochs |
| Loss | 2-class cross-entropy with inverse-frequency class weights |
| Precision | Mixed precision (`torch.amp.autocast` with `GradScaler`) |
| Augmentation | Rotate 7 deg (p 0.5), shift-scale-rotate 0.05/0.05 (p 0.5), brightness/contrast 0.10/0.10 (p 0.5), resize 224 |
| Checkpoints saved | Best validation ROC-AUC (`_best.pth`) and final epoch (`_epoch25.pth`) |
| Hardware | One Colab account, one NVIDIA A100 |

Data splits: the 26,684 RSNA patients are split 70/20/10 at the patient level, stratified on `Target` with `random_state=42`, giving 18,678 / 5,340 / 2,666 images. Localization is scored only on the 601 positive test images that have boxes.

"frozen" in names such as `frozen_xrv_seed42_epoch25.pth` refers to the frozen (pre-registered) configuration, not to frozen backbone weights.

## How to reproduce

All notebooks except `01_Data_Splits.ipynb` run in Google Colab with a GPU and expect Google Drive mounted at `/content/drive`. Every path is set at the top of each notebook; change `/content/drive/MyDrive/AI_Research/...` to your own Drive folder. Each notebook installs its own packages with `pip` in its first cell.

1. **Get the data.** Accept the rules of the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) on Kaggle. You need `stage_2_train_images/` and `stage_2_train_labels.csv`. We do not redistribute the images.
2. **Make the splits.** Run `model/01_Data_Splits.ipynb` on Kaggle with the competition attached. It writes `train_split.csv`, `val_split.csv` and `test_split.csv`. The cell also uploads the DICOMs to a private Hugging Face dataset; set `token` and `REPO_ID`, or delete the upload part if you do not need it.
3. **Convert to .npy.** [ADD: code that converts the DICOMs to `.npy`, writes them to `images/shard_XXX/`, and zips them with the split CSVs and `stage_2_train_labels.csv` as `rsna-pneumonia-dataset-npy.zip`.] Put the zip on Drive. `npy_map.json` is built automatically the first time notebook 03 or 04 runs.
4. **(Optional) Hyperparameter search.** Run `model/02_Hyperparameter_Search.ipynb`. The paper uses the best trial's values, listed under Setup, so you can skip this step.
5. **Train.** In `model/03_Training_and_Scoring_Pipeline.ipynb`, run cells 1 to 22 (setup and `train_one_run`), then:
   1. Cell 23 trains seed 42 for 5 epochs (T_max = 5) twice and asserts the two checkpoints have the same SHA-256. Do not continue if it fails.
   2. Cell 24 trains all 21 seeds. Training resumes from the last saved epoch if Colab disconnects; re-run the cell.
   3. (Optional) Cells 25 to 34 score the maps under the original, unamended protocol (no ReLU before normalization). This reproduces the Appendix B sensitivity run. Cells 37 to 46 are an earlier rectified scoring pass with a 7-method ranking table; notebook 04 supersedes them and is the version used in the paper.
6. **Main analysis.** In `notebooks/04_Figures_and_Tables.ipynb`:
   1. Run cells 1 to 15 (setup). **Skip cell 16**, which overwrites the seed list with `[9999]` only.
   2. Run cells 23 to 32 (data pipeline, model, Grad-CAM hook, test-set composition).
   3. Run cells 47 to 52 (epoch-25 export, confirmatory scoring, threshold sweep). This gives the main 0.290 / 0.0243 result.
   4. Run cells 41 to 46 (best-validation arm), then cells 53 to 58 (paired comparison).
   5. Run cell 35 with `REHEARSAL = True` and check that the two seed-42 hashes match, then run cell 37 for all 21 seeds (vanilla gradient export). Cells 38 to 40 check that every seed folder has 601 maps.
   6. Run cells 17 to 22 for Figures 1 to 4, the average-box baseline, Figure 7, and `T1_per_seed.csv`.
7. **Layer ablation.** Run `notebooks/05_Gradcam_Paper_and_Rebuttal_Layers.ipynb` with `LAYER = "final"`. It must reproduce mean 0.290 and seed SD 0.0243. Then run it again with `LAYER = "denseblock3"`. It needs `grad_vanilla_npy_25/` and `paper_outputs/T1_per_seed.csv` from step 6.
8. **Upsampling and best-checkpoint ablations.** Run `notebooks/06_B3_Upsampling_B4_Best_Checkpoint.ipynb`. It needs `gradcam_final_npy_25/` from step 7 and checks four gates before reporting numbers (below). Its last cell is a diagnostic for the first gate.
9. **Table 4, Figures 5 and 6.** Back in notebook 04, run cells 59 to 65.

### Reproduction gates (notebook 06)

| Gate | Expected | Our run |
|---|---|---|
| `vg_regeneration_matches_disk`: regenerated vanilla gradient maps equal files on disk | max rel diff < 1e-3 | FAIL (0.21), see Known issues |
| `gc_final_bilinear_matches_T1_per_seed`: re-scored 7x7 Grad-CAM matches paper per-seed means | max abs diff < 1e-4 | PASS (8.2e-5) |
| `vg_final_matches_paper`: epoch-25 vanilla gradient aggregate | mean 0.369, SD 0.008 | PASS (0.3692, 0.0080) |
| `gc_best_matches_tracker`: best-checkpoint Grad-CAM aggregate | mean 0.328, SD 0.033 | PASS (0.3280, 0.0332) |

## Scoring protocol

For each (seed, image) pair:

1. Rectify the raw map (ReLU).
2. Max-normalize to [0, 1], with the max clamped at 1e-8.
3. Upsample bilinearly to 1024 x 1024.
4. Score with `sklearn.metrics.average_precision_score(mask.ravel(), map.ravel())`, where `mask` is the union of all ground-truth boxes for that patient. No threshold is applied anywhere.
5. Pointing game: hit if the maximum-intensity location falls inside any box.

Grad-CAM uses the post-ReLU final DenseNet features (7x7) and the pneumonia logit (index 1). Vanilla gradient is the absolute gradient of the pneumonia logit with respect to the 1 x 224 x 224 input, no smoothing.

Image-independent baselines, scored with the same protocol:

- **Random:** the expected score of a random pixel ranking, which equals the positive-pixel rate.
- **Center prior:** an isotropic Gaussian centered on the 1024 x 1024 grid with sigma = 1024 / 5 = 204.8 pixels.
- **Average box:** the mean of the box masks of the positive RSNA patients outside the test split.

## Determinism

Every training run and export sets `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` (`set_determinism`, cell 9 of notebook 03 and cell 11 of notebook 04). Data loader workers are seeded with `seed_worker` and a seeded `torch.Generator`. Library versions were identical across runs, images were served from the same pre-built .npy cache, and checkpoints load with `strict=True`. Determinism was verified only in our one Colab A100 environment; other GPUs or library versions may not give bit-identical results.

Vanilla gradient export check: seed 42 was exported twice and both outputs hashed to

```
c9ffc0eed01a47614314d56491274036e76445e9fecb7003ead7f8e592ce1bb7
```

and the seed 42 output of the full run matched this hash.

## Seeds

All 21 seeds were fixed in the pre-registration before any confirmatory run, and none were excluded:

```
42, 739, 2026, 7, 13, 21, 99, 123, 256, 314, 500, 777, 1000, 1234, 2024, 2025, 3407, 4096, 5555, 8080, 9999
```

## Pre-registration

The study was pre-registered before confirmatory training (v1, then v2 on 2026-08-31, v3 on 2026-09-01 to 2026-09-02). The full analysis plan and the dated amendments log are in [PREREGISTRATION.md](PREREGISTRATION.md), reproduced from Appendix H of the paper.

## Checkpoints

Trained checkpoints are not redistributed. Notebook 03 regenerates all 42 checkpoints (21 seeds x {best-validation, epoch 25}) from the seeds and configuration above.

## Known issues

- **Vanilla gradient regeneration gate fails at the pixel level.** Notebook 06 regenerates the epoch-25 vanilla gradient maps with its own code path, which does not call `torch.use_deterministic_algorithms(True)`. Its maps differ from the saved ones (exported deterministically in notebook 04, hash above) by a max relative pixel difference of 0.21. The maps still correlate at r = 0.998, per-image AUPRC changes by 0.0008 on average (diagnostic in the notebook's last cell), and the aggregate result matches the paper (gate `vg_final_matches_paper` passes).
- **Notebook 04, cell 16** redefines `HPARAMS2` with `"seeds": [9999]`. Running the notebook top to bottom therefore processes only seed 9999. Skip cell 16 to process all 21 seeds.
- **Notebook 04 cell order** does not match run order: the figure and baseline cells (17 to 22) read outputs written by the scoring cells further down. Follow the order in "How to reproduce".
- **Notebook 05** is committed with `LAYER = "denseblock3"`. Set it to `"final"` for the first run.

## Citation

```bibtex
@inproceedings{wijaya2026seed,
  title     = {Seed Sensitivity of Grad-CAM Localization Rankings in Chest X-ray Pneumonia Detection},
  author    = {Wijaya, Elaine Wynette and Peter, Nathan and Varghese, Ryan and Krupinski, Elizabeth Anne},
  booktitle = {Machine Learning for Health (ML4H), Findings Track},
  year      = {2026}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE).

The RSNA Pneumonia Detection Challenge data is subject to its own Kaggle competition terms and is not covered by this license. The torchxrayvision pretrained weights are subject to that project's license.
