# [PAPER TITLE HERE]

Code for **"[PAPER TITLE HERE]"**, ML4H 2026 (Findings track), Paper 331.

Elaine [LAST NAME]\*, Nathan Peter\*, Ryan Varghese\*, [SENIOR AUTHOR NAME]
\* equal contribution

[arXiv link] | [OpenReview link]

## What this repo does

We train 21 seeds of the same frozen-backbone DenseNet-121 (torchxrayvision `densenet121-res224-mimic_nb`) on the RSNA Pneumonia Detection Challenge data, then measure how much saliency-map localization quality (AUPRC against radiologist bounding boxes, and pointing-game hit rate) changes across seeds. The 601 box-bearing test patients are scored for every seed.

## Repo layout

```
notebooks/
  01_train_21_seeds.ipynb                     training, determinism check, 21-seed run
  02_gradcam_export_and_layer_ablation.ipynb  Grad-CAM export + scoring (7x7 paper result, 14x14 rebuttal)
  03_upsampling_and_best_checkpoint.ipynb     B3 (bilinear vs nearest) and B4 (best-val checkpoint arm)
archive/
  early_3seed_ensemble_baseline.ipynb         early exploratory pipeline, NOT used for any paper number
results/                                      small CSV/JSON outputs behind the paper tables
requirements.txt
```

## Which notebook produces which result

| Paper result | Notebook | Setting | Output file |
|---|---|---|---|
| 21 trained checkpoints (`_best.pth`, `_epoch25.pth`) | 01 | cells 22 to 23 | Drive, see "Checkpoints" |
| Bit-identical determinism check | 01 | cell 22 | printed SHA-256 |
| Main Grad-CAM result, 7x7 (mean 0.290, seed SD 0.0243) | 02 | `LAYER = "final"` | `gradcam_metrics_25_final/aggregate.json` |
| Rebuttal: Grad-CAM from denseblock3, 14x14 | 02 | `LAYER = "denseblock3"` | `gradcam_metrics_25_denseblock3/aggregate.json` |
| B3: bilinear vs nearest upsampling | 03 | | `B3_B4/B3_B4_summary.json` |
| B4: best-validation checkpoint arm | 03 | | `B3_B4/B3_B4_summary.json` |
| Vanilla gradient result (mean 0.369, SD 0.008) | [MISSING, see below] | | `grad_vanilla_npy_25/` |

## How to reproduce

All notebooks run in Google Colab with a GPU. Data and checkpoints live on Google Drive; the notebooks read them from the paths set at the top of each one.

1. **Get the data.** Download `stage_2_train_images` and `stage_2_train_labels.csv` from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) on Kaggle (you must accept the competition rules). We do not redistribute the images.
2. **Convert to .npy and make splits.** [ADD: script/notebook that converts DICOM to .npy, shards into `images/shard_XXX/`, builds `npy_map.json`, and writes `train_split.csv` / `val_split.csv` / `test_split.csv`.] Our splits are in `results/splits/` so you can reuse the exact patients.
3. **Set paths.** At the top of each notebook, point `ROOT`, `ZIP_PATH`, `LABELS`, and `NPY_MAP` at your own Drive folders.
4. **Train.** Run notebook 01 top to bottom. It first runs a 5-epoch rehearsal twice and asserts the checkpoints are bit-identical, then trains all 21 seeds (resumable if Colab disconnects).
5. **Grad-CAM.** Run notebook 02 with `LAYER = "final"` first. It must reproduce mean 0.290, spread 0.0243. Then run it again with `LAYER = "denseblock3"`.
6. **B3 / B4.** Run notebook 03. It checks four gates before reporting numbers (see below).

### Built-in reproduction gates (notebook 03)

| Gate | Expected | Our run |
|---|---|---|
| Re-scored 7x7 Grad-CAM matches paper per-seed means | max abs diff < 1e-4 | PASS (8.2e-5) |
| Epoch-25 vanilla gradient aggregate | mean 0.369, SD 0.008 | PASS (0.3692, 0.0080) |
| Best-checkpoint Grad-CAM aggregate | mean 0.328, SD 0.033 | PASS (0.3280, 0.0332) |
| Regenerated vanilla gradient maps equal files on disk | max rel diff < 1e-3 | FAIL (0.21), see Known issues |

## Checkpoints

The 42 checkpoints (21 seeds x {best-val, epoch 25}) are at [ADD LINK: Hugging Face Hub or Zenodo]. File names follow `frozen_xrv_seed{SEED}_{best|epoch25}.pth`.

Seeds: 7, 13, 21, 42, 99, 123, 256, 314, 500, 739, 777, 1000, 1234, 2024, 2025, 2026, 3407, 4096, 5555, 8080, 9999.

## Known issues

- **Vanilla gradient regeneration gate fails at the pixel level.** Regenerating the epoch-25 vanilla gradient maps gives a max relative pixel difference of 0.21 against the saved maps, but the maps correlate at r = 0.998 and the per-image AUPRC changes by 0.0008 on average (diagnostic in the last cell of notebook 03). The aggregate result still matches the paper (gate 2 passes). The likely cause is non-deterministic GPU kernels in the backward pass to the input.
- **Notebook 01, last cell (Captum saliency export) does not run.** It references `saliency_npy_dirs` (typo), calls `gradcam_model_gc` instead of the saliency model, and saves `raw_cam`. It did not produce any paper result. [FIX OR REPLACE with the code that actually made `grad_vanilla_npy_25`.]
- **Notebook 01, Grad-CAM export cell** writes maps from the best-validation checkpoints into a flat `gradcam_npy/` folder. The paper's epoch-25 Grad-CAM maps come from notebook 02 with `LAYER = "final"` instead.

## Citation

```bibtex
@inproceedings{[KEY],
  title     = {[PAPER TITLE HERE]},
  author    = {[Elaine LAST] and Peter, Nathan and Varghese, Ryan and [SENIOR AUTHOR]},
  booktitle = {Machine Learning for Health (ML4H), Findings Track},
  year      = {2026}
}
```

## License

[ADD LICENSE, e.g. MIT. Confirm with the senior author first.]