# Kaggle Competitions

My solutions to [Kaggle](https://www.kaggle.com/) competitions: notebooks, submissions and the write-up behind each approach.

**Kaggle profile:** [kaggle.com/ranatalha71](https://www.kaggle.com/ranatalha71) · Kaggle **Expert** in Datasets and Discussions

Each competition lives in its own folder with the notebook (and, where useful, an HTML export with all outputs rendered) and the submission files I produced. Raw competition data is **not** included. Download it from the competition page.

---

## Competitions

| # | Competition | Task | Approach | Metric | Score / Rank | Folder |
|---|---|---|---|---|---|---|
| 1 | [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos) | Computer vision · multi-output regression (11-point grading curve) | Frozen DINOv2 (ViT-S/14 + ViT-B/14) embeddings on physically rescaled 56 mm tiles + logit-target ridge ensemble | Log-weighted EMD (lower is better) | Public LB EMD **46.5** · CV (cross-camera LOSO) 47.1 · final rank — | [`Soil-Grain-competition/`](Soil-Grain-competition/) |
<!-- TEMPLATE ROW: copy, fill in and remove the comment markers
| N | [Competition name](https://www.kaggle.com/competitions/<slug>) | Task type | One-line approach | Metric | Score / rank (or —) | [`Folder-Name/`](Folder-Name/) |
-->

> Scores are copied from the notebooks. "—" means the value is not known yet (e.g. the competition has not closed).

---

## 1. Predicting Soil Grain Size Distributions from Images

📁 [`Soil-Grain-competition/`](Soil-Grain-competition/) · 🔗 [Competition page](https://www.kaggle.com/competitions/soil-grain-size-from-photos)

### Problem
Predict the **cumulative grain-size distribution** (sieve/hydrometer curve) of a soil sample at the 11 DIN EN ISO 14688-1 support points (0.002 mm to 200 mm) **from photographs alone**. The metric is a log-weighted Earth Mover's Distance: the area between the true and predicted curves on a log₁₀(d) axis (range 0 to 500).

The dataset is tiny and shifted:
- **Very few labels:** 24 training samples (127 photos), 10 test samples (35 photos).
- **Camera shift:** training photos come from Samsung A52 / Motorola phones; **all test photos come from iPhone 14 / 16**.
- **Resolution mismatch:** training images are downscaled (~4.5 px/mm), test images are full resolution (14 to 19.5 px/mm).

### Approach
1. Map files to samples and attach the per-phone calibration (pixels per mm).
2. Resample every photo to a common **physical resolution of 4 px/mm** and mask out the dark sample tray.
3. Extract features per photo:
   - **DINOv2 embeddings** (final model): frozen ViT-S/14 and ViT-B/14 on 224 px tiles (= 56 mm of soil), 50 % overlap, flip/transpose test-time augmentation, averaged per photo.
   - **Granulometry features** (interpretable reference): morphological pattern spectra with discs of physical diameter 0.5 to 32 mm, band-pass energy, local contrast, colour.
4. Validate with **leave-one-sample-out (LOSO)** and **cross-camera LOSO** (train only on photos from other phones than the held-out sample).
5. **Final model:** equal-weight average of DINOv2-S and DINOv2-B ridge regressions fitted on **logit-transformed** cumulative percentages, averaged over all photos of a sample and post-processed into valid monotone curves.

### Key insight
**Calibrate to physical scale, and don't trust CV alone under domain shift.** Because test photos are 3 to 4× sharper per mm than training photos, a raw-pixel model sees the same grain as much larger, and hand-crafted texture statistics read fine-grained grey sand as "smooth" silt. Rescaling every image to the same px/mm fixed most of that. A cross-camera CV exposed models that memorise camera offsets (k-NN collapses), but even that could not detect that 112 mm tiles and hand-crafted features transfer badly to iPhone photos. Only a few public-leaderboard checks of *model families* revealed it. Regressing in logit space helped because the EMD is an L1 loss, which rewards median-like predictions.

### Results
Cross-validated and public-leaderboard EMD (lower is better), from the notebook:

| Submission | Tiles | Target | CV cross-camera | Public LB |
|---|---|---|---|---|
| Equal-share baseline | – | – | – | 125.6 |
| Training-mean curve | – | – | – | 102.4 |
| Granulometry ridge + DINOv2 blend | – | raw | – | 104.4 |
| DINOv2-S + B ensemble | 56 + 112 mm | raw | 50.4 | 62.0 |
| DINOv2-B + L ensemble | 112 mm | logit | 42.7 | 79.9 |
| DINOv2-S ridge | 56 mm | raw | 53.3 | 49.6 |
| **DINOv2-S + B (final)** | **56 mm** | **logit** | **47.1** | **46.5** |

Final model CV: **LOSO EMD 40.6**, **cross-camera LOSO EMD 47.1** (equal-share baseline ≈ 100). The public leaderboard covers only ~3 of the 10 test samples, so it was used to choose model families, never to tune weights. Final private-leaderboard rank: —

**Limitations:** clay and silt (< 0.063 mm) are far below the 0.25 mm pixel size and are inferred indirectly; dry fine-grained soils form aggregates that look like gravel in surface photos.

### Files
| File | Description |
|---|---|
| [`soil-grain-size-from-photos.ipynb`](Soil-Grain-competition/soil-grain-size-from-photos.ipynb) | Full notebook: EDA, preprocessing, feature extraction, CV, final model, submission |
| [`soil-grain-size-from-photos_with-outputs.html`](Soil-Grain-competition/soil-grain-size-from-photos_with-outputs.html) | Same notebook exported with all plots and tables (download and open in a browser) |
| `final_dinov2_ensemble.csv` | Submission from the final DINOv2-S + B logit-ridge ensemble |
| `submission_v2_logit_ensemble.csv`, `submission_v3_logit_56mm.csv` | Other submission variants |

### How to run
- **On Kaggle (recommended):** open the notebook in the competition, attach the competition dataset, enable a **GPU** and **Internet = On** (DINOv2 weights are downloaded via `timm` from the Hugging Face hub), then *Run All*. `submission.csv` is written to `/kaggle/working`.
- **Locally:** download the competition data, set `SOIL_DATA_DIR` to the folder containing `Training_labels_updated.csv`, and run the notebook:
  ```bash
  pip install numpy pandas opencv-python pillow matplotlib scikit-learn torch timm
  export SOIL_DATA_DIR=/path/to/soil-grain-size-from-photos   # PowerShell: $env:SOIL_DATA_DIR="..."
  jupyter notebook Soil-Grain-competition/soil-grain-size-from-photos.ipynb
  ```
  Without a GPU the notebook falls back to DINOv2-S only and no TTA, so results will differ from the final model.

---

<!--
================ TEMPLATE: copy this block for a new competition ================

## N. <Competition name>

📁 [`Folder-Name/`](Folder-Name/) · 🔗 [Competition page](https://www.kaggle.com/competitions/<slug>)

### Problem
<What is predicted, from what data, and the evaluation metric.>

### Approach
1. <Preprocessing>
2. <Features / model>
3. <Validation scheme>
4. <Final model / ensemble>

### Key insight
<The one thing that made the difference.>

### Results
| Submission | CV | Public LB | Private LB |
|---|---|---|---|
| <model> | — | — | — |

Final rank: — (e.g. 123 / 4,567, top 3 %)

### Files
| File | Description |
|---|---|
| `<notebook>.ipynb` | <description> |

### How to run
<Kaggle settings (GPU / Internet), local data path, dependencies.>

---
==================================================================================
-->

## How to add a new competition

1. Create a folder named after the competition, e.g. `My-Competition/`, with the notebook(s) and small submission files. Do **not** add raw data, model weights or `kaggle.json` (the [`.gitignore`](.gitignore) blocks the common cases).
2. Add a row to the [Competitions](#competitions) table by copying the commented **TEMPLATE ROW** below the table.
3. Add a section by copying the commented **TEMPLATE** block above this heading (view the raw `README.md` to see it).
4. Only report scores and ranks you can see in your notebook or on Kaggle. Use "—" until they are known (e.g. update the final rank after the competition closes).
5. Add the folder to [Repository structure](#repository-structure).

## Repository structure

```
Kaggle-Competitions/
├── README.md
├── .gitignore
└── Soil-Grain-competition/
    ├── soil-grain-size-from-photos.ipynb               # main notebook
    ├── soil-grain-size-from-photos_with-outputs.html   # rendered notebook with outputs
    ├── final_dinov2_ensemble.csv                        # final submission
    ├── submission_v2_logit_ensemble.csv
    └── submission_v3_logit_56mm.csv
```

## Tech stack

- **Language:** Python 3, Jupyter / Kaggle Notebooks
- **Data & ML:** NumPy, pandas, scikit-learn (Ridge, PCA, pipelines)
- **Computer vision:** OpenCV, Pillow, PyTorch, timm (DINOv2 ViT-S/14, ViT-B/14)
- **Visualisation:** Matplotlib

## Contact

**Talha Aslam**, MS Data Science, FAST-NUCES

- GitHub: [@TalhaAslam44](https://github.com/TalhaAslam44)
- Kaggle: [ranatalha71](https://www.kaggle.com/ranatalha71)
