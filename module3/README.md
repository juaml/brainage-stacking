# Module 3 – Stacking Model Training

This module trains a region-wise stacking ensemble for brain age prediction using HTCondor parallelization.

## Method

- Level-0: One ElasticNetCV model per brain region (870 models)
- Level-1: ElasticNetCV meta-model combining all regional predictions
- Nested 5-fold cross-validation for unbiased performance estimation

## Input

Output from Module 2:
- `regional_voxels.pkl` — voxel data per region
- `regional_voxels_metadata.pkl` — subject IDs and ages

## Output

- `model.pkl` — final trained stacking model
- `scores.pkl` — cross-validation scores per fold
- `summary.csv` — MAE, R², RMSE, training time

## Usage
```bash
nohup python train_stacking_joblib.py \
  <features_dir> \
  <output_dir> \
  --n_regions 870 \
  --n_alphas 100 \
  --seed 42 \
  > training.log 2>&1 &
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `features_dir` | required | Directory with Module 2 output |
| `output_dir` | required | Directory to save results |
| `--n_regions` | 10 | Number of brain regions to use |
| `--n_alphas` | 100 | Alpha values for ElasticNetCV |
| `--seed` | 42 | Random seed |

## HTCondor

Jobs are parallelized on HTCondor cluster:
- Throttle level-0: 6 jobs (outer CV folds)
- Throttle level-1: 40 jobs (stacking inner folds)
- Shared data dir: `/data/group/appliedml/fkarateke_joblib_htcondor`
