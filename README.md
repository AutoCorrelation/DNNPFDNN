# DNNPFDNN

This repository implements and compares two localization pipelines:

- `DNN1-PF-DNN2`
- `DNN1-LS-DNN2`

`DNN2` uses the same input interface in both pipelines so results are directly comparable.

## End-to-End Data Flow

1. Raw data generation  
- Script: `DataGenerator.m` (or your prefixed variant such as `00_DataGenerator.m`)  
- Output: `ranging_data_cv.h5`

2. DNN1 training (denoising ranging measurements)  
- Script: `train_dnn1_single_noise_residual.m`  
- Output: `checkpoints/dnn1_residual_single_<noise>.mat`

3. First-stage state estimation (branch)
- PF branch: `filter_pf_cv.m`  
  - Output: `checkpoints/pf_results_cv.h5`
- LS branch: `filter_lls_cv.m`  
  - Output: `checkpoints/lls_dnn1_results.h5`

4. DNN2 dataset build (branch)
- PF dataset: `build_dnn2_dataset.m`  
  - Input: `checkpoints/pf_results_cv.h5`  
  - Output: `checkpoints/dnn2_dataset.h5`
- LS dataset: `build_dnn2_dataset_ls.m`  
  - Input: `checkpoints/lls_dnn1_results.h5`  
  - Output: `checkpoints/dnn2_dataset_ls.h5`

5. DNN2 training (branch)
- PF model training: `train_dnn2.m`  
  - Input: `checkpoints/dnn2_dataset.h5`  
  - Output: `checkpoints/dnn2_postprocess_<noise>.mat`
- LS model training: `train_dnn2_ls.m`  
  - Input: `checkpoints/dnn2_dataset_ls.h5`  
  - Output: `checkpoints/dnn2_postprocess_ls_<noise>.mat`

6. Evaluation and final comparison
- Numeric summary: `eval_dnn2.m`
- Final plot: `z_finalPlot.m`

## DNN2 Interface (Common to PF and LS)

- Input: `[state_x, state_y, r1, r2, r3, r4]` (6 channels)
  - PF pipeline: `state = PF estimate`
  - LS pipeline: `state = LS estimate`
- Target: `[x, y]`

## Recommended Run Order

1. `DataGenerator.m`
2. `train_dnn1_single_noise_residual.m`
3. `filter_pf_cv.m`
4. `filter_lls_cv.m`
5. `build_dnn2_dataset.m`
6. `build_dnn2_dataset_ls.m`
7. `train_dnn2.m`
8. `train_dnn2_ls.m`
9. `z_finalPlot.m`

## Notes
- Both branches should use the same split ratios, seed, and training options for fair comparison.

## DNN Model Properties

### DNN1 (Ranging Denoiser)
- Role: first-stage denoising of noisy ranging vectors
- Training script: `train_dnn1_single_noise_residual.m`
- Input: 4 features `[r1, r2, r3, r4]`
- Output: 4 features `[r1_clean, r2_clean, r3_clean, r4_clean]`
- Normalization: z-score for both input/output (train-set statistics)
- Split: train/val/test = `0.80 / 0.10 / 0.10`
- Optimizer: Adam
- Main training params:
  - `maxEpochs = 60`
  - `miniBatchSize = 512`
  - `initialLearnRate = 1e-3`
  - `seed = 42`
- Loss: MATLAB `regressionLayer` (MSE)
- Architecture (residual MLP, `y = x + f(x)`):
  - `featureInputLayer(4, Normalization="none")`
  - `fullyConnectedLayer(32) -> ReLU`
  - `fullyConnectedLayer(64) -> ReLU`
  - `fullyConnectedLayer(32) -> ReLU`
  - `fullyConnectedLayer(4)` as residual delta
  - `additionLayer(2)` skip connection from input
  - `regressionLayer`

### DNN2 (Postprocessor, Common Interface)
- Role: refine first-stage state estimate using denoised ranging
- Training scripts:
  - PF branch: `train_dnn2.m`
  - LS branch: `train_dnn2_ls.m`
- Input: 6 features `[state_x, state_y, r1, r2, r3, r4]`
  - PF branch: `state = PF estimate`
  - LS branch: `state = LS estimate`
- Output: 2 features `[x, y]`
- Normalization: z-score for input (train-set statistics)
- Split: train/val/test = `0.80 / 0.10 / 0.10`
- Optimizer: Adam
- Main training params:
  - `maxEpochs = 80`
  - `miniBatchSize = 512`
  - `initialLearnRate = 1e-3`
  - `seed = 42`
- Loss option (`lossType`):
  - `mse` (default; implemented via `regressionLayer`)
  - `mae` option exists, but current implementation still uses `regressionLayer` (MSE)
- Architecture (`dnn2_create_model.m`):
  - `featureInputLayer(6, Normalization="none")`
  - `fullyConnectedLayer(128) -> BatchNorm -> ReLU`
  - `fullyConnectedLayer(64) -> BatchNorm -> ReLU`
  - `fullyConnectedLayer(2)`
  - `regressionLayer`
