# Blood Glucose Monitoring From PPG: Reproducible Benchmark Pipeline

## Abstract
This repository provides a reproducible machine-learning benchmark for blood glucose estimation from photoplethysmography (PPG). The framework standardizes heterogeneous datasets, applies literature-inspired signal-processing pipelines, constructs model-ready representations, and evaluates multiple regression models under unified experimental protocols. The implementation supports end-to-end execution from raw data extraction to artifacted result summaries.

## 1. Study Objective
The project evaluates how different preprocessing and feature-construction strategies influence glucose-regression performance from PPG signals. It is designed to support systematic comparisons across:

- Datasets (VitalDB, MUST)
- Processing pipelines (Zeynali et al. [1], Gupta et al. [2], Islam et al. [3], Alghlayini et al. [4], Prabha et al. [5])
- Model families (XGBoost, PLS, TinyCNN, ResNet34)
- Evaluation modes (leave some participant out cross-validation, multi-session cross-validation, random holdout)

## 2. Repository Scope
The pipeline is organized as a modular workflow:

1. Data extraction and harmonization into standardized HDF5 artifacts
2. Signal processing and quality filtering
3. Window/segment and feature construction
4. Model training/evaluation per benchmark plan
5. Unified metric reporting and artifact export

Primary orchestrator:

- `bench.py` (entry point)
- `src/benchmark.py` (task normalization, execution, and aggregation)
- `constants.py` (benchmark configuration, component registry, experiment plans)

## 3. Experimental Design
### 3.1 Processing and Feature Pipelines
Implemented modules include:

- Processors: `zeynali_proc`, `sen_gupta_proc`, `islam_proc`, `alghlayini_proc`, `prabha_proc`
- Builders: `zeynali_builder`, `sen_gupta_builder`, `islam_builder`, `alghlayini_builder`, `prabha_builder`

### 3.2 Evaluation Metrics
The evaluator (`src/evaluator.py`) computes:

- RMSE
- MAE
- MARD
- R²
- Clarke Error Grid zone proportions (A-E)

These are reported at fold/case level and summarized with mean/std.

## 4. Dataset Prerequisites
MUST extractor expects the following structure:

- `data/MUST/Labels/*.csv`
- `data/MUST/RawData/*.csv`
MUST dataset has to downloaded from [7].

VitalDB [6] extraction downloads metadata/signals through the `vitaldb` client/API at runtime.

## 5. Running the pipeline
Execute the full pipeline plan defined in `constants.py`:

```bash
python bench.py
```

The script loads `PIPELINE` from `constants.py`, executes all plan tasks, and writes consolidated results.

## 6. Output Artifacts
Default artifact root:

- `artifacts/bench/`

Typical outputs include:

- Stage caches (`raw/`, `processed/`, `built/`)
- Model run directories (`runs/`)
- Per-run serialized result files (`results.pkl` and success markers)
- Global summary table (`artifacts/bench/results.csv`)
- Execution log (`artifacts/bench/benchmark_run.log`)

## 7. Repository Structure
- `bench.py`: benchmark entry script
- `constants.py`: datasets/processors/builders/models and experiment plan
- `src/benchmark.py`: execution engine and artifact manager
- `src/extractors/`: dataset-specific extractors (VitalDB, MUST, custom)
- `src/processors/`: signal preprocessing pipelines
- `src/builders/`: segmentation/feature builders
- `src/models/`: model cores, trainers, tuners, runners, dataset utilities
- `src/evaluator.py`: metric computation and report serialization
- `src/ceg.py`: Clarke Error Grid computations

## 8. Methodological Notes
- The benchmark uses hash-based artifact versioning for cache consistency across configuration changes.
- Task modes include `train_cv`, `train_cv_multi_session`, and `train_cv_random`.
- Random seeds and model hyperparameters are centrally defined in `constants.py`.

## 9. Citation
If this repository is used in academic work, please cite:

```
@misc{ramesh2026reassessing,
      title={Reassessing the Feasibility of PPG-Based Non-Invasive Blood Glucose Level Estimation}, 
      author={Ramesh,Supraja and Neufeld, Markus  and K\"uttner, Michael and R\"oddiger, Tobias and Beigl, Michael},
      year={2026},
      eprint={2608.01820},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2608.01820}, 
}
```

## 10. Contribution
We encourage all the researchers in the field of PPG-based blood glucose level estimation to utilise this pipeline to evalute your models and add new datasets to this pipeline. If you would like to contribute to this pipeline, feel free to raise a pull request.


## References
1. Zeynali, M., Alipour, K., Tarvirdizadeh, B., & Ghamari, M. (2025). Non-invasive blood glucose monitoring using PPG signals with various deep learning models and implementation using TinyML. Scientific Reports, 15(1), 581.
2. Gupta, S. S., Kwon, T. H., Hossain, S., & Kim, K. D. (2021). Towards non-invasive blood glucose measurement using machine learning: An all-purpose PPG system design. Biomedical Signal Processing and Control, 68, 102706.
3. Islam, T. T., Ahmed, M. S., Hassanuzzaman, M., Bin Amir, S. A., & Rahman, T. (2021). Blood glucose level regression for smartphone ppg signals using machine learning. Applied Sciences, 11(2), 618.
4. Alghlayini, S., Hosni, A., & Atef, M. (2023, February). Photoplethysmography based blood glucose estimation using convolutional neural networks. In 2023 Advances in Science and Engineering Technology International Conferences (ASET) (pp. 1-5). IEEE.
5. Prabha, A., Yadav, J., Rani, A., & Singh, V. (2022). Intelligent estimation of blood glucose level using wristband PPG signal and physiological parameters. Biomedical Signal Processing and Control, 78, 103876.
6. Lee, H. C., Park, Y., Yoon, S. B., Yang, S. M., Park, D., & Jung, C. W. (2022). VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. Scientific Data, 9(1), 279.
7. Kermani, A., & Esmaeili, H. (2023). The dataset of photoplethysmography signals collected from a pulse sensor to measure blood glucose level. Mendeley Data, 3, 10-17632.
