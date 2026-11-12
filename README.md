# A Comprehensive Evaluation of Shape Classification Methods for Population Time Series: From Synthetic Data to the Living Planet Database

## 1. Project Overview

This repository contains the code and experimental results for time-series morphology classification using deep learning and traditional machine learning methods.

The project focuses on evaluating classification performance under different data-generation conditions, sequence lengths, random seeds, and noise configurations. It also includes experiments on real-world time-series data to assess the performance and generalization ability of the proposed classification methods.

The repository is organized into five main components:

- `data/`: Generated time-series datasets.
- `data_generation/`: Scripts for generating synthetic datasets.
- `models/`: Model training and classification experiments.
- `real_LPD_experiments/`: Experiments using real-world LPD data.
- `split_records/`: Records of dataset splitting.

## 2. Repository Structure

```text
.
├── data_generation/
│   ├── seed_0/
│   ├── seed_1/
│   ├── seed_2/
│   ├── seed_3/
│   └── seed_42/
│
├── models/
│   ├── seed_0/
│   ├── seed_1/
│   ├── seed_2/
│   ├── seed_3/
│   └── seed_42/
│
├── real_LPD_experiments/
│   ├── deeplearning.py
│   ├── evaluate_test.py
│   ├── fixed_split.py
│   ├── unified_real_data_benchmark.py
│   ├── LPD.csv
│   ├── LPD_with_split.csv
│   ├── real_data_split.csv
│   ├── deeplearning_results/
│   ├── evaluate_test_results/
│   └── unified_real_data_benchmark_results/
│
├── split_records/
│   ├── length_50/
│   ├── length_50_100/
│   └── length_200/
│
├── best_experiment_configs/
│   ├── seed_0/
│   ├── seed_1/
│   ├── seed_2/
│   ├── seed_3/
│   └── seed_42/
│
└── README.md
```

Each synthetic-data experiment is organized by random seed and sequence length.

## 3. Synthetic Dataset Generation

### 3.1 Data Organization

The `data/` directory stores the generated time-series datasets.

Five random seeds are used:

- Seed 0
- Seed 1
- Seed 2
- Seed 3
- Seed 42

For each seed, datasets are organized according to three sequence-length settings:

- `length_50`
- `length_50_100`
- `length_200`

### 3.2 Noise Configurations

The generated datasets include different noise configurations:

- Multiplicative noise: low, medium, and high.
- Multiplicative noise with outliers: low, medium, and high.
- Additive and multiplicative noise: low, medium, and high.

Each configuration is stored as a CSV file. The corresponding `strength_metadata.csv` files record noise-strength information.

### 3.3 Data Generation Scripts

The `data_generation/` directory contains the scripts used to generate synthetic datasets.

The main scripts are:

- `generate_length_50.py`
- `generate_length_50_100.py`
- `generate_length_200.py`

These scripts are organized by random seed.

## 4. Classification Models

The `models/` directory contains the model training and classification experiments.

The experiments are organized by random seed and sequence length, consistent with the synthetic dataset structure.

The main scripts include:

- `deeplearning.py`
- `machine learning.py`
- `poly_classification.py` (available in the corresponding experiment directory)

### 4.1 Deep Learning Models

The deep-learning experiments include:

- Convolutional Neural Network (CNN)
- InceptionTime

The corresponding directories contain trained model files, hyperparameter tuning configurations, and experimental results.

Typical output files include:

- `.h5`: Saved deep-learning models.
- `_tuning.json`: Hyperparameter tuning configurations.
- Comparison figures (`.png`).
- `search_summary.csv`: Hyperparameter search summaries.

### 4.2 Machine Learning Models

Traditional machine learning experiments are stored in the `machinglearning/` directories.

Typical output files include:

- `.pkl`: Saved models and scalers.
- `_results.json`: Experimental results.

### 4.3 Polynomial Classification

The polynomial classification experiments include summary tables and comparison results.

For example:

- `Table4_FINAL_Independent5Seeds.csv`

## 5. Dataset Splitting

The `split_records/` directory stores the dataset-splitting records.

The records are organized according to sequence length:

- `length_50`
- `length_50_100`
- `length_200`

The CSV files record the splitting information for different noise configurations and random seeds.

These records are intended to support consistent dataset partitioning across experiments.

## 6. Real-World Data Experiments

The `real_LPD_experiments/` directory contains experiments based on real-world LPD data.

### 6.1 Data Files

The main data files include:

- `LPD.csv`
- `LPD_with_split.csv`
- `real_data_split.csv`

The directory also contains scripts for data splitting, model training, and evaluation.

### 6.2 Validation Experiments

`deeplearning.py` is used for deep-learning validation experiments.

The corresponding results are stored in `deeplearning_results/`, including:

- `Table10_FINAL_validation.csv`
- `Table10_FINAL_validation.xlsx`
- `Table10_seed_level_validation_results.csv`
- `Table10_validation_summary.csv`

### 6.3 Final Test Evaluation

`evaluate_test.py` performs the final evaluation using the held-out test data.

The results are stored in `evaluate_test_results/`, including:

- `Table11_FINAL_test.csv`
- `Table11_FINAL_test.xlsx`
- `Table11_overall_metrics.csv`
- `Figure8_FINAL_test.pdf`
- `Figure8_FINAL_test.png`
- `Figure8_normalized_confusion_matrix.csv`
- `confusion_matrix_raw.csv`
- `final_test_predictions.csv`

### 6.4 Unified Real-Data Benchmark

`unified_real_data_benchmark.py` conducts unified benchmark experiments on real-world data.

The corresponding results are stored in `unified_real_data_benchmark_results/`, including:

- `Table12_FINAL.csv`
- `Table12_FINAL.xlsx`
- `Table12_seed_level_results.csv`
- `fixed_few_shot_224_samples.csv`
- 
### 6.5 Selected Configuration for Real-World Time-Series Experiments
The repository provides the selected experimental configuration used in the final real-world time-series classification experiments.

The configuration file records the parameter settings adopted for the corresponding experiments and serves as a reference for reproducing the reported results.

The configuration is stored in:

real_LPD_experiments/best_experiment_configs/

This file documents the selected configuration for the real-world experiments. Other experimental settings can be found in the corresponding training and evaluation scripts.

## 7. Experimental Workflow

The overall experimental workflow can be summarized as follows:

1. **Generate synthetic datasets:** Run the scripts in `data_generation/` to generate time-series datasets under different random seeds, sequence lengths, and noise configurations.
2. **Prepare dataset splits:** Use the splitting records in `split_records/` to maintain consistent data partitions across experiments.
3. **Train classification models:** Run the corresponding scripts in `models/` to conduct deep-learning and traditional machine-learning experiments.
4. **Evaluate real-world data:** Run the scripts in `real_LPD_experiments/` to perform validation, final test evaluation, and unified benchmarking.
5. **Analyze experimental results:** Examine the generated CSV files, figures, and model outputs.

For the real-world experiments, validation and final test results are stored separately. The held-out test set should be reserved for final performance evaluation rather than model selection.

## 8. Environment and Dependencies

The experiments are implemented in Python.

The following packages may be required, depending on the specific script:

- NumPy
- Pandas
- SciPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Openpyxl
- tqdm

A possible installation command is:

```bash
pip install numpy pandas scipy scikit-learn tensorflow matplotlib seaborn openpyxl tqdm
```

The repository does not currently include a pinned dependency file. Therefore, the package versions used in the original experiments should be checked before reproducing the results.

## 9. Reproducibility Notes

To improve reproducibility, the following aspects should be considered:

- Use the same random seeds as those specified in the experiment directories.
- Keep the dataset sequence-length settings consistent.
- Use the corresponding noise configuration when loading synthetic datasets.
- Preserve the original dataset-splitting records when reproducing comparisons.
- Keep validation and final test datasets separate.
- Ensure that the required Python packages and their versions are compatible with the scripts.

Some scripts may contain environment-specific file paths. These paths should be updated according to the local execution environment before running the experiments.

## 10. Notes

- The `.idea/` directory contains IDE-related configuration files and is not required for running the experiments.
- The exact data-column definitions and class-label encoding should be checked in the corresponding data-generation and training scripts.
- The selected configuration for the final real-world time-series experiments is provided in best_experiment_configs/.
- Other experimental parameter settings can be checked in the corresponding data-generation, training, and evaluation scripts.
- The result files are provided for reference and experimental analysis.

## 11. Contact

For questions about the experimental code, dataset organization, or reproduction of the results, please contact the repository author.
