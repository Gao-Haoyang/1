
import numpy as np
import pandas as pd
import warnings
import os
import json
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from pycatch22 import catch22_all

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import joblib

warnings.filterwarnings('ignore')
np.random.seed(3)


# ===================== Global Configuration =====================

TARGET_LENGTH = 200
SEED = 3

# Directory containing the nine synthetic datasets
DATA_DIR = "../../../data/seed_3/length_200"

NOISE_COMBS = [
    "multiplicative",
    "multiplicative_outlier",
    "additive_multiplicative"
]

STRENGTHS = ["low", "medium", "high"]

# Build dataset names and file paths
DATASET_NAMES = [
    f"{comb}_{strength}"
    for comb in NOISE_COMBS
    for strength in STRENGTHS
]

DATA_PATHS = {
    name: os.path.join(DATA_DIR, f"timeseries_{name}.csv")
    for name in DATASET_NAMES
}

# Output directory
OUTPUT_DIR = "./machinelearning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Experimental settings
TEST_SIZE = 0.2
N_SPLITS = 3
GRID_N_JOBS = -1


# ===================== Data Preprocessing =====================

def preprocess_data(file_path, target_length=TARGET_LENGTH):
    """
    Load one synthetic dataset, group rows by unique series ID,
    and linearly interpolate each series to a fixed target length.

    Returns:
        samples: Interpolated time series.
        labels: Original class labels.
        sample_ids: Unique sample identifiers.
    """

    df = pd.read_csv(file_path)

    required_columns = [
        'class_label',
        'sample_id',
        'time_step_id',
        'value'
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {file_path}: {missing_columns}"
        )

    print(f"\nPreprocessing data from: {file_path}")
    print(f"Original data shape: {df.shape}")

    # Construct unique identifiers
    df['unique_id'] = (
        df['class_label'].astype(str)
        + '_'
        + df['sample_id'].astype(str)
    )

    samples = []
    labels = []
    sample_ids = []

    for unique_id, group in df.groupby('unique_id', sort=True):

        label = group['class_label'].iloc[0]

        # Sort observations by time
        group_sorted = group.sort_values('time_step_id')

        # Average observations with duplicated time points
        group_sorted = (
            group_sorted
            .groupby('time_step_id', as_index=False)['value']
            .mean()
        )

        original_steps = group_sorted['time_step_id'].to_numpy()
        values = group_sorted['value'].to_numpy(dtype=float)

        if len(values) == 0:
            raise ValueError(
                f"Empty time series encountered: {unique_id}"
            )

        elif len(values) == 1:
            # Repeat the only available observation
            interpolated_values = np.full(
                target_length,
                values[0]
            )

        else:
            f = interp1d(
                original_steps,
                values,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )

            target_steps = np.linspace(
                original_steps.min(),
                original_steps.max(),
                target_length
            )

            interpolated_values = f(target_steps)

        samples.append(interpolated_values)
        labels.append(label)
        sample_ids.append(unique_id)

    samples = np.asarray(samples, dtype=np.float32)
    labels = np.asarray(labels)
    sample_ids = np.asarray(sample_ids)

    print(f"Samples shape: {samples.shape}")
    print(f"Labels shape: {labels.shape}")

    return samples, labels, sample_ids


# ===================== Feature Extraction =====================

def extract_catch22_features(samples, sample_ids):
    """
    Extract the 22 canonical catch22 features for each time series.

    Non-finite values are replaced by NaN and subsequently handled
    inside the machine-learning pipeline.
    """

    print("\nExtracting catch22 time series features...")

    features = []
    failed_ids = []
    feature_names = None

    for i, series in enumerate(samples):

        try:
            c22_result = catch22_all(series)

            feat = np.asarray(
                c22_result['values'],
                dtype=np.float64
            )

            if len(feat) != 22:
                raise ValueError(
                    f"Expected 22 features, got {len(feat)}"
                )

            if feature_names is None:
                feature_names = c22_result['names']

        except Exception as e:

            print(
                f"Warning: Feature extraction failed for "
                f"{sample_ids[i]}: {e}"
            )

            feat = np.full(22, np.nan)
            failed_ids.append(str(sample_ids[i]))

        feat = np.where(np.isfinite(feat), feat, np.nan)

        features.append(feat)

    features = np.asarray(features, dtype=np.float32)

    if feature_names is None:
        feature_names = [
            f"catch22_feature_{i + 1}"
            for i in range(22)
        ]

    print(f"Extracted feature shape: {features.shape}")
    print(f"Failed samples: {len(failed_ids)}")

    return features, feature_names, failed_ids


# ===================== Model Configuration =====================

def get_models_and_params(n_classes):
    """
    Define six classical machine-learning classifiers
    and their corresponding hyperparameter search spaces.
    """

    models_params = {

        "RandomForest": {
            "model": RandomForestClassifier(
                random_state=SEED,
                n_jobs=1
            ),
            "params": {
                "n_estimators": [200, 300],
                "max_depth": [20, None],
                "min_samples_split": [2, 5],
                "max_features": ['sqrt', None]
            }
        },

        "XGBoost": {
            "model": XGBClassifier(
                random_state=SEED,
                objective='multi:softprob',
                num_class=n_classes,
                eval_metric='mlogloss',
                n_jobs=1
            ),
            "params": {
                "n_estimators": [200, 300],
                "max_depth": [6, 8],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        },

        "SVM": {
            "model": SVC(
                random_state=SEED,
                probability=False
            ),
            "params": {
                "C": [1, 10],
                "gamma": ['scale', 0.1],
                "kernel": ['rbf', 'linear']
            }
        },

        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [5, 7],
                "weights": ['distance'],
                "metric": ['euclidean']
            }
        },

        "LogisticRegression": {
            "model": LogisticRegression(
                random_state=SEED,
                max_iter=2000
            ),
            "params": {
                "C": [1, 10],
                "penalty": ['l2'],
                "solver": ['saga']
            }
        },

        "DecisionTree": {
            "model": DecisionTreeClassifier(
                random_state=SEED
            ),
            "params": {
                "max_depth": [10, None],
                "min_samples_split": [2, 5],
                "criterion": ['gini']
            }
        }
    }

    return models_params


# ===================== Model Tuning and Evaluation =====================

def tune_ml_light_params(
    X,
    y,
    sample_ids,
    dataset_name,
    feature_names
):
    """
    Perform model selection using training-set cross-validation.

    The test set is kept independent and is not used to select
    either hyperparameters or the final classifier.

    All six classifiers are evaluated on the test set after
    model selection, for transparent performance comparison.
    """

    print(f"\n{'=' * 60}")
    print(f"Machine Learning Experiments: {dataset_name}")
    print(f"{'=' * 60}")

    # Encode class labels to consecutive integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    n_classes = len(label_encoder.classes_)

    print(f"Number of classes: {n_classes}")
    print(f"Class labels: {label_encoder.classes_}")

    # Check whether each class has enough samples
    class_counts = np.bincount(y_encoded)

    if np.min(class_counts) < N_SPLITS:
        raise ValueError(
            f"{dataset_name}: Each class must contain at least "
            f"{N_SPLITS} samples for {N_SPLITS}-fold CV."
        )

    # Split data into training and independent test sets
    indices = np.arange(len(X))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_encoded
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    train_ids = sample_ids[train_idx]
    test_ids = sample_ids[test_idx]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Save sample identifiers for reproducibility
    safe_name = dataset_name.replace(" ", "_")

    split_records = {
        "dataset": dataset_name,
        "random_seed": SEED,
        "test_size": TEST_SIZE,
        "train_sample_ids": train_ids.tolist(),
        "test_sample_ids": test_ids.tolist(),
        "class_mapping": {
            str(label): int(index)
            for index, label in enumerate(
                label_encoder.classes_
            )
        }
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_split_ids.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            split_records,
            f,
            indent=4,
            ensure_ascii=False
        )

    # Define stratified cross-validation
    cv_strategy = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    models_params = get_models_and_params(n_classes)

    all_results = {}
    fitted_models = {}
    test_predictions = pd.DataFrame({
        "sample_id": test_ids,
        "true_label": label_encoder.inverse_transform(y_test)
    })

    # Select the final classifier using CV accuracy only
    best_overall_cv_acc = -np.inf
    best_overall_model = None
    best_overall_name = ""

    # Train and evaluate each classifier
    for model_name, config in models_params.items():

        print(f"\nTuning {model_name}...")

        # Put preprocessing inside the CV pipeline
        # This prevents information leakage between CV folds
        pipeline = Pipeline([
            (
                'imputer',
                SimpleImputer(
                    strategy='constant',
                    fill_value=0
                )
            ),
            ('scaler', StandardScaler()),
            ('classifier', config["model"])
        ])

        # Add the pipeline prefix to parameter names
        param_grid = {
            f'classifier__{key}': value
            for key, value in config["params"].items()
        }

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            verbose=0,
            scoring='accuracy',
            n_jobs=GRID_N_JOBS,
            refit=True,
            error_score='raise'
        )

        # Hyperparameter selection uses training data only
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        cv_acc = grid.best_score_

        # Evaluate the fitted model on the independent test set
        y_pred = best_model.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)

        # Convert encoded predictions back to original labels
        y_pred_original = label_encoder.inverse_transform(
            y_pred.astype(int)
        )

        test_predictions[
            f"{model_name}_pred"
        ] = y_pred_original

        # Generate the classification report
        report = classification_report(
            y_test,
            y_pred,
            labels=np.arange(n_classes),
            target_names=[
                str(label)
                for label in label_encoder.classes_
            ],
            output_dict=True,
            zero_division=0
        )

        # Remove the pipeline prefix from the saved parameters
        best_params = {
            key.replace('classifier__', ''): value
            for key, value in grid.best_params_.items()
        }

        # Save all hyperparameter candidates and their CV results
        cv_candidates = []

        for i in range(len(grid.cv_results_['params'])):

            candidate_params = {
                key.replace('classifier__', ''): value
                for key, value in
                grid.cv_results_['params'][i].items()
            }

            cv_candidates.append({
                "params": candidate_params,
                "mean_cv_accuracy": float(
                    grid.cv_results_['mean_test_score'][i]
                ),
                "std_cv_accuracy": float(
                    grid.cv_results_['std_test_score'][i]
                ),
                "rank": int(
                    grid.cv_results_['rank_test_score'][i]
                )
            })

        all_results[model_name] = {
            "best_params": best_params,
            "cv_best_acc": float(cv_acc),
            "test_acc": float(test_acc),
            "classification_report": report,
            "cv_candidates": cv_candidates
        }

        fitted_models[model_name] = best_model

        print(
            f"{model_name} - "
            f"CV Acc: {cv_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

        print(f"Best Params: {best_params}")

        # Choose the final classifier using CV accuracy,
        # NOT test accuracy
        if cv_acc > best_overall_cv_acc:

            best_overall_cv_acc = cv_acc
            best_overall_model = best_model
            best_overall_name = model_name

    # Obtain the test accuracy of the CV-selected classifier
    best_test_acc = all_results[
        best_overall_name
    ]["test_acc"]

    print(
        f"\nSelected model on {dataset_name}: "
        f"{best_overall_name}"
    )

    print(f"Best CV Accuracy: {best_overall_cv_acc:.4f}")
    print(f"Final Test Accuracy: {best_test_acc:.4f}")

    # Save the selected fitted pipeline
    # The pipeline already contains the imputer and scaler
    joblib.dump(
        best_overall_model,
        os.path.join(
            OUTPUT_DIR,
            f"best_{safe_name}_model.pkl"
        )
    )

    # Save the label encoder
    joblib.dump(
        label_encoder,
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_label_encoder.pkl"
        )
    )

    # Save sample-level predictions from all six classifiers
    test_predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_test_predictions.csv"
        ),
        index=False,
        encoding='utf-8-sig'
    )

    # Save extracted catch22 features and sample labels
    feature_df = pd.DataFrame(
        X,
        columns=feature_names
    )

    feature_df.insert(0, 'sample_id', sample_ids)
    feature_df.insert(
        1,
        'class_label',
        label_encoder.inverse_transform(y_encoded)
    )

    feature_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_catch22_features.csv"
        ),
        index=False,
        encoding='utf-8-sig'
    )

    # Save experiment results
    results = {
        "dataset": dataset_name,
        "random_seed": SEED,
        "target_length": TARGET_LENGTH,
        "test_size": TEST_SIZE,
        "cross_validation_folds": N_SPLITS,
        "selection_criterion": "mean cross-validation accuracy",
        "best_model_name": best_overall_name,
        "best_cv_acc": float(best_overall_cv_acc),
        "best_test_acc": float(best_test_acc),
        "all_models_results": all_results
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_results.json"
        ),
        "w",
        encoding='utf-8'
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    # Save feature extraction failure records
    return results


# ===================== Main =====================

if __name__ == "__main__":

    all_results = []

    for dataset_name, file_path in DATA_PATHS.items():

        if not os.path.exists(file_path):
            print(
                f"Warning: File not found: {file_path}, skipping."
            )
            continue

        # Load and preprocess time series
        samples, labels, sample_ids = preprocess_data(
            file_path,
            TARGET_LENGTH
        )

        # Extract catch22 features
        features, feature_names, failed_ids = (
            extract_catch22_features(
                samples,
                sample_ids
            )
        )

        # Save feature extraction failure records
        safe_name = dataset_name.replace(" ", "_")

        with open(
            os.path.join(
                OUTPUT_DIR,
                f"{safe_name}_feature_extraction.json"
            ),
            "w",
            encoding='utf-8'
        ) as f:

            json.dump(
                {
                    "dataset": dataset_name,
                    "failed_sample_ids": failed_ids,
                    "number_of_failed_samples": len(failed_ids)
                },
                f,
                indent=4,
                ensure_ascii=False
            )

        # Run machine-learning experiments
        res = tune_ml_light_params(
            features,
            labels,
            sample_ids,
            dataset_name,
            feature_names
        )

        all_results.append(res)

    # ===================== Final Summary =====================

    print("\n" + "=" * 60)
    print("Final catch22 + Classical ML Summary")
    print("=" * 60)

    for res in all_results:

        print(f"\n{res['dataset']}:")

        print(f"Selected Model: {res['best_model_name']}")

        print(f"Best CV Accuracy: {res['best_cv_acc']:.4f}")

        print(f"Final Test Accuracy: {res['best_test_acc']:.4f}")

        print("All Models Test Accuracy:")

        for model_name, model_res in (
            res["all_models_results"].items()
        ):

            print(
                f"  - {model_name}: "
                f"{model_res['test_acc']:.4f}"
            )

    print(
        f"\nAll experiments completed. "
        f"Results saved to {OUTPUT_DIR}/"
    )
