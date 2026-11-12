import numpy as np
import pandas as pd
import warnings
import os
import json
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from pycatch22 import catch22_all

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
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
TARGET_LENGTH = 100
SEED = 3

# Directory containing the nine synthetic datasets
DATA_DIR = "../../../data/seed_3/length_50_100"
NOISE_COMBS = ["multiplicative", "multiplicative_outlier", "additive_multiplicative"]
STRENGTHS = ["low", "medium", "high"]

# Build dataset names and file paths
DATASET_NAMES = [f"{comb}_{strength}" for comb in NOISE_COMBS for strength in STRENGTHS]
DATA_PATHS = {name: os.path.join(DATA_DIR, f"timeseries_{name}.csv") for name in DATASET_NAMES}

# Output directory for catch22 + classical ML results
OUTPUT_DIR = "./machinelearning"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== Data Preprocessing =====================
def preprocess_data(file_path, target_length=TARGET_LENGTH):
    """
    Load one synthetic dataset, group rows by unique series ID,
    and linearly interpolate each series to a fixed target length.
    """
    df = pd.read_csv(file_path)
    print(f"\nPreprocessing data from: {file_path}")
    print(f"Original data shape: {df.shape}")

    df['unique_id'] = df['class_label'].astype(str) + '_' + df['sample_id'].astype(str)

    samples = []
    labels = []
    for unique_id, group in df.groupby('unique_id'):
        label = group['class_label'].iloc[0]
        group_sorted = group.sort_values('time_step_id')
        values = group_sorted['value'].values
        original_steps = group_sorted['time_step_id'].values

        if len(values) == 0:
            interpolated_values = np.zeros(target_length)
        else:
            f = interp1d(
                original_steps, values,
                kind='linear',
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

    samples = np.array(samples, dtype=np.float32)
    labels = np.array(labels)

    print(f"Samples shape: {samples.shape}")
    print(f"Labels shape: {labels.shape}")
    return samples, labels


# ===================== Feature Extraction =====================
def extract_catch22_features(samples):
    """
    Extract the 22 canonical catch22 features for each time series.
    Non-finite values are replaced with zero.
    """
    print("\nExtracting catch22 time series features...")
    features = []
    for series in samples:
        try:
            c22_result = catch22_all(series)
            feat = c22_result['values']
        except Exception as e:
            print(f"Warning: {e}, using zeros")
            feat = np.zeros(22)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(feat)
    features = np.array(features, dtype=np.float32)
    print(f"Extracted catch22 feature shape: {features.shape}")
    return features


# ===================== Model Tuning =====================
def tune_ml_light_params(X, y, dataset_name):
    """
    Lightweight grid search over six classical classifiers using catch22 features.
    A 3-fold cross-validation is used for model selection, and the best model
    is chosen by held-out test accuracy.
    """
    print(f"\n{'=' * 60}")
    print(f"Light Parameter Tuning on {dataset_name}")
    print(f"{'=' * 60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_params = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=SEED),
            "params": {
                "n_estimators": [200, 300],
                "max_depth": [20, None],
                "min_samples_split": [2, 5],
                "max_features": ['sqrt', None]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=SEED),
            "params": {
                "n_estimators": [200, 300],
                "max_depth": [6, 8],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        },
        "SVM": {
            "model": SVC(random_state=SEED, probability=False),
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
            "model": LogisticRegression(random_state=SEED, max_iter=2000),
            "params": {
                "C": [1, 10],
                "penalty": ['l2'],
                "solver": ['saga']
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=SEED),
            "params": {
                "max_depth": [10, None],
                "min_samples_split": [2, 5],
                "criterion": ['gini']
            }
        }
    }

    all_results = {}
    best_overall_acc = 0.0
    best_overall_model = None
    best_overall_name = ""

    for model_name, config in models_params.items():
        print(f"\nTuning {model_name}...")
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=3,
            verbose=0,
            scoring='accuracy',
            error_score='raise'
        )
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        test_acc = best_model.score(X_test_scaled, y_test)
        cv_acc = grid.best_score_

        all_results[model_name] = {
            "best_params": grid.best_params_,
            "cv_best_acc": cv_acc,
            "test_acc": test_acc,
            "classification_report": classification_report(
                y_test, best_model.predict(X_test_scaled), output_dict=True, zero_division=0
            )
        }
        print(f"{model_name} - CV Acc: {cv_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"   Best Params: {grid.best_params_}")

        if test_acc > best_overall_acc:
            best_overall_acc = test_acc
            best_overall_model = best_model
            best_overall_name = model_name

    print(f"\nBest model on {dataset_name}: {best_overall_name} (Test Acc: {best_overall_acc:.4f})")

    # Save the best model and the fitted scaler
    safe_name = dataset_name.replace(" ", "_")
    joblib.dump(best_overall_model, os.path.join(OUTPUT_DIR, f"best_{safe_name}_model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"{safe_name}_scaler.pkl"))

    # Save results as JSON
    results = {
        "dataset": dataset_name,
        "best_model_name": best_overall_name,
        "best_test_acc": best_overall_acc,
        "all_models_results": all_results
    }
    with open(os.path.join(OUTPUT_DIR, f"{safe_name}_results.json"), "w") as f:
        json.dump({
            k: v for k, v in results.items() if k != 'all_models_results' or isinstance(v, dict)
        }, f, indent=4, default=str)

    return results


# ===================== Main =====================
if __name__ == "__main__":
    all_results = []
    for dataset_name, file_path in DATA_PATHS.items():
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}, skipping.")
            continue
        samples, labels = preprocess_data(file_path, TARGET_LENGTH)
        features = extract_catch22_features(samples)
        res = tune_ml_light_params(features, labels, dataset_name)
        all_results.append(res)

    print("\n" + "=" * 60)
    print("Final catch22 + ML Summary on Strength Experiments")
    print("=" * 60)
    for res in all_results:
        print(f"\n{res['dataset']}:")
        print(f"   Best Model: {res['best_model_name']}")
        print(f"   Best Test Accuracy: {res['best_test_acc']:.4f}")
        print("   All Models Accuracy:")
        for model_name, model_res in res["all_models_results"].items():
            print(f"     - {model_name}: {model_res['test_acc']:.4f}")

    print(f"\nAll tuning completed. Results saved to {OUTPUT_DIR}/")