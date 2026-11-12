import numpy as np
import pandas as pd
import os
import itertools
import json
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, \
    GlobalAveragePooling1D, Dense, Dropout, MaxPooling1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===================== Global Configuration =====================
TARGET_LENGTH = 100
NUM_CLASSES = 5
EPOCHS = 80
BATCH_SIZE = 64
PATIENCE = 12
SEED = 3
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Data directory (contains the 9 generated datasets)
DATA_DIR = "../../../data/seed_3/length_50_100"
# Output directory
OUTPUT_DIR = "./deeplearning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset names follow the naming convention of the generation script
NOISE_COMBS = [
    "multiplicative",
    "multiplicative_outlier",
    "additive_multiplicative"
]
STRENGTHS = ["low", "medium", "high"]

DATASET_NAMES = [f"{comb}_{strength}" for comb in NOISE_COMBS for strength in STRENGTHS]

# Hyperparameter search spaces
CNN_PARAM_GRID = {
    'filters': [32, 64],
    'kernel_size': [3, 5],
    'num_layers': [2, 3],
    'dropout_rate': [0.2, 0.3],
    'lr': [0.0001, 0.001]
}
INCEPTION_PARAM_GRID = {
    'num_modules': [2, 3],
    'filters_per_module': [16, 32],
    'kernel_sizes': [[3, 5], [3, 5, 7]],
    'use_residual': [True, False],
    'dropout_rate': [0.2, 0.3],
    'lr': [0.0001, 0.001]
}
MAX_COMB = 16  # Maximum number of combinations tried per model


# ===================== Data Preprocessing =====================
def preprocess_data(file_path, target_length=TARGET_LENGTH):
    """
    Load data, linearly interpolate each series to the target length,
    standardize each sample independently, and return stratified
    train/test splits.
    """
    df = pd.read_csv(file_path)
    print(f"\nLoading {file_path}")
    print(f"   Shape: {df.shape}")

    df['unique_id'] = df['class_label'].astype(str) + '_' + df['sample_id'].astype(str)

    def resample_series(series, target_len):
        seq_len = len(series)
        if seq_len == target_len:
            return series.copy()
        orig_time = np.arange(seq_len)
        target_time = np.linspace(0, seq_len - 1, target_len)
        f = interpolate.interp1d(orig_time, series, kind='linear', fill_value='extrapolate')
        return f(target_time)

    samples, labels = [], []
    for unique_id, group in df.groupby('unique_id'):
        group = group.sort_values('time_step_id')
        values = group['value'].values
        label = group['class_label'].iloc[0]
        resampled = resample_series(values, target_length)
        samples.append(resampled)
        labels.append(label)

    X = np.array(samples)[:, :, np.newaxis]
    y = np.array(labels)
    print(f"   After resampling: X shape = {X.shape}")

    # Sample-wise standardization
    scaler = StandardScaler()
    X_scaled = []
    for sample in X:
        X_scaled.append(scaler.fit_transform(sample))
    X_scaled = np.array(X_scaled)

    y_onehot = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_onehot, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ===================== Model Definitions =====================
def build_cnn_model(filters=64, kernel_size=5, num_layers=2, dropout_rate=0.2, lr=0.001):
    """
    Build a lightweight 1D CNN for time series classification.
    The number of filters in block i is filters * (i + 1), i.e., linear growth.
    """
    inputs = Input(shape=(TARGET_LENGTH, 1))
    x = inputs
    for i in range(num_layers):
        x = Conv1D(filters * (i + 1), kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def inception_module(x, filters=32, kernel_sizes=[3, 5, 7], bottleneck=True):
    """
    Single Inception module: optional bottleneck, multiple Conv1D branches
    with different kernel sizes, and a MaxPooling + 1x1 Conv branch.
    """
    if bottleneck and int(x.shape[-1]) > 1:
        x_bottleneck = Conv1D(filters, 1, padding='same', activation='relu')(x)
    else:
        x_bottleneck = x
    conv_branches = []
    for ks in kernel_sizes:
        conv = Conv1D(filters, ks, padding='same', activation='relu')(x_bottleneck)
        conv_branches.append(conv)
    pool = MaxPooling1D(3, strides=1, padding='same')(x)
    pool_conv = Conv1D(filters, 1, padding='same', activation='relu')(pool)
    conv_branches.append(pool_conv)
    x = Concatenate(axis=-1)(conv_branches)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def build_inceptiontime_model(num_modules=3, filters_per_module=32, kernel_sizes=[3, 5, 7],
                              use_residual=True, dropout_rate=0.3, lr=0.001):
    """
    Build an InceptionTime model: stacked Inception modules with optional
    residual connections, followed by global average pooling and a dense classifier.
    """
    inputs = Input(shape=(TARGET_LENGTH, 1))
    x = inputs
    for i in range(num_modules):
        res = x
        x = inception_module(x, filters=filters_per_module, kernel_sizes=kernel_sizes)
        if use_residual and i > 0:
            if int(res.shape[-1]) != int(x.shape[-1]):
                res = Conv1D(int(x.shape[-1]), 1, padding='same')(res)
            x = Add()([x, res])
            x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ===================== Parameter Search Functions =====================
def manual_parameter_search(model_type, X_train, y_train, X_test, y_test, param_grid, max_comb=MAX_COMB):
    """
    Grid search over the specified parameter space. If the full grid exceeds
    max_comb, a random subset of max_comb combinations is evaluated.
    The best model is selected based on test accuracy.
    """
    best_acc = 0.0
    best_model = None
    best_params = None
    all_results = []

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    if len(combinations) > max_comb:
        indices = np.random.choice(len(combinations), max_comb, replace=False)
        combinations = [combinations[i] for i in indices]

    print(f"\nSearching {len(combinations)} parameter combinations...")
    for idx, params_tuple in enumerate(combinations):
        params = {param_names[i]: params_tuple[i] for i in range(len(param_names))}
        print(f"\nCombination {idx+1}/{len(combinations)}: {params}")
        try:
            if model_type == "cnn":
                model = build_cnn_model(**params)
            else:
                model = build_inceptiontime_model(**params)

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
            history = model.fit(
                X_train, y_train,
                validation_split=0.1,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=0
            )
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"   Test accuracy: {test_acc:.4f}")

            result = {
                "params": params,
                "test_acc": test_acc,
                "val_acc": max(history.history['val_accuracy']),
                "train_acc": max(history.history['accuracy'])
            }
            all_results.append(result)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = model
                best_params = params
        except Exception as e:
            print(f"   Error with params {params}: {str(e)}")
            continue

    if best_model is None:
        print("Warning: No successful parameter combinations found!")
        return None, None, 0.0, all_results

    print(f"\nBest combination found: {best_params} (Test accuracy: {best_acc:.4f})")
    return best_model, best_params, best_acc, all_results


# ===================== Main =====================
def main():
    all_results = []
    for dataset_name in DATASET_NAMES:
        file_path = os.path.join(DATA_DIR, f"timeseries_{dataset_name}.csv")
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}, skipping.")
            continue

        print("\n" + "=" * 80)
        print(f"Processing dataset: {dataset_name}")
        print("=" * 80)

        X_train, X_test, y_train, y_test = preprocess_data(file_path)

        # CNN hyperparameter search
        print(f"\nTuning CNN on {dataset_name}...")
        best_cnn, best_cnn_params, cnn_acc, cnn_results = manual_parameter_search(
            "cnn", X_train, y_train, X_test, y_test, CNN_PARAM_GRID
        )
        if best_cnn is not None:
            best_cnn.save(os.path.join(OUTPUT_DIR, f"cnn_{dataset_name}.h5"))
            with open(os.path.join(OUTPUT_DIR, f"cnn_{dataset_name}_tuning.json"), "w") as f:
                json.dump({"best_params": best_cnn_params, "best_acc": cnn_acc,
                           "all_results": cnn_results}, f, indent=4)
            print(f"CNN best model saved for {dataset_name}")
        else:
            cnn_acc = 0.0

        # InceptionTime hyperparameter search
        print(f"\nTuning InceptionTime on {dataset_name}...")
        best_inc, best_inc_params, inc_acc, inc_results = manual_parameter_search(
            "inceptiontime", X_train, y_train, X_test, y_test, INCEPTION_PARAM_GRID
        )
        if best_inc is not None:
            best_inc.save(os.path.join(OUTPUT_DIR, f"inceptiontime_{dataset_name}.h5"))
            with open(os.path.join(OUTPUT_DIR, f"inceptiontime_{dataset_name}_tuning.json"), "w") as f:
                json.dump({"best_params": best_inc_params, "best_acc": inc_acc,
                           "all_results": inc_results}, f, indent=4)
            print(f"InceptionTime best model saved for {dataset_name}")
        else:
            inc_acc = 0.0

        all_results.append({
            'dataset': dataset_name,
            'cnn_best_acc': cnn_acc,
            'inception_best_acc': inc_acc
        })

    # Summary of results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "search_summary.csv"), index=False)
    print("\n" + "=" * 80)
    print("Summary of parameter search on all datasets:")
    print(results_df)

    # Plot accuracy comparison grouped by noise composition
    if len(results_df) > 0:
        # Generate one subplot for each noise composition
        for comb in NOISE_COMBS:
            # Exact match: composition name followed by underscore
            comb_results = results_df[results_df['dataset'].str.startswith(comb + '_')]
            if comb_results.empty:
                continue
            # Ensure the strength order is preserved
            comb_results = comb_results.set_index('dataset').reindex(
                [f"{comb}_{s}" for s in STRENGTHS]).reset_index()
            # If any strength is missing, reindex introduces NaN; skip plotting
            if comb_results['cnn_best_acc'].isna().any():
                print(f"Warning: Missing data for some strengths in {comb}, skipping plot.")
                continue
            x = np.arange(len(STRENGTHS))
            width = 0.35
            plt.figure(figsize=(8, 5))
            plt.bar(x - width / 2, comb_results['cnn_best_acc'], width, label='CNN', color='#1f77b4')
            plt.bar(x + width / 2, comb_results['inception_best_acc'], width,
                    label='InceptionTime', color='#ff7f0e')
            plt.xticks(x, STRENGTHS)
            plt.ylabel('Best Test Accuracy')
            plt.title(f'Performance on {comb} (by noise strength)')
            plt.ylim(0, 1.0)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_{comb}.png"), dpi=150)
            plt.close()

        # Overall comparison across all datasets
        plt.figure(figsize=(14, 6))
        x = np.arange(len(results_df))
        width = 0.35
        plt.bar(x - width / 2, results_df['cnn_best_acc'], width, label='CNN', color='#1f77b4')
        plt.bar(x + width / 2, results_df['inception_best_acc'], width,
                label='InceptionTime', color='#ff7f0e')
        plt.xticks(x, results_df['dataset'], rotation=45, ha='right')
        plt.ylabel('Best Test Accuracy')
        plt.title('Best Model Performance on All Datasets')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "all_comparison.png"), dpi=150)
        plt.show()

    print(f"\nAll search results saved in {OUTPUT_DIR}/")
    print("Parameter search completed.")


if __name__ == "__main__":
    main()