
import numpy as np
import pandas as pd
import os
import itertools
import json
import matplotlib.pyplot as plt

from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, Add,
    GlobalAveragePooling1D, Dense, Dropout, MaxPooling1D,
    Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ===================== Global Configuration =====================

TARGET_LENGTH = 50
NUM_CLASSES = 5
EPOCHS = 80
BATCH_SIZE = 64
PATIENCE = 12

SEED = 3

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Data directory (contains the nine generated datasets)
DATA_DIR = "../../../data/seed_3/length_50"

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

DATASET_NAMES = [
    f"{comb}_{strength}"
    for comb in NOISE_COMBS
    for strength in STRENGTHS
]

# Hyperparameter search spaces
CNN_PARAM_GRID = {
    "filters": [32, 64],
    "kernel_size": [3, 5],
    "num_layers": [2, 3],
    "dropout_rate": [0.2, 0.3],
    "lr": [0.0001, 0.001]
}

INCEPTION_PARAM_GRID = {
    "num_modules": [2, 3],
    "filters_per_module": [16, 32],
    "kernel_sizes": [[3, 5], [3, 5, 7]],
    "use_residual": [True, False],
    "dropout_rate": [0.2, 0.3],
    "lr": [0.0001, 0.001]
}

MAX_COMB = 16

# Data splitting proportions:
# 80% training data and 20% independent test data.
# The training data are further divided into:
# 90% training subset and 10% validation subset.
TEST_SIZE = 0.2
VAL_SIZE = 0.1


# ===================== Data Preprocessing =====================

def preprocess_data(file_path, target_length=TARGET_LENGTH):
    """
    Load data, linearly interpolate each series to the target length,
    standardize each sample independently, and generate stratified
    training and independent test sets.
    """

    df = pd.read_csv(file_path)

    print(f"\nLoading {file_path}")
    print(f"   Original shape: {df.shape}")

    df["unique_id"] = (
        df["class_label"].astype(str)
        + "_"
        + df["sample_id"].astype(str)
    )

    def resample_series(series, target_len):
        seq_len = len(series)

        if seq_len == target_len:
            return series.copy()

        orig_time = np.arange(seq_len)
        target_time = np.linspace(
            0, seq_len - 1, target_len
        )

        f = interpolate.interp1d(
            orig_time,
            series,
            kind="linear",
            fill_value="extrapolate"
        )

        return f(target_time)

    samples = []
    labels = []

    for unique_id, group in df.groupby("unique_id"):
        group = group.sort_values("time_step_id")

        values = group["value"].values
        label = group["class_label"].iloc[0]

        resampled = resample_series(
            values, target_length
        )

        samples.append(resampled)
        labels.append(label)

    X = np.array(samples)[:, :, np.newaxis]
    y = np.array(labels)

    print(f"   After resampling: X shape = {X.shape}")

    # Sample-wise standardization
    scaler = StandardScaler()

    X_scaled = []

    for sample in X:
        X_scaled.append(
            scaler.fit_transform(sample)
        )

    X_scaled = np.array(X_scaled)

    # Convert labels to one-hot encoding
    y_onehot = tf.keras.utils.to_categorical(
        y, num_classes=NUM_CLASSES
    )

    # Split the data into training and independent test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_onehot,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ===================== Model Definitions =====================

def build_cnn_model(
    filters=64,
    kernel_size=5,
    num_layers=2,
    dropout_rate=0.2,
    lr=0.001
):
    """
    Build a lightweight 1D CNN for time series classification.
    The number of filters in block i is filters * (i + 1).
    """

    inputs = Input(shape=(TARGET_LENGTH, 1))
    x = inputs

    for i in range(num_layers):
        x = Conv1D(
            filters * (i + 1),
            kernel_size,
            padding="same",
            kernel_initializer="he_normal"
        )(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling1D(pool_size=2)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def inception_module(
    x,
    filters=32,
    kernel_sizes=[3, 5, 7],
    bottleneck=True
):
    """
    Build a single Inception module with multiple convolution branches,
    a pooling branch, and batch normalization.
    """

    if bottleneck and int(x.shape[-1]) > 1:
        x_bottleneck = Conv1D(
            filters,
            1,
            padding="same",
            activation="relu"
        )(x)
    else:
        x_bottleneck = x

    conv_branches = []

    for ks in kernel_sizes:
        conv = Conv1D(
            filters,
            ks,
            padding="same",
            activation="relu"
        )(x_bottleneck)

        conv_branches.append(conv)

    pool = MaxPooling1D(
        3,
        strides=1,
        padding="same"
    )(x)

    pool_conv = Conv1D(
        filters,
        1,
        padding="same",
        activation="relu"
    )(pool)

    conv_branches.append(pool_conv)

    x = Concatenate(axis=-1)(conv_branches)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def build_inceptiontime_model(
    num_modules=3,
    filters_per_module=32,
    kernel_sizes=[3, 5, 7],
    use_residual=True,
    dropout_rate=0.3,
    lr=0.001
):
    """
    Build an InceptionTime model using stacked Inception modules
    with optional residual connections.
    """

    inputs = Input(shape=(TARGET_LENGTH, 1))
    x = inputs

    for i in range(num_modules):
        res = x

        x = inception_module(
            x,
            filters=filters_per_module,
            kernel_sizes=kernel_sizes
        )

        if use_residual and i > 0:
            if int(res.shape[-1]) != int(x.shape[-1]):
                res = Conv1D(
                    int(x.shape[-1]),
                    1,
                    padding="same"
                )(res)

            x = Add()([x, res])
            x = Activation("relu")(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ===================== Hyperparameter Search =====================

def manual_parameter_search(
    model_type,
    X_train,
    y_train,
    X_test,
    y_test,
    param_grid,
    max_comb=MAX_COMB
):
    """
    Perform hyperparameter search using validation accuracy.

    The test set is not used during hyperparameter selection.
    After selecting the best configuration, the final model is
    retrained on the complete training set and evaluated once
    on the independent test set.
    """

    best_val_acc = -1.0
    best_params = None
    best_epoch = None
    all_results = []

    # Split the original training set into training and validation subsets
    y_labels = np.argmax(y_train, axis=1)

    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y_labels
    )

    print("\nData partition for hyperparameter search:")
    print(f"   Subtraining set: {X_subtrain.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Independent test set: {X_test.shape}")

    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    combinations = list(
        itertools.product(*param_values)
    )

    # Randomly select a subset if the search space is too large
    if len(combinations) > max_comb:
        indices = np.random.choice(
            len(combinations),
            max_comb,
            replace=False
        )

        combinations = [
            combinations[i] for i in indices
        ]

    print(
        f"\nSearching {len(combinations)} "
        f"parameter combinations..."
    )

    # Evaluate each parameter combination using validation data
    for idx, params_tuple in enumerate(combinations):

        params = {
            param_names[i]: params_tuple[i]
            for i in range(len(param_names))
        }

        print(
            f"\nCombination {idx + 1}/"
            f"{len(combinations)}: {params}"
        )

        try:
            # Build the corresponding model
            if model_type == "cnn":
                model = build_cnn_model(**params)
            else:
                model = build_inceptiontime_model(**params)

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=PATIENCE,
                    restore_best_weights=True
                ),

                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]

            # Train the model using the training subset
            history = model.fit(
                X_subtrain,
                y_subtrain,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=0
            )

            # Evaluate the model on the validation set
            val_loss, val_acc = model.evaluate(
                X_val,
                y_val,
                verbose=0
            )

            train_acc = max(
                history.history["accuracy"]
            )

            # Record the epoch with the lowest validation loss
            current_best_epoch = (
                int(np.argmin(history.history["val_loss"])) + 1
            )

            print(
                f"   Validation accuracy: {val_acc:.4f}"
            )

            result = {
                "params": params,
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "best_epoch": current_best_epoch
            }

            all_results.append(result)

            # Select the best configuration using validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = float(val_acc)
                best_params = params.copy()
                best_epoch = current_best_epoch

        except Exception as e:
            print(
                f"   Error with params {params}: {str(e)}"
            )
            continue

    # Check whether any configuration was successfully evaluated
    if best_params is None:
        print(
            "Warning: No successful parameter combinations found!"
        )

        return None, None, 0.0, 0.0, all_results

    print("\nBest configuration selected:")
    print(f"   Parameters: {best_params}")
    print(f"   Validation accuracy: {best_val_acc:.4f}")
    print(f"   Selected training epochs: {best_epoch}")

    # Rebuild the final model using the selected configuration
    # and retrain it on the complete original training set
    if model_type == "cnn":
        best_model = build_cnn_model(**best_params)
    else:
        best_model = build_inceptiontime_model(**best_params)

    print("\nRetraining the selected model on the complete training set...")

    best_model.fit(
        X_train,
        y_train,
        epochs=best_epoch,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # Evaluate the final model on the independent test set only once
    test_loss, final_test_acc = best_model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("\nFinal independent test evaluation:")
    print(f"   Test loss: {test_loss:.4f}")
    print(f"   Test accuracy: {final_test_acc:.4f}")

    return (
        best_model,
        best_params,
        best_val_acc,
        float(final_test_acc),
        all_results
    )


# ===================== Main =====================

def main():

    all_results = []

    for dataset_name in DATASET_NAMES:

        file_path = os.path.join(
            DATA_DIR,
            f"timeseries_{dataset_name}.csv"
        )

        if not os.path.exists(file_path):
            print(
                f"Warning: File not found: {file_path}, skipping."
            )
            continue

        print("\n" + "=" * 80)
        print(f"Processing dataset: {dataset_name}")
        print("=" * 80)

        # Load and preprocess the dataset
        X_train, X_test, y_train, y_test = preprocess_data(
            file_path
        )

        # ===================== CNN Hyperparameter Search =====================

        print(f"\nTuning CNN on {dataset_name}...")

        (
            best_cnn,
            best_cnn_params,
            cnn_val_acc,
            cnn_test_acc,
            cnn_results
        ) = manual_parameter_search(
            "cnn",
            X_train,
            y_train,
            X_test,
            y_test,
            CNN_PARAM_GRID
        )

        if best_cnn is not None:

            # Save the final CNN model
            best_cnn.save(
                os.path.join(
                    OUTPUT_DIR,
                    f"cnn_{dataset_name}.h5"
                )
            )

            # Save the selected configuration and search results
            with open(
                os.path.join(
                    OUTPUT_DIR,
                    f"cnn_{dataset_name}_tuning.json"
                ),
                "w"
            ) as f:

                json.dump(
                    {
                        "best_params": best_cnn_params,
                        "best_val_acc": cnn_val_acc,
                        "final_test_acc": cnn_test_acc,
                        "all_results": cnn_results
                    },
                    f,
                    indent=4
                )

            print(
                f"CNN final model saved for {dataset_name}"
            )

        else:
            cnn_val_acc = 0.0
            cnn_test_acc = 0.0

        # ===================== InceptionTime Hyperparameter Search =====================

        print(
            f"\nTuning InceptionTime on {dataset_name}..."
        )

        (
            best_inc,
            best_inc_params,
            inc_val_acc,
            inc_test_acc,
            inc_results
        ) = manual_parameter_search(
            "inceptiontime",
            X_train,
            y_train,
            X_test,
            y_test,
            INCEPTION_PARAM_GRID
        )

        if best_inc is not None:

            # Save the final InceptionTime model
            best_inc.save(
                os.path.join(
                    OUTPUT_DIR,
                    f"inceptiontime_{dataset_name}.h5"
                )
            )

            # Save the selected configuration and search results
            with open(
                os.path.join(
                    OUTPUT_DIR,
                    f"inceptiontime_{dataset_name}_tuning.json"
                ),
                "w"
            ) as f:

                json.dump(
                    {
                        "best_params": best_inc_params,
                        "best_val_acc": inc_val_acc,
                        "final_test_acc": inc_test_acc,
                        "all_results": inc_results
                    },
                    f,
                    indent=4
                )

            print(
                f"InceptionTime final model saved for {dataset_name}"
            )

        else:
            inc_val_acc = 0.0
            inc_test_acc = 0.0

        # Record the final results for the current dataset
        all_results.append(
            {
                "dataset": dataset_name,
                "cnn_best_val_acc": cnn_val_acc,
                "cnn_final_test_acc": cnn_test_acc,
                "inception_best_val_acc": inc_val_acc,
                "inception_final_test_acc": inc_test_acc
            }
        )

    # ===================== Save Summary =====================

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "search_summary.csv"
        ),
        index=False
    )

    print("\n" + "=" * 80)
    print("Final results on all datasets:")
    print(results_df)

    # ===================== Plot Results =====================

    if len(results_df) > 0:

        # Plot results grouped by noise composition
        for comb in NOISE_COMBS:

            comb_results = results_df[
                results_df["dataset"].str.startswith(
                    comb + "_"
                )
            ]

            if comb_results.empty:
                continue

            # Preserve the predefined noise-strength order
            comb_results = (
                comb_results
                .set_index("dataset")
                .reindex(
                    [f"{comb}_{s}" for s in STRENGTHS]
                )
                .reset_index()
            )

            if comb_results[
                "cnn_final_test_acc"
            ].isna().any():
                print(
                    f"Warning: Missing data for {comb}, skipping plot."
                )
                continue

            x = np.arange(len(STRENGTHS))
            width = 0.35

            plt.figure(figsize=(8, 5))

            plt.bar(
                x - width / 2,
                comb_results["cnn_final_test_acc"],
                width,
                label="CNN",
                color="#1f77b4"
            )

            plt.bar(
                x + width / 2,
                comb_results["inception_final_test_acc"],
                width,
                label="InceptionTime",
                color="#ff7f0e"
            )

            plt.xticks(x, STRENGTHS)
            plt.ylabel("Final Test Accuracy")
            plt.title(
                f"Final Performance on {comb}"
            )

            plt.ylim(0, 1.0)
            plt.legend()
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    OUTPUT_DIR,
                    f"comparison_{comb}.png"
                ),
                dpi=150
            )

            plt.close()

        # Overall comparison across all datasets
        plt.figure(figsize=(14, 6))

        x = np.arange(len(results_df))
        width = 0.35

        plt.bar(
            x - width / 2,
            results_df["cnn_final_test_acc"],
            width,
            label="CNN",
            color="#1f77b4"
        )

        plt.bar(
            x + width / 2,
            results_df["inception_final_test_acc"],
            width,
            label="InceptionTime",
            color="#ff7f0e"
        )

        plt.xticks(
            x,
            results_df["dataset"],
            rotation=45,
            ha="right"
        )

        plt.ylabel("Final Test Accuracy")
        plt.title(
            "Final Test Performance on All Datasets"
        )

        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "all_comparison.png"
            ),
            dpi=150
        )

        plt.show()

    print(f"\nAll results saved in {OUTPUT_DIR}/")
    print("Hyperparameter search completed.")


if __name__ == "__main__":
    main()
