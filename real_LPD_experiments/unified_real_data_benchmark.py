import os
import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import interpolate

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

import pycatch22

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 0. Basic configuration
# ============================================================

# Data with the fixed train / validation / test split
REAL_DATA_PATH = (
    "./LPD_with_split.csv"
)

# Output directory
OUTPUT_DIR = (
    "./unified_real_data_benchmark_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Five random seeds
# ============================================================

SEEDS = [
    0,
    1,
    2,
    3,
    42
]

# ============================================================
# Synthetic pre-trained model directory for each seed
# ============================================================

SEED_MODEL_DIRS = {

    0: "../models/seed_0/length_50_100/deeplearning",
    1: "../models/seed_1/length_50_100/deeplearning",
    2: "../models/seed_2/length_50_100/deeplearning",
    3: "../models/seed_3/length_50_100/deeplearning",
    42: "../models/seed_42/length_50_100/deeplearning"
}


# ============================================================
# Best CNN configuration locked by the validation set
# ============================================================

BEST_CONFIG = (
    "additive_multiplicative_high"
)


# ============================================================
# Data / model parameters
# ============================================================

TARGET_LENGTH = 100

NUM_CLASSES = 5

EPOCHS = 80

FULL_BATCH_SIZE = 32

FEW_SHOT_BATCH_SIZE = 16

PATIENCE = 10

FINE_TUNE_LR = 1e-4


# ============================================================
# Few-shot
#
# 224 / 1121 ≈ 20%
#
# These 224 samples must be drawn exclusively from the 672
# training samples. Neither the validation nor the test split
# may enter few-shot training.
# ============================================================

FEW_SHOT_N = 224

FEW_SHOT_RANDOM_STATE = 42


# ============================================================

# Freeze Input + the first Conv / BN / ReLU / MaxPool block,
# i.e., the first 5 layers
# ============================================================

FREEZE_FIRST_N_LAYERS = 5


# ============================================================
# 1. Set random seeds
# ============================================================

def set_all_seeds(seed):

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)


# ============================================================
# 2. Load real data and preprocess by split
# ============================================================

def load_dl_split(
        file_path,
        split_name
):
    """
    Load a fixed split, interpolate each series to TARGET_LENGTH,
    and standardize each series independently.
    """

    df = pd.read_csv(
        file_path
    )

    required_columns = [
        "class_label",
        "sample_id",
        "time_step_id",
        "value",
        "split"
    ]

    for col in required_columns:

        if col not in df.columns:

            raise ValueError(
                f"Missing column: {col}"
            )


    # --------------------------------------------------------
    # Keep only the specified split
    # --------------------------------------------------------

    df = df[
        df["split"]
        .astype(str)
        .str.lower()
        ==
        split_name.lower()
    ].copy()


    if len(df) == 0:

        raise ValueError(
            f"No samples found for split={split_name}"
        )


    # --------------------------------------------------------
    # Construct unique_id if absent
    # --------------------------------------------------------

    if "unique_id" not in df.columns:

        df["unique_id"] = (
            df["class_label"]
            .astype(str)
            +
            "_"
            +
            df["sample_id"]
            .astype(str)
        )


    samples = []

    labels = []

    sample_ids = []


    # --------------------------------------------------------
    # Each unique_id corresponds to one time series
    # --------------------------------------------------------

    for unique_id, group in df.groupby(
        "unique_id",
        sort=False
    ):

        group = group.sort_values(
            "time_step_id"
        )

        values = (
            group["value"]
            .astype(float)
            .values
        )

        label = int(
            group["class_label"]
            .iloc[0]
        )


        # ====================================================
        # Interpolate to TARGET_LENGTH points
        # ====================================================

        seq_len = len(
            values
        )


        if seq_len == TARGET_LENGTH:

            resampled = (
                values.copy()
            )

        else:

            original_time = np.arange(
                seq_len
            )

            target_time = np.linspace(
                0,
                seq_len - 1,
                TARGET_LENGTH
            )

            f = interpolate.interp1d(

                original_time,

                values,

                kind="linear",

                fill_value="extrapolate"
            )

            resampled = f(
                target_time
            )


        # ====================================================
        # Standardize each sample independently,
        # consistent with the original CNN preprocessing
        # ====================================================

        scaler = StandardScaler()

        scaled = scaler.fit_transform(
            resampled.reshape(
                -1,
                1
            )
        ).flatten()


        samples.append(
            scaled
        )

        labels.append(
            label
        )

        sample_ids.append(
            unique_id
        )


    X = np.array(
        samples,
        dtype=np.float32
    )[:, :, np.newaxis]


    y = np.array(
        labels,
        dtype=int
    )


    return (
        X,
        y,
        sample_ids
    )


# ============================================================
# 3. Load the three fixed splits
# ============================================================

def load_all_splits():
    """
    Load the fixed train / validation / test splits and verify
    their sizes and mutual exclusivity.
    """

    print(
        "=" * 80
    )

    print(
        "LOADING FIXED REAL-DATA SPLITS"
    )

    print(
        "=" * 80
    )


    X_train, y_train, train_ids = (
        load_dl_split(
            REAL_DATA_PATH,
            "train"
        )
    )


    X_val, y_val, val_ids = (
        load_dl_split(
            REAL_DATA_PATH,
            "validation"
        )
    )


    X_test, y_test, test_ids = (
        load_dl_split(
            REAL_DATA_PATH,
            "test"
        )
    )


    print(
        f"\nTrain:      {len(y_train)}"
    )

    print(
        f"Validation: {len(y_val)}"
    )

    print(
        f"Test:       {len(y_test)}"
    )


    # --------------------------------------------------------
    # Hard assertions
    # --------------------------------------------------------

    assert len(
        y_train
    ) == 672

    assert len(
        y_val
    ) == 224

    assert len(
        y_test
    ) == 225


    assert len(
        set(train_ids)
        &
        set(val_ids)
    ) == 0


    assert len(
        set(train_ids)
        &
        set(test_ids)
    ) == 0


    assert len(
        set(val_ids)
        &
        set(test_ids)
    ) == 0


    print(
        "\nFixed split verified."
    )

    print(
        "No overlap between Train / Validation / Test."
    )


    return {

        "X_train":
            X_train,

        "y_train":
            y_train,

        "train_ids":
            train_ids,

        "X_val":
            X_val,

        "y_val":
            y_val,

        "val_ids":
            val_ids,

        "X_test":
            X_test,

        "y_test":
            y_test,

        "test_ids":
            test_ids
    }


# ============================================================
# 4. Locate pre-trained CNN
# ============================================================

def find_cnn_model(
        seed
):
    """
    Locate the pre-trained CNN model for a given seed. First try the
    standard filename; if not found, fall back to fuzzy matching.
    """

    model_dir = (
        SEED_MODEL_DIRS[
            seed
        ]
    )


    standard_candidates = [

        os.path.join(
            model_dir,
            f"cnn_{BEST_CONFIG}.h5"
        ),

        os.path.join(
            model_dir,
            f"cnn_{BEST_CONFIG}.keras"
        )
    ]


    for path in standard_candidates:

        if os.path.isfile(
            path
        ):

            return path


    # --------------------------------------------------------
    # Fuzzy matching
    # --------------------------------------------------------

    patterns = [

        os.path.join(
            model_dir,
            f"*cnn*{BEST_CONFIG}*.h5"
        ),

        os.path.join(
            model_dir,
            f"*cnn*{BEST_CONFIG}*.keras"
        )
    ]


    matches = []

    for pattern in patterns:

        matches.extend(
            glob.glob(
                pattern
            )
        )


    matches = list(
        dict.fromkeys(
            matches
        )
    )


    if len(matches) == 1:

        return matches[0]


    if len(matches) > 1:

        print(
            f"\nWARNING: Seed {seed}: multiple models found."
        )

        for p in matches:

            print(
                p
            )

        print(
            "Using the first match:"
        )

        print(
            matches[0]
        )

        return matches[0]


    raise FileNotFoundError(
        f"Cannot find CNN for seed={seed}\n"
        f"Directory={model_dir}\n"
        f"Configuration={BEST_CONFIG}"
    )


# ============================================================
# 5. Zero-shot CNN
# ============================================================

def run_zero_shot(
        X_test,
        y_test
):
    """
    Evaluate the synthetic pre-trained CNNs directly on the test set
    without any real-data fine-tuning.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ZERO-SHOT CNN"
    )

    print(
        "=" * 80
    )


    results = []


    for seed in SEEDS:

        set_all_seeds(
            seed
        )

        model_path = (
            find_cnn_model(
                seed
            )
        )


        print(
            f"\nSeed {seed}"
        )

        print(
            model_path
        )


        model = (
            tf.keras.models
            .load_model(
                model_path
            )
        )


        probs = model.predict(
            X_test,
            verbose=0
        )


        pred = np.argmax(
            probs,
            axis=1
        )


        accuracy = accuracy_score(
            y_test,
            pred
        )


        print(
            f"Test Accuracy: "
            f"{accuracy:.4f}"
        )


        results.append(
            {
                "method":
                    "Zero-shot CNN",

                "seed":
                    seed,

                "accuracy":
                    accuracy
            }
        )


        del model

        tf.keras.backend.clear_session()


    return results


# ============================================================
# 6. Fixed few-shot subset
# ============================================================

def create_fixed_few_shot_subset(
        X_train,
        y_train,
        train_ids
):
    """
    Draw a fixed stratified subset of 224 training samples
    (approximately 20% of the full 1,121 real-world samples).
    """

    # --------------------------------------------------------
    # Fixed draw of 224 samples
    # Approximately 20% of the 1,121 real-world samples
    # --------------------------------------------------------

    all_indices = np.arange(
        len(y_train)
    )


    few_indices, _ = train_test_split(

        all_indices,

        train_size=FEW_SHOT_N,

        random_state=FEW_SHOT_RANDOM_STATE,

        stratify=y_train
    )


    X_few = (
        X_train[
            few_indices
        ]
    )


    y_few = (
        y_train[
            few_indices
        ]
    )


    few_ids = [
        train_ids[i]
        for i in few_indices
    ]


    print(
        "\n"
        + "=" * 80
    )

    print(
        "FIXED FEW-SHOT TRAINING SUBSET"
    )

    print(
        "=" * 80
    )


    print(
        f"N = {len(y_few)}"
    )


    print(
        "\nClass distribution:"
    )

    print(
        pd.Series(
            y_few
        )
        .value_counts()
        .sort_index()
    )


    # --------------------------------------------------------
    # Save the selected IDs
    # --------------------------------------------------------

    few_df = pd.DataFrame(
        {
            "unique_id":
                few_ids,

            "class_label":
                y_few
        }
    )


    few_path = os.path.join(
        OUTPUT_DIR,
        "fixed_few_shot_224_samples.csv"
    )


    few_df.to_csv(
        few_path,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        f"\nSaved: {few_path}"
    )


    return (
        X_few,
        y_few,
        few_ids
    )


# ============================================================
# 7. Prepare model for fine-tuning
# ============================================================

def prepare_model_for_finetuning(
        model_path,
        seed
):
    """
    Load a pre-trained CNN and freeze the first five layers
    for fine-tuning.
    """

    set_all_seeds(
        seed
    )


    model = (
        tf.keras.models
        .load_model(
            model_path
        )
    )


    # --------------------------------------------------------
    # Freeze the first 5 layers
    # --------------------------------------------------------

    for i, layer in enumerate(
        model.layers
    ):

        if i < FREEZE_FIRST_N_LAYERS:

            layer.trainable = False

        else:

            layer.trainable = True


    # --------------------------------------------------------
    # Recompile the model
    # --------------------------------------------------------

    model.compile(

        optimizer=Adam(
            learning_rate=FINE_TUNE_LR
        ),

        loss="categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    return model


# ============================================================
# 8. Single-seed fine-tuning
# ============================================================

def fine_tune_one_seed(
        seed,
        X_train_ft,
        y_train_ft,
        X_val,
        y_val,
        X_test,
        y_test,
        batch_size
):
    """
    Fine-tune one pre-trained CNN on real training samples and
    evaluate on the held-out test set.
    """

    model_path = (
        find_cnn_model(
            seed
        )
    )


    model = (
        prepare_model_for_finetuning(
            model_path,
            seed
        )
    )


    y_train_onehot = (
        tf.keras.utils
        .to_categorical(
            y_train_ft,
            num_classes=NUM_CLASSES
        )
    )


    y_val_onehot = (
        tf.keras.utils
        .to_categorical(
            y_val,
            num_classes=NUM_CLASSES
        )
    )


    callbacks = [

        EarlyStopping(

            monitor="val_loss",

            patience=PATIENCE,

            restore_best_weights=True,

            verbose=0
        ),

        ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=5,

            min_lr=1e-6,

            verbose=0
        )
    ]


    # --------------------------------------------------------
    # The fixed 224 validation samples are used for early stopping.
    # Do NOT use validation_split here.
    # --------------------------------------------------------

    history = model.fit(

        X_train_ft,

        y_train_onehot,

        validation_data=(
            X_val,
            y_val_onehot
        ),

        epochs=EPOCHS,

        batch_size=batch_size,

        callbacks=callbacks,

        verbose=0
    )


    # --------------------------------------------------------
    # Final evaluation only on the fixed 225 test samples
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        verbose=0
    )


    predictions = np.argmax(
        probabilities,
        axis=1
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    best_epoch = (
        np.argmin(
            history.history[
                "val_loss"
            ]
        )
        +
        1
    )


    del model

    tf.keras.backend.clear_session()


    return (
        accuracy,
        best_epoch
    )


# ============================================================
# 9. Few-shot CNN
# ============================================================

def run_few_shot(
        X_few,
        y_few,
        X_val,
        y_val,
        X_test,
        y_test
):
    """
    Fine-tune the pre-trained CNN on a fixed 224-sample subset
    and evaluate on the held-out test set.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FEW-SHOT CNN"
    )

    print(
        f"Training samples = {len(y_few)}"
    )

    print(
        "=" * 80
    )


    results = []


    for seed in SEEDS:

        print(
            f"\nSeed {seed}"
        )


        accuracy, best_epoch = (
            fine_tune_one_seed(

                seed,

                X_few,

                y_few,

                X_val,

                y_val,

                X_test,

                y_test,

                FEW_SHOT_BATCH_SIZE
            )
        )


        print(
            f"Test Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Best Epoch: "
            f"{best_epoch}"
        )


        results.append(
            {
                "method":
                    "Few-shot CNN",

                "seed":
                    seed,

                "accuracy":
                    accuracy,

                "best_epoch":
                    best_epoch
            }
        )


    return results


# ============================================================
# 10. Full-data fine-tuned CNN
# ============================================================

def run_full_data_finetune(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
):
    """
    Fine-tune the pre-trained CNN on all 672 training samples
    and evaluate on the held-out test set.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FULL-DATA FINE-TUNED CNN"
    )

    print(
        f"Training samples = {len(y_train)}"
    )

    print(
        "=" * 80
    )


    results = []


    for seed in SEEDS:

        print(
            f"\nSeed {seed}"
        )


        accuracy, best_epoch = (
            fine_tune_one_seed(

                seed,

                X_train,

                y_train,

                X_val,

                y_val,

                X_test,

                y_test,

                FULL_BATCH_SIZE
            )
        )


        print(
            f"Test Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Best Epoch: "
            f"{best_epoch}"
        )


        results.append(
            {
                "method":
                    "Full-data fine-tuned CNN",

                "seed":
                    seed,

                "accuracy":
                    accuracy,

                "best_epoch":
                    best_epoch
            }
        )


    return results




# ============================================================
# 10A. Scratch CNN: same architecture, random initialization, trained from scratch
# ============================================================

def reinitialize_cloned_model(model, seed):
    """
    Keep the network architecture of the pre-trained CNN but discard
    all pre-trained weights. Conv1D / Dense layers are reinitialized with
    their original initializer types; BatchNormalization layers are reset
    to the standard initial state.
    """
    for layer_index, layer in enumerate(model.layers):
        weights = layer.get_weights()

        if not weights:
            continue

        layer_seed = int(seed * 1009 + layer_index * 9176 + 17)

        if isinstance(layer, (tf.keras.layers.Conv1D,
                              tf.keras.layers.Dense)):
            new_weights = []

            kernel_initializer = layer.kernel_initializer
            kernel_config = kernel_initializer.get_config()

            if "seed" in kernel_config:
                kernel_config["seed"] = layer_seed

            kernel_initializer_new = type(kernel_initializer).from_config(
                kernel_config
            )

            new_kernel = kernel_initializer_new(
                layer.kernel.shape,
                dtype=layer.kernel.dtype
            )
            new_weights.append(new_kernel.numpy())

            if layer.use_bias:
                bias_initializer = layer.bias_initializer
                bias_config = bias_initializer.get_config()

                if "seed" in bias_config:
                    bias_config["seed"] = layer_seed + 1

                bias_initializer_new = type(bias_initializer).from_config(
                    bias_config
                )

                new_bias = bias_initializer_new(
                    layer.bias.shape,
                    dtype=layer.bias.dtype
                )
                new_weights.append(new_bias.numpy())

            layer.set_weights(new_weights)

        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            # gamma=1, beta=0, moving_mean=0, moving_variance=1
            layer.set_weights([
                np.ones_like(weights[0]),
                np.zeros_like(weights[1]),
                np.zeros_like(weights[2]),
                np.ones_like(weights[3])
            ])

        else:
            raise TypeError(
                "Scratch initialization encountered an unsupported "
                f"weighted layer: {layer.name} ({layer.__class__.__name__}). "
                "Please define an explicit reinitialization rule for it."
            )

    return model


def scratch_cnn_one_seed(
        seed,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
):
    """
    Use the seed-specific pre-trained model only as an architecture
    template. clone_model creates the same structure, and all trainable
    weights are then randomly reinitialized. All layers are trainable.
    """
    set_all_seeds(seed)

    architecture_path = find_cnn_model(seed)

    architecture_template = tf.keras.models.load_model(
        architecture_path,
        compile=False
    )

    # Copy the network architecture without inheriting the trained weights
    model = tf.keras.models.clone_model(architecture_template)

    # Explicitly reinitialize every weighted layer
    model = reinitialize_cloned_model(model, seed)

    # All layers are trainable for the scratch CNN
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    y_train_onehot = tf.keras.utils.to_categorical(
        y_train,
        num_classes=NUM_CLASSES
    )
    y_val_onehot = tf.keras.utils.to_categorical(
        y_val,
        num_classes=NUM_CLASSES
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
    ]

    history = model.fit(
        X_train,
        y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=EPOCHS,
        batch_size=FULL_BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, predictions)

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

    del model
    del architecture_template
    tf.keras.backend.clear_session()

    return accuracy, best_epoch


def run_scratch_cnn(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
):
    """
    Train a randomly initialized CNN with the same architecture as the
    pre-trained model, using all 672 real training samples.
    """
    print("\n" + "=" * 80)
    print("SCRATCH CNN (RANDOM INITIALIZATION)")
    print(f"Training samples = {len(y_train)}")
    print("=" * 80)

    results = []

    for seed in SEEDS:
        print(f"\nSeed {seed}")

        accuracy, best_epoch = scratch_cnn_one_seed(
            seed,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        )

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Best Epoch: {best_epoch}")

        results.append({
            "method": "Scratch CNN",
            "seed": seed,
            "accuracy": accuracy,
            "best_epoch": best_epoch
        })

    return results




# ============================================================
# 10B. Pretrained CNN with all layers unfrozen
# Matched comparison against the scratch CNN
# (same data and training setup)
# ============================================================

def pretrained_all_layers_one_seed(
        seed,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
):
    """
    Fine-tune a pre-trained CNN with all layers trainable, using the
    same data and training setup as the scratch CNN.
    """
    set_all_seeds(seed)

    model_path = find_cnn_model(seed)

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    # All layers are trainable, matching the scratch CNN setup
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    y_train_onehot = tf.keras.utils.to_categorical(
        y_train,
        num_classes=NUM_CLASSES
    )
    y_val_onehot = tf.keras.utils.to_categorical(
        y_val,
        num_classes=NUM_CLASSES
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
    ]

    history = model.fit(
        X_train,
        y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=EPOCHS,
        batch_size=FULL_BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, predictions)

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

    del model
    tf.keras.backend.clear_session()

    return accuracy, best_epoch


def run_pretrained_all_layers(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
):
    """
    Fine-tune the pre-trained CNN with all layers trainable, using
    all 672 real training samples.
    """
    print("\n" + "=" * 80)
    print("PRETRAINED CNN (ALL LAYERS FINE-TUNED)")
    print(f"Training samples = {len(y_train)}")
    print("=" * 80)

    results = []

    for seed in SEEDS:
        print(f"\nSeed {seed}")

        accuracy, best_epoch = pretrained_all_layers_one_seed(
            seed,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        )

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Best Epoch: {best_epoch}")

        results.append({
            "method": "Pretrained CNN (all layers)",
            "seed": seed,
            "accuracy": accuracy,
            "best_epoch": best_epoch
        })

    return results



# ============================================================
# 11. catch22 features
# ============================================================

def extract_catch22_features(
        X
):
    """
    Extract the 22 canonical catch22 features for each time series.
    """

    print(
        f"Extracting catch22 features "
        f"for {len(X)} samples..."
    )


    features = []


    for i, sample in enumerate(
        X
    ):

        # X shape:
        # (N, 100, 1)
        ts = sample[
            :,
            0
        ]


        feat = (
            pycatch22.catch22_all(
                ts
            )
        )


        features.append(
            feat[
                "values"
            ]
        )


        if (
            i + 1
        ) % 100 == 0:

            print(
                f"  {i + 1}/{len(X)}"
            )


    return np.array(
        features,
        dtype=float
    )


# ============================================================
# 12. Random Forest
# ============================================================

def run_random_forest(
        X_train,
        y_train,
        X_test,
        y_test
):
    """
    Train a Random Forest classifier on catch22 features extracted
    from all 672 real training samples and evaluate on the test set.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RF + catch22"
    )

    print(
        "=" * 80
    )


    # --------------------------------------------------------
    # Features only need to be extracted once
    # --------------------------------------------------------

    X_train_feat = (
        extract_catch22_features(
            X_train
        )
    )


    X_test_feat = (
        extract_catch22_features(
            X_test
        )
    )


    print(
        f"\nTrain features: "
        f"{X_train_feat.shape}"
    )

    print(
        f"Test features: "
        f"{X_test_feat.shape}"
    )


    results = []


    # --------------------------------------------------------
    # Five RF random seeds
    # --------------------------------------------------------

    for seed in SEEDS:

        rf = RandomForestClassifier(

            n_estimators=100,

            random_state=seed,

            n_jobs=-1
        )


        rf.fit(
            X_train_feat,
            y_train
        )


        y_pred = rf.predict(
            X_test_feat
        )


        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        print(
            f"Seed {seed}: "
            f"{accuracy:.4f}"
        )


        results.append(
            {
                "method":
                    "RF-catch22",

                "seed":
                    seed,

                "accuracy":
                    accuracy
            }
        )


    return results


# ============================================================
# 13. Build Table 12 (including Scratch CNN)
# ============================================================

def generate_table12(
        all_results
):
    """
    Build the final Table 12 summarizing mean +- SD accuracy for each
    method and its difference relative to the RF and full fine-tune
    baselines.
    """

    raw_df = pd.DataFrame(
        all_results
    )


    raw_path = os.path.join(
        OUTPUT_DIR,
        "Table12_seed_level_results.csv"
    )


    raw_df.to_csv(
        raw_path,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # Mean +- SD
    # ========================================================

    summary = (
        raw_df
        .groupby(
            "method"
        )
        .agg(
            N_seeds=(
                "seed",
                "nunique"
            ),

            accuracy_mean=(
                "accuracy",
                "mean"
            ),

            accuracy_sd=(
                "accuracy",
                "std"
            )
        )
        .reset_index()
    )


    # ========================================================
    # Fix the method order
    # ========================================================

    method_order = [

        "RF-catch22",

        "Zero-shot CNN",

        "Few-shot CNN",

        "Full-data fine-tuned CNN",

        "Pretrained CNN (all layers)",

        "Scratch CNN"
    ]


    summary[
        "method"
    ] = pd.Categorical(

        summary[
            "method"
        ],

        categories=method_order,

        ordered=True
    )


    summary = (
        summary
        .sort_values(
            "method"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Training data
    # ========================================================

    training_data_map = {

        "RF-catch22":
            "672 real training samples",

        "Zero-shot CNN":
            "0 real labels (synthetic only)",

        "Few-shot CNN":
            "224 real training samples",

        "Full-data fine-tuned CNN":
            "672 real training samples",

        "Pretrained CNN (all layers)":
            "672 real training samples",

        "Scratch CNN":
            "672 real training samples"
    }


    summary[
        "Training Data"
    ] = (
        summary[
            "method"
        ]
        .astype(str)
        .map(
            training_data_map
        )
    )


    # ========================================================
    # Accuracy %
    # ========================================================

    summary[
        "Accuracy"
    ] = summary.apply(

        lambda row:

        (
            f"{row['accuracy_mean'] * 100:.2f}% "
            f"+- "
            f"{row['accuracy_sd'] * 100:.2f}%"
        ),

        axis=1
    )


    # ========================================================
    # Relative to RF baseline
    # ========================================================

    rf_mean = float(
        summary.loc[
            summary[
                "method"
            ]
            ==
            "RF-catch22",
            "accuracy_mean"
        ].iloc[0]
    )


    # ========================================================
    # Relative to full-data fine-tune
    # ========================================================

    full_mean = float(
        summary.loc[
            summary[
                "method"
            ]
            ==
            "Full-data fine-tuned CNN",
            "accuracy_mean"
        ].iloc[0]
    )


    summary[
        "vs RF baseline"
    ] = (
        (
            summary[
                "accuracy_mean"
            ]
            -
            rf_mean
        )
        *
        100
    )


    summary[
        "vs Full-data fine-tune"
    ] = (
        (
            summary[
                "accuracy_mean"
            ]
            -
            full_mean
        )
        *
        100
    )


    def format_difference(x):

        if abs(
            x
        ) < 1e-10:

            return "--"

        return (
            f"{x:+.2f}%"
        )


    summary[
        "vs RF baseline"
    ] = (
        summary[
            "vs RF baseline"
        ]
        .apply(
            format_difference
        )
    )


    summary[
        "vs Full-data fine-tune"
    ] = (
        summary[
            "vs Full-data fine-tune"
        ]
        .apply(
            format_difference
        )
    )


    # ========================================================
    # Final manuscript version of Table 12
    # ========================================================

    table12 = summary[
        [
            "method",
            "Training Data",
            "Accuracy",
            "vs RF baseline",
            "vs Full-data fine-tune"
        ]
    ].copy()


    table12 = table12.rename(
        columns={
            "method":
                "Setting"
        }
    )


    # ========================================================
    # Save
    # ========================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        "Table12_FINAL.csv"
    )


    excel_path = os.path.join(
        OUTPUT_DIR,
        "Table12_FINAL.xlsx"
    )


    table12.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )


    table12.to_excel(
        excel_path,
        index=False
    )


    # ========================================================
    # Print
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FINAL TABLE 12"
    )

    print(
        "All methods evaluated on the SAME "
        "225 held-out test samples"
    )

    print(
        "=" * 120
    )


    print(
        table12.to_string(
            index=False
        )
    )


    return (
        table12,
        raw_df
    )


# ============================================================
# 14. Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "UNIFIED REAL-DATA BENCHMARK"
    )

    print(
        "All final evaluations use the same "
        "225 held-out test samples"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # Load fixed splits
    # ========================================================

    data = (
        load_all_splits()
    )


    X_train = (
        data[
            "X_train"
        ]
    )

    y_train = (
        data[
            "y_train"
        ]
    )

    train_ids = (
        data[
            "train_ids"
        ]
    )


    X_val = (
        data[
            "X_val"
        ]
    )

    y_val = (
        data[
            "y_val"
        ]
    )


    X_test = (
        data[
            "X_test"
        ]
    )

    y_test = (
        data[
            "y_test"
        ]
    )


    # ========================================================
    # Create fixed 224-sample few-shot subset
    # ========================================================

    X_few, y_few, few_ids = (
        create_fixed_few_shot_subset(

            X_train,

            y_train,

            train_ids
        )
    )


    all_results = []


    # ========================================================
    # RF
    # ========================================================

    rf_results = (
        run_random_forest(

            X_train,

            y_train,

            X_test,

            y_test
        )
    )


    all_results.extend(
        rf_results
    )


    # ========================================================
    # Zero-shot
    # ========================================================

    zero_results = (
        run_zero_shot(

            X_test,

            y_test
        )
    )


    all_results.extend(
        zero_results
    )


    # ========================================================
    # Few-shot
    # ========================================================

    few_results = (
        run_few_shot(

            X_few,

            y_few,

            X_val,

            y_val,

            X_test,

            y_test
        )
    )


    all_results.extend(
        few_results
    )


    # ========================================================
    # Full-data fine-tune
    # ========================================================

    full_results = (
        run_full_data_finetune(

            X_train,

            y_train,

            X_val,

            y_val,

            X_test,

            y_test
        )
    )


    all_results.extend(
        full_results
    )


    # ========================================================
    # Pretrained CNN with all layers trainable
    # Matched comparison against Scratch CNN
    # ========================================================

    pretrained_all_results = (
        run_pretrained_all_layers(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        )
    )

    all_results.extend(
        pretrained_all_results
    )


    # ========================================================
    # Scratch CNN: same architecture, random init, trained from scratch
    # ========================================================

    scratch_results = (
        run_scratch_cnn(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        )
    )

    all_results.extend(
        scratch_results
    )


    # ========================================================
    # Table 12
    # ========================================================

    table12, raw_df = (
        generate_table12(
            all_results
        )
    )


    # ========================================================
    # Final protocol check
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PROTOCOL CHECK"
    )

    print(
        "=" * 80
    )


    print(
        f"Train N      = "
        f"{len(y_train)}"
    )

    print(
        f"Validation N = "
        f"{len(y_val)}"
    )

    print(
        f"Test N       = "
        f"{len(y_test)}"
    )

    print(
        f"Few-shot N   = "
        f"{len(y_few)}"
    )


    print(
        "\nRF tested on the same 225 samples."
    )

    print(
        "Zero-shot CNN tested on the same 225 samples."
    )

    print(
        "Few-shot CNN tested on the same 225 samples."
    )

    print(
        "Full-data fine-tuned CNN tested on the same 225 samples."
    )

    print(
        "Pretrained CNN (all layers) tested on the same 225 samples."
    )

    print(
        "Scratch CNN tested on the same 225 samples."
    )

    print(
        "Matched comparison: pretrained and scratch CNNs used the same "
        "architecture, 672 real training samples, validation set, optimizer, "
        "learning rate, batch size, callbacks, and all-layer training."
    )

    print(
        "Validation samples were used only for "
        "configuration selection / early stopping."
    )

    print(
        "Test samples were never used for training "
        "or model selection."
    )

    print(
        "=" * 80
    )


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":

    main()