import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from scipy import interpolate

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 0. Configuration
# ============================================================

# Real-world data with the fixed split column
REAL_DATA_PATH = "./LPD_with_split.csv"

# Output directory
OUTPUT_DIR = "./evaluate_test_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Best CNN configuration locked by the validation set
# ------------------------------------------------------------

BEST_CONFIG = "additive_multiplicative_high"


# ------------------------------------------------------------
# Five random seeds
# ------------------------------------------------------------

SEEDS = [0, 1, 2, 3, 42]


# ------------------------------------------------------------
# Model directories corresponding to the five seeds
# ------------------------------------------------------------

SEED_MODEL_DIRS = {

    0: "../models/seed_0/length_50_100/deeplearning",
    1: "../models/seed_1/length_50_100/deeplearning",
    2: "../models/seed_2/length_50_100/deeplearning",
    3: "../models/seed_3/length_50_100/deeplearning",
    42: "../models/seed_42/length_50_100/deeplearning"
}


# ------------------------------------------------------------
# Other parameters
# ------------------------------------------------------------

TARGET_LENGTH = 100

NUM_CLASSES = 5

CLASS_NAMES = [
    "Linear Increasing",
    "Linear Decreasing",
    "Single-Valley",
    "Single-Peak",
    "Multi-Peak"
]

SHORT_NAMES = [
    "LI",
    "LD",
    "SV",
    "SP",
    "MP"
]


# ------------------------------------------------------------
# Bootstrap settings
# ------------------------------------------------------------

N_BOOTSTRAP = 1000

BOOTSTRAP_SEED = 42


# ============================================================
# 1. Load the fixed TEST set
# ============================================================

def preprocess_test_data(
        file_path,
        target_length=TARGET_LENGTH
):
    """
    Load the fixed test split, interpolate each series to TARGET_LENGTH,
    and standardize each series independently.
    """

    print("=" * 80)
    print("Step 1: Loading held-out TEST set")
    print("=" * 80)

    df = pd.read_csv(
        file_path
    )

    print(
        f"Number of rows in the full per-timestep dataset: "
        f"{len(df)}"
    )

    print(
        f"Columns: "
        f"{df.columns.tolist()}"
    )


    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

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
                f"Missing required column: {col}"
            )


    # --------------------------------------------------------
    # Keep only the test split
    # --------------------------------------------------------

    df = df[
        df["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()


    if len(df) == 0:

        raise ValueError(
            "No samples with split == test were found."
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


    # --------------------------------------------------------
    # Sample-level loading
    # --------------------------------------------------------

    samples = []
    labels = []
    sample_ids = []


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

        seq_len = len(
            values
        )


        # ====================================================
        # Interpolate to TARGET_LENGTH points
        # ====================================================

        if seq_len == target_length:

            resampled = (
                values.copy()
            )

        else:

            orig_time = np.arange(
                seq_len
            )

            target_time = np.linspace(
                0,
                seq_len - 1,
                target_length
            )

            f = interpolate.interp1d(
                orig_time,
                values,
                kind="linear",
                fill_value="extrapolate"
            )

            resampled = f(
                target_time
            )


        # ====================================================
        # Standardize each time series independently
        # ====================================================

        scaler = StandardScaler()

        resampled_scaled = scaler.fit_transform(
            resampled.reshape(
                -1,
                1
            )
        ).flatten()


        samples.append(
            resampled_scaled
        )

        labels.append(
            label
        )

        sample_ids.append(
            unique_id
        )


    # --------------------------------------------------------
    # Convert to model input format
    # --------------------------------------------------------

    X = np.array(
        samples,
        dtype=np.float32
    )[:, :, np.newaxis]

    y = np.array(
        labels,
        dtype=int
    )


    print(
        f"\nNumber of test samples: "
        f"{len(y)}"
    )

    print(
        f"X shape: "
        f"{X.shape}"
    )


    print(
        "\nTest class distribution:"
    )

    distribution = (
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    for i in range(
        NUM_CLASSES
    ):

        print(
            f"{i} "
            f"{CLASS_NAMES[i]}: "
            f"{distribution.get(i, 0)}"
        )

    if len(y) != 225:

        print(
            "\nWARNING:"
        )

        print(
            f"Current test sample count = {len(y)}, "
            f"expected 225."
        )


    return (
        X,
        y,
        sample_ids
    )


# ============================================================
# 2. Locate the best CNN model for each seed
# ============================================================

def find_cnn_model_file(
        seed,
        best_config
):
    """
    Locate the best CNN model file for a given seed. First try the
    standard filename; if not found, fall back to fuzzy matching.
    """

    if seed not in SEED_MODEL_DIRS:

        return None


    model_dir = (
        SEED_MODEL_DIRS[
            seed
        ]
    )


    # --------------------------------------------------------
    # Prefer the standard filenames
    # --------------------------------------------------------

    possible_files = [

        os.path.join(
            model_dir,
            f"cnn_{best_config}.h5"
        ),

        os.path.join(
            model_dir,
            f"cnn_{best_config}.keras"
        )
    ]


    for path in possible_files:

        if os.path.isfile(
            path
        ):

            return path


    # --------------------------------------------------------
    # Fall back to fuzzy matching
    # --------------------------------------------------------

    search_patterns = [

        os.path.join(
            model_dir,
            f"*cnn*{best_config}*.h5"
        ),

        os.path.join(
            model_dir,
            f"*cnn*{best_config}*.keras"
        )
    ]


    candidates = []

    for pattern in search_patterns:

        candidates.extend(
            glob.glob(
                pattern
            )
        )


    candidates = list(
        dict.fromkeys(
            candidates
        )
    )


    if len(candidates) == 1:

        return candidates[0]


    elif len(candidates) > 1:

        print(
            f"\nWARNING: Seed {seed} "
            f"matched multiple CNN candidate models:"
        )

        for p in candidates:

            print(
                "  ",
                p
            )

        print(
            "Using the first match by default:"
        )

        print(
            candidates[0]
        )

        return candidates[0]


    return None


# ============================================================
# 3. Load the five seed models
# ============================================================

def load_seed_models():
    """
    Load the five seed models corresponding to BEST_CONFIG.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Step 2: Loading five seed CNN models"
    )

    print(
        "=" * 80
    )


    models = []

    model_paths = []


    for seed in SEEDS:

        model_path = (
            find_cnn_model_file(
                seed,
                BEST_CONFIG
            )
        )


        if model_path is None:

            raise FileNotFoundError(
                f"\nCannot find the CNN model for seed {seed}.\n"
                f"Configuration: {BEST_CONFIG}\n"
                f"Directory: "
                f"{SEED_MODEL_DIRS[seed]}"
            )


        print(
            f"\nSeed {seed}:"
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


        models.append(
            model
        )

        model_paths.append(
            model_path
        )


    print(
        f"\nLoaded {len(models)} models in total."
    )


    return (
        models,
        model_paths
    )


# ============================================================
# 4. Five-seed softmax aggregation
# ============================================================

def aggregate_predictions(
        models,
        X
):
    """
    Average the softmax probabilities of the five seed models and take
    the argmax to obtain the final prediction for each test sample.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Step 3: Five-seed softmax aggregation"
    )

    print(
        "=" * 80
    )


    probs_list = []


    for idx, model in enumerate(
        models
    ):

        seed = SEEDS[idx]

        probs = model.predict(
            X,
            verbose=0
        )

        probs_list.append(
            probs
        )

        print(
            f"Seed {seed} prediction completed."
        )


    # --------------------------------------------------------
    # Shape:
    # (5 seeds, N samples, 5 classes)
    # --------------------------------------------------------

    probs_array = np.array(
        probs_list
    )


    print(
        f"\nProbability array shape: "
        f"{probs_array.shape}"
    )


    # --------------------------------------------------------
    # Average softmax over the five seeds
    # --------------------------------------------------------

    avg_probs = np.mean(
        probs_array,
        axis=0
    )


    # --------------------------------------------------------
    # Single final prediction
    # --------------------------------------------------------

    y_pred = np.argmax(
        avg_probs,
        axis=1
    )


    print(
        f"Aggregated probability shape: "
        f"{avg_probs.shape}"
    )

    print(
        f"Final prediction shape: "
        f"{y_pred.shape}"
    )


    return (
        probs_array,
        avg_probs,
        y_pred
    )


# ============================================================
# 5. Stratified bootstrap CI
# ============================================================

def bootstrap_per_class_metrics(
        y_true,
        y_pred,
        n_bootstrap=N_BOOTSTRAP,
        random_state=BOOTSTRAP_SEED
):
    """
    Compute 95% percentile bootstrap CIs for per-class precision, recall,
    and F1 by resampling within each class while preserving class sizes.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Step 4: Stratified bootstrap "
        f"({n_bootstrap} resamples)"
    )

    print(
        "=" * 80
    )

    rng = np.random.default_rng(
        random_state
    )

    class_indices = {}

    for class_id in range(
        NUM_CLASSES
    ):

        indices = np.where(
            y_true == class_id
        )[0]

        class_indices[
            class_id
        ] = indices

        print(
            f"{CLASS_NAMES[class_id]}: "
            f"N = {len(indices)}"
        )

    # --------------------------------------------------------
    # Store bootstrap metrics
    # --------------------------------------------------------

    boot_precision = {
        i: []
        for i in range(
            NUM_CLASSES
        )
    }

    boot_recall = {
        i: []
        for i in range(
            NUM_CLASSES
        )
    }

    boot_f1 = {
        i: []
        for i in range(
            NUM_CLASSES
        )
    }

    # ========================================================
    # Stratified bootstrap
    # ========================================================

    for b in range(
        n_bootstrap
    ):

        bootstrap_indices = []

        # ----------------------------------------------------
        # Sample with replacement within each class.
        # Sample size equals the original class size.
        # ----------------------------------------------------

        for class_id in range(
            NUM_CLASSES
        ):

            original_indices = (
                class_indices[
                    class_id
                ]
            )

            sampled_indices = (
                rng.choice(
                    original_indices,
                    size=len(
                        original_indices
                    ),
                    replace=True
                )
            )

            bootstrap_indices.extend(
                sampled_indices
            )

        bootstrap_indices = np.array(
            bootstrap_indices,
            dtype=int
        )

        # ----------------------------------------------------
        # True and predicted labels for this bootstrap round
        # ----------------------------------------------------

        y_true_b = (
            y_true[
                bootstrap_indices
            ]
        )

        y_pred_b = (
            y_pred[
                bootstrap_indices
            ]
        )

        # ====================================================
        # Compute per-class metrics
        # ====================================================

        for class_id in range(
            NUM_CLASSES
        ):

            # Precision
            precision = (
                precision_score(
                    y_true_b,
                    y_pred_b,
                    labels=[
                        class_id
                    ],
                    average=None,
                    zero_division=0
                )[0]
            )

            # Recall
            recall = (
                recall_score(
                    y_true_b,
                    y_pred_b,
                    labels=[
                        class_id
                    ],
                    average=None,
                    zero_division=0
                )[0]
            )

            # F1
            f1 = (
                f1_score(
                    y_true_b,
                    y_pred_b,
                    labels=[
                        class_id
                    ],
                    average=None,
                    zero_division=0
                )[0]
            )

            boot_precision[
                class_id
            ].append(
                precision
            )

            boot_recall[
                class_id
            ].append(
                recall
            )

            boot_f1[
                class_id
            ].append(
                f1
            )

    # ========================================================
    # Compute percentile 95% CIs
    # ========================================================

    bootstrap_ci = {}

    for class_id in range(
        NUM_CLASSES
    ):

        bootstrap_ci[
            class_id
        ] = {

            "precision_lower":
                np.percentile(
                    boot_precision[
                        class_id
                    ],
                    2.5
                ),

            "precision_upper":
                np.percentile(
                    boot_precision[
                        class_id
                    ],
                    97.5
                ),

            "recall_lower":
                np.percentile(
                    boot_recall[
                        class_id
                    ],
                    2.5
                ),

            "recall_upper":
                np.percentile(
                    boot_recall[
                        class_id
                    ],
                    97.5
                ),

            "f1_lower":
                np.percentile(
                    boot_f1[
                        class_id
                    ],
                    2.5
                ),

            "f1_upper":
                np.percentile(
                    boot_f1[
                        class_id
                    ],
                    97.5
                )
        }

    print(
        "\nStratified bootstrap completed."
    )

    print(
        "Class counts were preserved "
        "in every bootstrap resample."
    )

    return bootstrap_ci


# ============================================================
# 6. Final evaluation
# ============================================================

def evaluate_final(
        y_true,
        y_pred,
        avg_probs,
        bootstrap_ci
):
    """
    Compute overall accuracy, balanced accuracy, macro-F1, AUC,
    confusion matrix, and per-class metrics with bootstrap CIs.
    """

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    # --------------------------------------------------------
    # Balanced Accuracy
    # --------------------------------------------------------

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )


    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------

    macro_f1 = (
        report[
            "macro avg"
        ][
            "f1-score"
        ]
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(
                NUM_CLASSES
            )
        )
    )


    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    try:

        y_onehot = (
            tf.keras.utils
            .to_categorical(
                y_true,
                num_classes=NUM_CLASSES
            )
        )

        auc = roc_auc_score(
            y_onehot,
            avg_probs,
            multi_class="ovr",
            average="macro"
        )

    except Exception as e:

        print(
            "\nWARNING: AUC calculation failed:"
        )

        print(
            e
        )

        auc = np.nan


    # --------------------------------------------------------
    # Per-class
    # --------------------------------------------------------

    rows = []


    for i, name in enumerate(
        CLASS_NAMES
    ):

        p = report[
            name
        ]


        rows.append(
            {
                "Class":
                    name,

                "N":
                    int(
                        p[
                            "support"
                        ]
                    ),

                "Precision":
                    p[
                        "precision"
                    ],

                "Precision_Lower":
                    bootstrap_ci[
                        i
                    ][
                        "precision_lower"
                    ],

                "Precision_Upper":
                    bootstrap_ci[
                        i
                    ][
                        "precision_upper"
                    ],

                "Recall":
                    p[
                        "recall"
                    ],

                "Recall_Lower":
                    bootstrap_ci[
                        i
                    ][
                        "recall_lower"
                    ],

                "Recall_Upper":
                    bootstrap_ci[
                        i
                    ][
                        "recall_upper"
                    ],

                "F1":
                    p[
                        "f1-score"
                    ],

                "F1_Lower":
                    bootstrap_ci[
                        i
                    ][
                        "f1_lower"
                    ],

                "F1_Upper":
                    bootstrap_ci[
                        i
                    ][
                        "f1_upper"
                    ]
            }
        )


    per_class_df = pd.DataFrame(
        rows
    )


    return {
        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1":
            macro_f1,

        "auc":
            auc,

        "confusion_matrix":
            cm,

        "per_class_df":
            per_class_df
    }


# ============================================================
# 7. Build Table 11
# ============================================================

def generate_table11(
        results,
        total_n
):
    """
    Build the manuscript version of Table 11 and save it as CSV and Excel.
    """

    df = (
        results[
            "per_class_df"
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Format CIs
    # --------------------------------------------------------

    df[
        "Precision (95% CI)"
    ] = df.apply(

        lambda row:

        (
            f"{row['Precision']:.3f} "
            f"[{row['Precision_Lower']:.3f}, "
            f"{row['Precision_Upper']:.3f}]"
        ),

        axis=1
    )


    df[
        "Recall (95% CI)"
    ] = df.apply(

        lambda row:

        (
            f"{row['Recall']:.3f} "
            f"[{row['Recall_Lower']:.3f}, "
            f"{row['Recall_Upper']:.3f}]"
        ),

        axis=1
    )


    df[
        "F1-score (95% CI)"
    ] = df.apply(

        lambda row:

        (
            f"{row['F1']:.3f} "
            f"[{row['F1_Lower']:.3f}, "
            f"{row['F1_Upper']:.3f}]"
        ),

        axis=1
    )


    table11 = df[
        [
            "Class",
            "N",
            "Precision (95% CI)",
            "Recall (95% CI)",
            "F1-score (95% CI)"
        ]
    ].copy()


    # --------------------------------------------------------
    # Save Table 11
    # --------------------------------------------------------

    table11_csv = os.path.join(
        OUTPUT_DIR,
        "Table11_FINAL_test.csv"
    )

    table11_excel = os.path.join(
        OUTPUT_DIR,
        "Table11_FINAL_test.xlsx"
    )


    table11.to_csv(
        table11_csv,
        index=False,
        encoding="utf-8-sig"
    )

    table11.to_excel(
        table11_excel,
        index=False
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FINAL TABLE 11"
    )

    print(
        "Held-out Test Set"
    )

    print(
        "=" * 120
    )


    print(
        table11.to_string(
            index=False
        )
    )


    print(
        "\nOverall metrics"
    )

    print(
        "-" * 60
    )

    print(
        f"N = "
        f"{total_n}"
    )

    print(
        f"Overall Accuracy = "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy = "
        f"{results['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro-F1 = "
        f"{results['macro_f1']:.4f}"
    )

    print(
        f"AUC = "
        f"{results['auc']:.4f}"
    )


    # --------------------------------------------------------
    # Save overall metrics separately
    # --------------------------------------------------------

    overall_df = pd.DataFrame(
        [
            {
                "N":
                    total_n,

                "Overall Accuracy":
                    results[
                        "accuracy"
                    ],

                "Balanced Accuracy":
                    results[
                        "balanced_accuracy"
                    ],

                "Macro-F1":
                    results[
                        "macro_f1"
                    ],

                "AUC":
                    results[
                        "auc"
                    ]
            }
        ]
    )


    overall_path = os.path.join(
        OUTPUT_DIR,
        "Table11_overall_metrics.csv"
    )


    overall_df.to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig"
    )


    return (
        table11,
        table11_csv,
        table11_excel,
        overall_path
    )


# ============================================================
# 8. Build Figure 8
# ============================================================

def generate_figure8(
        cm
):
    """
    Generate the row-normalized confusion matrix (Figure 8) and save it
    as PNG, PDF, and CSV.
    """

    # --------------------------------------------------------
    # Row normalization
    # --------------------------------------------------------

    row_sums = (
        cm.sum(
            axis=1,
            keepdims=True
        )
    )


    cm_normalized = (
        cm.astype(float)
        /
        row_sums
        *
        100
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(7.5, 6.5)
    )


    ax = sns.heatmap(

        cm_normalized,

        annot=True,

        fmt=".1f",

        cmap="Blues",

        xticklabels=SHORT_NAMES,

        yticklabels=SHORT_NAMES,

        vmin=0,

        vmax=100,

        cbar_kws={
            "label":
                "Normalized Classification Rate (%)"
        },

        annot_kws={
            "fontsize":
                12
        }
    )


    ax.set_xlabel(
        "Predicted Label",
        fontsize=13
    )

    ax.set_ylabel(
        "True Label",
        fontsize=13
    )


    ax.tick_params(
        axis="x",
        labelsize=11
    )

    ax.tick_params(
        axis="y",
        labelsize=11
    )


    plt.tight_layout()


    png_path = os.path.join(
        OUTPUT_DIR,
        "Figure8_FINAL_test.png"
    )

    pdf_path = os.path.join(
        OUTPUT_DIR,
        "Figure8_FINAL_test.pdf"
    )


    plt.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight"
    )


    plt.savefig(
        pdf_path,
        bbox_inches="tight"
    )


    plt.close()


    # --------------------------------------------------------
    # Report the normalized matrix
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Figure 8 normalized confusion matrix (%)"
    )

    print(
        "=" * 80
    )


    cm_df = pd.DataFrame(

        cm_normalized.round(
            1
        ),

        index=CLASS_NAMES,

        columns=CLASS_NAMES
    )


    print(
        cm_df.to_string()
    )


    cm_csv = os.path.join(
        OUTPUT_DIR,
        "Figure8_normalized_confusion_matrix.csv"
    )


    cm_df.to_csv(
        cm_csv,
        encoding="utf-8-sig"
    )


    print(
        f"\nFigure 8 saved:"
    )

    print(
        png_path
    )

    print(
        pdf_path
    )


    return (
        cm_normalized,
        png_path,
        pdf_path,
        cm_csv
    )


# ============================================================
# 9. Save final predictions
# ============================================================

def save_final_predictions(
        sample_ids,
        y_true,
        y_pred,
        avg_probs,
        probs_array
):
    """
    Save the final ensemble predictions and per-seed predictions to CSV.
    """

    output_df = pd.DataFrame(
        {
            "sample_id":
                sample_ids,

            "true_label":
                y_true,

            "final_pred":
                y_pred,

            "avg_prob_LI":
                avg_probs[
                    :,
                    0
                ],

            "avg_prob_LD":
                avg_probs[
                    :,
                    1
                ],

            "avg_prob_SV":
                avg_probs[
                    :,
                    2
                ],

            "avg_prob_SP":
                avg_probs[
                    :,
                    3
                ],

            "avg_prob_MP":
                avg_probs[
                    :,
                    4
                ]
        }
    )


    # --------------------------------------------------------
    # Also save per-seed predicted classes
    # --------------------------------------------------------

    for idx, seed in enumerate(
        SEEDS
    ):

        output_df[
            f"pred_seed_{seed}"
        ] = np.argmax(
            probs_array[
                idx
            ],
            axis=1
        )


    output_path = os.path.join(
        OUTPUT_DIR,
        "final_test_predictions.csv"
    )


    output_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        f"\nFinal predictions saved:"
    )

    print(
        output_path
    )


    return output_path


# ============================================================
# 10. Check Table 11 Recall vs Figure 8 diagonal
# ============================================================

def consistency_check(
        results,
        cm_normalized
):
    """
    Verify that Table 11 recalls exactly match the diagonal of the
    normalized confusion matrix in Figure 8.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CONSISTENCY CHECK"
    )

    print(
        "Table 11 Recall vs Figure 8 diagonal"
    )

    print(
        "=" * 80
    )


    per_class_df = (
        results[
            "per_class_df"
        ]
    )


    all_ok = True


    for i, name in enumerate(
        CLASS_NAMES
    ):

        recall_table = (
            per_class_df
            .loc[
                per_class_df[
                    "Class"
                ]
                == name,
                "Recall"
            ]
            .iloc[0]
            *
            100
        )


        diagonal_cm = (
            cm_normalized[
                i,
                i
            ]
        )


        diff = abs(
            recall_table
            -
            diagonal_cm
        )


        print(
            f"{name:<20} "
            f"Table11 Recall = "
            f"{recall_table:.1f}%   "
            f"Figure8 diagonal = "
            f"{diagonal_cm:.1f}%   "
            f"Difference = "
            f"{diff:.6f}"
        )


        if diff > 1e-8:

            all_ok = False


    if all_ok:

        print(
            "\nTable 11 Recall and "
            "Figure 8 diagonal are fully consistent."
        )

    else:

        print(
            "\nWARNING: "
            "Table 11 and Figure 8 are inconsistent."
        )


# ============================================================
# 11. Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "FINAL FIVE-SEED AGGREGATED TEST EVALUATION"
    )

    print(
        f"Best CNN configuration: "
        f"{BEST_CONFIG}"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # 1. Test set
    # ========================================================

    X_test, y_true, sample_ids = (
        preprocess_test_data(
            REAL_DATA_PATH
        )
    )


    # ========================================================
    # 2. Load five models
    # ========================================================

    models, model_paths = (
        load_seed_models()
    )


    # ========================================================
    # 3. Softmax aggregation
    # ========================================================

    probs_array, avg_probs, y_pred = (
        aggregate_predictions(
            models,
            X_test
        )
    )


    # ========================================================
    # 4. Bootstrap CI
    # ========================================================

    bootstrap_ci = (
        bootstrap_per_class_metrics(
            y_true,
            y_pred
        )
    )


    # ========================================================
    # 5. Final evaluation
    # ========================================================

    results = evaluate_final(
        y_true,
        y_pred,
        avg_probs,
        bootstrap_ci
    )


    # ========================================================
    # 6. Table 11
    # ========================================================

    table11, \
    table11_csv, \
    table11_excel, \
    overall_path = (
        generate_table11(
            results,
            total_n=len(
                y_true
            )
        )
    )


    # ========================================================
    # 7. Figure 8
    # ========================================================

    cm_normalized, \
    fig_png, \
    fig_pdf, \
    cm_csv = (
        generate_figure8(
            results[
                "confusion_matrix"
            ]
        )
    )


    # ========================================================
    # 8. Save predictions
    # ========================================================

    pred_path = (
        save_final_predictions(
            sample_ids,
            y_true,
            y_pred,
            avg_probs,
            probs_array
        )
    )


    # ========================================================
    # 9. Save the raw confusion matrix
    # ========================================================

    raw_cm_path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix_raw.csv"
    )


    raw_cm_df = pd.DataFrame(

        results[
            "confusion_matrix"
        ],

        index=CLASS_NAMES,

        columns=CLASS_NAMES
    )


    raw_cm_df.to_csv(
        raw_cm_path,
        encoding="utf-8-sig"
    )


    # ========================================================
    # 10. Consistency check
    # ========================================================

    consistency_check(
        results,
        cm_normalized
    )


    # ========================================================
    # 11. Final summary
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 80
    )


    print(
        f"Test N = "
        f"{len(y_true)}"
    )


    print(
        f"Overall Accuracy = "
        f"{results['accuracy'] * 100:.2f}%"
    )


    print(
        f"Balanced Accuracy = "
        f"{results['balanced_accuracy']:.4f}"
    )


    print(
        f"Macro-F1 = "
        f"{results['macro_f1']:.4f}"
    )


    print(
        f"AUC = "
        f"{results['auc']:.4f}"
    )


    print(
        "\n"
        + "=" * 80
    )


    print(
        "Only held-out TEST data were evaluated."
    )


    print(
        "Test data were not used for "
        "configuration selection."
    )


    print(
        "Five seed softmax probabilities "
        "were averaged."
    )


    print(
        "Table 11 and Figure 8 use "
        "the exact same final predictions."
    )


    print(
        f"Bootstrap CI: "
        f"{N_BOOTSTRAP} resamples."
    )


    print(
        "=" * 80
    )


    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )


    print(
        "\nKey files:"
    )

    print(
        table11_excel
    )

    print(
        fig_png
    )

    print(
        pred_path
    )

    print(
        raw_cm_path
    )


    # ========================================================
    # 12. Clean up models
    # ========================================================

    for model in models:

        del model


    tf.keras.backend.clear_session()


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":

    main()