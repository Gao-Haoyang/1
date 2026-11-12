import numpy as np
import pandas as pd
import warnings
import os

from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import euclidean, cosine
from scipy.signal import find_peaks

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


warnings.filterwarnings("ignore")


# ============================================================
# 0. Global Configuration
# ============================================================

TARGET_LENGTH = 50
DISTANCE_TYPE = "cosine"

SEEDS = [0, 1, 2, 3, 42]


# ============================================================
# 1. Data directories for the five random seeds
#
# Modify these paths according to your actual directory structure
# ============================================================

SEED_DIRS = {

    0: "../../../data/seed_0/length_50",

    1: "../../../data/seed_1/length_50",

    2: "../../../data/seed_2/length_50",

    3: "../../../data/seed_3/length_50",

    42: "../../../data/seed_42/length_50",
}


# ============================================================
# 2. Nine varying-length noise scenarios
# ============================================================

NOISE_COMBS = [
    "multiplicative",
    "multiplicative_outlier",
    "additive_multiplicative"
]

STRENGTHS = [
    "low",
    "medium",
    "high"
]


DATASET_NAMES = [

    "multiplicative_low",
    "multiplicative_medium",
    "multiplicative_high",

    "multiplicative_outlier_low",
    "multiplicative_outlier_medium",
    "multiplicative_outlier_high",

    "additive_multiplicative_low",
    "additive_multiplicative_medium",
    "additive_multiplicative_high"
]


# ============================================================
# 3. Output directory
# ============================================================

OUTPUT_DIR = "./poly_classification"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. Polynomial / spline strategies
# ============================================================

STRATEGIES = [

    {
        "name": "ZeroPad_Deg2_OriginalTime",
        "method": "poly",
        "degree": 2,
        "pad": "zero",
        "norm_time": False
    },

    {
        "name": "Interp_Deg2_OriginalTime",
        "method": "poly",
        "degree": 2,
        "pad": "interp",
        "norm_time": False
    },

    {
        "name": "Interp_Deg2_NormTime",
        "method": "poly",
        "degree": 2,
        "pad": "interp",
        "norm_time": True
    },

    {
        "name": "Interp_Deg3_OriginalTime",
        "method": "poly",
        "degree": 3,
        "pad": "interp",
        "norm_time": False
    },

    {
        "name": "Interp_Deg3_NormTime",
        "method": "poly",
        "degree": 3,
        "pad": "interp",
        "norm_time": True
    },

    {
        "name": "Interp_Deg4_NormTime",
        "method": "poly",
        "degree": 4,
        "pad": "interp",
        "norm_time": True
    },

    {
        "name": "Interp_Deg5_NormTime",
        "method": "poly",
        "degree": 5,
        "pad": "interp",
        "norm_time": True
    },

    {
        "name": "Interp_Spline_NormTime",
        "method": "spline",
        "smooth": 0.5,
        "pad": "interp",
        "norm_time": True
    }
]


# ============================================================
# 5. Data preprocessing
# ============================================================

def load_and_preprocess(
        file_path,
        target_length=TARGET_LENGTH,
        pad_method="interp"
):
    """
    Load one synthetic dataset and unify all series to a fixed target length.

    pad_method:
        "interp"  = linear interpolation to the target length
        "zero"    = truncation (if longer) or zero-padding (if shorter)
    """

    df = pd.read_csv(file_path)

    df["unique_id"] = (
        df["class_label"].astype(str)
        + "_"
        + df["sample_id"].astype(str)
    )

    all_series = []
    labels = []
    original_lengths = []

    for unique_id, group in df.groupby("unique_id"):

        group = group.sort_values("time_step_id")

        values = group["value"].values
        label = group["class_label"].iloc[0]

        seq_len = len(values)

        original_lengths.append(seq_len)

        # ====================================================
        # Linear interpolation
        # ====================================================

        if pad_method == "interp":

            if seq_len == target_length:

                resampled = values.copy()

            else:

                from scipy import interpolate

                orig_time = np.arange(seq_len)

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

                resampled = f(target_time)

        # ====================================================
        # Zero padding
        # ====================================================

        else:

            if seq_len >= target_length:

                resampled = values[:target_length]

            else:

                resampled = np.concatenate(
                    [
                        values,
                        np.zeros(
                            target_length - seq_len
                        )
                    ]
                )

        all_series.append(resampled)
        labels.append(label)

    all_series = np.array(
        all_series,
        dtype=np.float32
    )

    labels = np.array(labels)

    original_lengths = np.array(
        original_lengths
    )

    return (
        all_series,
        labels,
        original_lengths
    )


# ============================================================
# 6. Polynomial + trend feature extraction
# ============================================================

def extract_poly_features(
        series,
        degree=2,
        norm_time=False
):
    """
    Extract polynomial coefficients and six trend descriptors for each series.

    norm_time:
        True  = rescale the time axis to [0, 1]
        False = use the original time index [0, 1, ..., n-1]
    """

    features = []

    n = TARGET_LENGTH

    for seq in series:

        # --------------------------------------------------
        # Time coordinate
        # --------------------------------------------------

        if norm_time:

            t = np.linspace(
                0,
                1,
                n
            )

        else:

            t = np.arange(
                n
            )

        # --------------------------------------------------
        # Polynomial coefficients
        # --------------------------------------------------

        coeffs = np.polyfit(
            t,
            seq,
            deg=degree
        )

        # np.polyfit returns coefficients in descending order
        # (highest degree first). Reverse to ascending order
        # (constant term first) for a consistent feature layout.
        coeffs = coeffs[::-1]

        # --------------------------------------------------
        # Fitted polynomial curve
        # --------------------------------------------------

        poly = np.poly1d(
            np.polyfit(
                t,
                seq,
                deg=degree
            )
        )

        y_fit = poly(t)

        # --------------------------------------------------
        # Trend features
        # --------------------------------------------------

        peaks, _ = find_peaks(
            y_fit
        )

        valleys, _ = find_peaks(
            -y_fit
        )

        diff = np.diff(
            y_fit
        )

        if len(diff) > 0:

            up_ratio = (
                np.sum(diff > 0)
                /
                len(diff)
            )

            down_ratio = (
                np.sum(diff < 0)
                /
                len(diff)
            )

            mean_abs_diff = np.mean(
                np.abs(diff)
            )

        else:

            up_ratio = 0.0
            down_ratio = 0.0
            mean_abs_diff = 0.0

        trend_feat = [

            len(peaks),

            len(valleys),

            up_ratio,

            down_ratio,

            np.std(y_fit),

            mean_abs_diff
        ]

        combined = np.hstack(
            [
                coeffs,
                trend_feat
            ]
        )

        features.append(
            combined
        )

    features = np.array(
        features,
        dtype=np.float32
    )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features


# ============================================================
# 7. Spline feature extraction
# ============================================================

def extract_spline_features(
        series,
        smooth=0.5,
        norm_time=False
):
    """
    Extract spline-based features: statistics of the fitted curve and its
    first/second derivatives, plus peak/valley counts and mean absolute
    fitting residual.
    """

    features = []

    n = TARGET_LENGTH

    for seq in series:

        if norm_time:

            t = np.linspace(
                0,
                1,
                n
            )

        else:

            t = np.arange(
                n
            )

        try:

            spl = UnivariateSpline(
                t,
                seq,
                s=smooth * len(seq),
                k=3
            )

            y_fit = spl(t)

            deriv1 = (
                spl.derivative(1)(t)
            )

            deriv2 = (
                spl.derivative(2)(t)
            )

            feat = [

                np.mean(y_fit),

                np.std(y_fit),

                np.mean(deriv1),

                np.std(deriv1),

                np.mean(deriv2),

                np.std(deriv2),

                np.max(y_fit)
                -
                np.min(y_fit),

                np.mean(
                    np.abs(
                        seq - y_fit
                    )
                )
            ]

            peaks, _ = find_peaks(
                y_fit
            )

            valleys, _ = find_peaks(
                -y_fit
            )

            feat.extend(
                [
                    len(peaks),
                    len(valleys)
                ]
            )

        except Exception:

            feat = [0.0] * 10

        features.append(
            feat
        )

    features = np.array(
        features,
        dtype=np.float32
    )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features


# ============================================================
# 8. Single-seed strategy evaluation
# ============================================================

def evaluate_strategy(
        series,
        labels,
        strategy,
        seed
):
    """
    Evaluate one strategy on one seed: extract features, perform a stratified
    80/20 train-test split using the seed, standardize features, compute
    class prototypes from the training set, and assign each test sample to
    the nearest prototype by cosine (or Euclidean) distance.
    """

    # ========================================================
    # Feature extraction
    # ========================================================

    if strategy["method"] == "poly":

        feat = extract_poly_features(

            series,

            degree=strategy["degree"],

            norm_time=strategy.get(
                "norm_time",
                False
            )
        )

    else:

        feat = extract_spline_features(

            series,

            smooth=strategy.get(
                "smooth",
                0.5
            ),

            norm_time=strategy.get(
                "norm_time",
                False
            )
        )

    # ========================================================
    # Train / Test split
    # ========================================================

    X_train, \
    X_test, \
    y_train, \
    y_test = train_test_split(

        feat,

        labels,

        test_size=0.20,

        random_state=seed,

        stratify=labels
    )

    # ========================================================
    # Standardization
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    # ========================================================
    # Class centers
    # ========================================================

    class_centers = {}

    for cls in np.unique(
        y_train
    ):

        cls_features = (
            X_train_scaled[
                y_train == cls
            ]
        )

        class_centers[cls] = np.mean(
            cls_features,
            axis=0
        )

    # ========================================================
    # Prediction
    # ========================================================

    y_pred = []

    for test_feat in X_test_scaled:

        distances = {}

        for cls, center in (
            class_centers.items()
        ):

            if DISTANCE_TYPE == "cosine":

                distances[cls] = cosine(
                    test_feat,
                    center
                )

            else:

                distances[cls] = euclidean(
                    test_feat,
                    center
                )

        pred_cls = min(
            distances,
            key=distances.get
        )

        y_pred.append(
            pred_cls
        )

    y_pred = np.array(
        y_pred
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    return accuracy


# ============================================================
# 9. Main
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "POLYNOMIAL STRATEGY COMPARISON"
    )

    print(
        "Independent synthetic dataset for each seed"
    )

    print(
        f"Seeds = {SEEDS}"
    )

    print(
        "=" * 120
    )

    # ========================================================
    # Check that the five seed directories exist
    # ========================================================

    print(
        "\nChecking seed directories..."
    )

    for seed in SEEDS:

        seed_dir = SEED_DIRS[seed]

        if os.path.isdir(seed_dir):

            print(
                f"Seed {seed}: "
                f"{seed_dir}"
            )

        else:

            print(
                f"Seed {seed}: "
                f"Directory not found:"
                f"\n   {seed_dir}"
            )

            raise FileNotFoundError(
                seed_dir
            )

    # ========================================================
    # Per-seed results
    # ========================================================

    all_seed_results = []

    # ========================================================
    # Summary results (mean +- SD)
    # ========================================================

    summary_results = []

    # ========================================================
    # Dataset loop
    # ========================================================

    for dataset_name in DATASET_NAMES:

        print(
            "\n"
            + "=" * 120
        )

        print(
            f"DATASET: {dataset_name}"
        )

        print(
            "=" * 120
        )

        # ====================================================
        # Strategy loop
        # ====================================================

        for strategy in STRATEGIES:

            strategy_name = (
                strategy["name"]
            )

            print(
                f"\nStrategy: "
                f"{strategy_name}"
            )

            seed_accuracies = []

            # =================================================
            # Seed loop
            #
            # For each seed:
            #   1. Load the synthetic dataset generated for this seed
            #   2. Perform a train/test split using the same seed
            # =================================================

            for seed in SEEDS:

                # ---------------------------------------------
                # Each seed has its own independently generated
                # synthetic dataset
                # ---------------------------------------------

                file_path = os.path.join(

                    SEED_DIRS[seed],

                    f"timeseries_{dataset_name}.csv"
                )

                if not os.path.isfile(
                    file_path
                ):

                    print(
                        f"\nFile not found:"
                        f"\n{file_path}"
                    )

                    raise FileNotFoundError(
                        file_path
                    )

                # ---------------------------------------------
                # Select interpolation or zero-padding
                # according to the current strategy
                # ---------------------------------------------

                pad_method = strategy.get(
                    "pad",
                    "interp"
                )

                series, labels, original_lengths = (
                    load_and_preprocess(

                        file_path,

                        pad_method=pad_method
                    )
                )

                # ---------------------------------------------
                # Evaluation
                # ---------------------------------------------

                accuracy = (
                    evaluate_strategy(

                        series,

                        labels,

                        strategy,

                        seed
                    )
                )

                seed_accuracies.append(
                    accuracy
                )

                all_seed_results.append(
                    {

                        "dataset":
                            dataset_name,

                        "strategy":
                            strategy_name,

                        "seed":
                            seed,

                        "file":
                            file_path,

                        "accuracy":
                            accuracy,

                        "accuracy_percent":
                            accuracy * 100
                    }
                )

                print(
                    f"  Seed {seed:<2} | "
                    f"Accuracy = "
                    f"{accuracy * 100:.2f}%"
                )

            # =================================================
            # Mean accuracy across seeds
            # =================================================

            mean_accuracy = np.mean(
                seed_accuracies
            )

            # =================================================
            # Population standard deviation (ddof=0)
            # =================================================

            sd_accuracy = np.std(
                seed_accuracies,
                ddof=0
            )

            print(
                "  "
                + "-" * 50
            )

            print(
                f"  Mean +- SD = "
                f"{mean_accuracy * 100:.2f}% "
                f"+- "
                f"{sd_accuracy * 100:.2f}%"
            )

            summary_results.append(
                {

                    "dataset":
                        dataset_name,

                    "strategy":
                        strategy_name,

                    "mean_accuracy":
                        mean_accuracy,

                    "sd_accuracy":
                        sd_accuracy,

                    "mean_percent":
                        mean_accuracy * 100,

                    "sd_percent":
                        sd_accuracy * 100,

                    "formatted":
                        (
                            f"{mean_accuracy * 100:.1f}% "
                            f"+- "
                            f"{sd_accuracy * 100:.1f}%"
                        )
                }
            )

    # ========================================================
    # 10. Save all per-seed results
    # ========================================================

    seed_df = pd.DataFrame(
        all_seed_results
    )

    seed_output = os.path.join(
        OUTPUT_DIR,
        "all_seed_results.csv"
    )

    seed_df.to_csv(
        seed_output,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 11. Save the long-format mean +- SD table
    # ========================================================

    summary_df = pd.DataFrame(
        summary_results
    )

    summary_output = os.path.join(
        OUTPUT_DIR,
        "strategy_mean_sd_results.csv"
    )

    summary_df.to_csv(
        summary_output,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 12. Build the final Table 4
    # ========================================================

    table4 = summary_df.pivot(
        index="dataset",
        columns="strategy",
        values="formatted"
    )

    # ========================================================
    # Fix the row order
    # ========================================================

    table4 = table4.reindex(
        DATASET_NAMES
    )

    # ========================================================
    # Columns required for Table 4
    # ========================================================

    table4_columns = [

        "Interp_Deg2_OriginalTime",

        "Interp_Deg2_NormTime",

        "Interp_Deg3_NormTime",

        "Interp_Deg4_NormTime",

        "Interp_Deg5_NormTime"
    ]

    table4_final = table4[
        table4_columns
    ].copy()

    # ========================================================
    # Rename to the column names used in the manuscript
    # ========================================================

    rename_map = {

        "Interp_Deg2_OriginalTime":
            "Deg2(raw time)",

        "Interp_Deg2_NormTime":
            "Deg2(norm time)",

        "Interp_Deg3_NormTime":
            "Deg3(norm)",

        "Interp_Deg4_NormTime":
            "Deg4(norm)",

        "Interp_Deg5_NormTime":
            "Deg5(norm)"
    }

    table4_final = (
        table4_final.rename(
            columns=rename_map
        )
    )

    # ========================================================
    # 13. Save the final Table 4
    # ========================================================

    table4_output = os.path.join(
        OUTPUT_DIR,
        "Table4_FINAL_Independent5Seeds.csv"
    )

    table4_final.to_csv(
        table4_output,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 14. Print the final Table 4
    # ========================================================

    print(
        "\n\n"
        + "=" * 140
    )

    print(
        "FINAL TABLE 4"
    )

    print(
        "Independent synthetic dataset "
        "for each random seed"
    )

    print(
        "Mean% +- SD across seeds "
        "[0, 1, 2, 3, 42]"
    )

    print(
        "=" * 140
    )

    print(
        table4_final.to_string()
    )

    # ========================================================
    # 15. Table 3 vs Table 4 check
    #
    # Only Deg2(raw time)
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "TABLE 3 vs TABLE 4 CHECK"
    )

    print(
        "Deg2(raw time)"
    )

    print(
        "=" * 120
    )

    for dataset_name in (
        table4_final.index
    ):

        value = table4_final.loc[
            dataset_name,
            "Deg2(raw time)"
        ]

        print(
            f"{dataset_name:<40}"
            f"{value}"
        )

    # ========================================================
    # 16. Current Table 3 Exp10-18
    #
    # Provided here for manual comparison
    # ========================================================

    TABLE3_REFERENCE = {

        "multiplicative_low":
            "68.6% ± 1.1%",

        "multiplicative_medium":
            "69.4% ± 1.4%",

        "multiplicative_high":
            "67.4% ± 0.7%",

        "multiplicative_outlier_low":
            "68.5% ± 1.1%",

        "multiplicative_outlier_medium":
            "66.7% ± 1.6%",

        "multiplicative_outlier_high":
            "64.5% ± 1.3%",

        "additive_multiplicative_low":
            "69.6% ± 2.4%",

        "additive_multiplicative_medium":
            "68.2% ± 1.0%",

        "additive_multiplicative_high":
            "66.0% ± 0.9%"
    }

    print(
        "\n"
        + "=" * 120
    )

    print(
        "REFERENCE: CURRENT TABLE 3"
    )

    print(
        "=" * 120
    )

    for dataset_name in DATASET_NAMES:

        print(
            f"{dataset_name:<40}"
            f"{TABLE3_REFERENCE[dataset_name]}"
        )

    # ========================================================
    # 17. Save the comparison table
    # ========================================================

    comparison_rows = []

    for dataset_name in DATASET_NAMES:

        comparison_rows.append(
            {

                "dataset":
                    dataset_name,

                "Table4_new":
                    table4_final.loc[
                        dataset_name,
                        "Deg2(raw time)"
                    ],

                "Table3_current":
                    TABLE3_REFERENCE[
                        dataset_name
                    ]
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_output = os.path.join(
        OUTPUT_DIR,
        "Table3_vs_Table4_Deg2_check.csv"
    )

    comparison_df.to_csv(
        comparison_output,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 18. Final output
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 120
    )

    print(
        f"\n1. All per-seed results:"
        f"\n{seed_output}"
    )

    print(
        f"\n2. Mean +- SD results:"
        f"\n{summary_output}"
    )

    print(
        f"\n3. Final Table 4:"
        f"\n{table4_output}"
    )

    print(
        f"\n4. Table 3 vs Table 4 check:"
        f"\n{comparison_output}"
    )

    print(
        "\nAnalysis completed."
    )


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":

    main()