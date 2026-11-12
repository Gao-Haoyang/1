<<<<<<< HEAD
import numpy as np
import pandas as pd
import random
import os
import sys  # For real-time progress bar updates

# ===================== Global Configuration =====================
NUM_CLASSES = 5
SAMPLES_PER_CLASS = 1000      # Number of samples per class
LEN_MIN = 50
LEN_MAX = 100
SEED = 2
np.random.seed(SEED)
random.seed(SEED)

OUTPUT_DIR = "../../data/seed_2/length_50_100"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== Noise Combination and Strength Definitions =====================
# BASE_CONFIGS is defined first so that TOTAL_DATASETS can be computed correctly
BASE_CONFIGS = {
    "multiplicative": {
        'multiplicative': {}
    },
    "multiplicative_outlier": {
        'multiplicative': {},
        'outlier': {}
    },
    "additive_multiplicative": {
        'additive': {},
        'multiplicative': {}
    }
}

# Strength parameter definitions (low / medium / high)

STRENGTH_PARAMS = {
    "low": {
        'additive': {'std_ratio': 0.01},
        'ar': {'rho': 0.5, 'std_ratio': 0.01},
        'multiplicative': {'log_std': 0.05},
        'outlier': {'prob': 0.005}
    },
    "medium": {
        'additive': {'std_ratio': 0.04},
        'ar': {'rho': 0.7, 'std_ratio': 0.04},
        'multiplicative': {'log_std': 0.10},
        'outlier': {'prob': 0.01}
    },
    "high": {
        'additive': {'std_ratio': 0.10},
        'ar': {'rho': 0.9, 'std_ratio': 0.10},
        'multiplicative': {'log_std': 0.20},
        'outlier': {'prob': 0.02}
    }
}

# Total number of datasets to generate: 3 noise compositions x 3 intensities
TOTAL_DATASETS = len(BASE_CONFIGS) * 3
CURRENT_DATASET = 0  # Index of the dataset currently being generated


# ===================== Base Curve Generation =====================
def generate_base_curve(class_id, length):
    """
    Generate a clean (noise-free) trajectory for a given shape class.

    class_id:
        0 = linear increasing (three subtypes)
        1 = linear decreasing (three subtypes)
        2 = single-valley (asymmetric Gaussian)
        3 = single-peak (asymmetric Gaussian)
        4 = multi-peak (3-5 peaks)
    """
    t = np.arange(length, dtype=np.float64)
    y = np.zeros(length, dtype=np.float64)

    if class_id == 0:  # Linear increasing
        rise_type = random.choice([0, 1, 2])
        if rise_type == 0:
            # Straight-line increase: y = k*t + b
            k = random.uniform(0.005, 0.03)
            b = random.uniform(1, 10)
            y = k * t + b
        elif rise_type == 1:
            # Slow-to-fast increase: y = a*(t - t0)^2 + 1.0
            # Vertex position t0 placed in the first 5-20% of the series
            t0 = length * random.uniform(0.05, 0.2)
            a = random.uniform(0.0005, 0.002)
            y = a * (t - t0) ** 2 + 1.0
        elif rise_type == 2:
            # Fast-to-slow increase: y = -a*(t - t0)^2 + (a*t0^2 + 1.0)
            # Vertex position t0 placed in the last 80-95% of the series
            t0 = length * random.uniform(0.8, 0.95)
            a = random.uniform(0.0005, 0.002)
            y = -a * (t - t0) ** 2 + (a * t0 ** 2 + 1.0)

    elif class_id == 1:  # Linear decreasing
        fall_type = random.choice([0, 1, 2])
        if fall_type == 0:
            # Straight-line decrease: y = -k*t + b
            k = random.uniform(0.005, 0.03)
            b = k * (length - 1) + random.uniform(1, 10)
            y = -k * t + b
        elif fall_type == 1:
            # Fast-to-slow decrease: y = a*(t - t0)^2 + (1.0 - a*t0^2)
            # Vertex position t0 placed in the last 80-95% of the series
            t0 = length * random.uniform(0.8, 0.95)
            a = random.uniform(0.0005, 0.002)
            y = a * (t - t0) ** 2 + (1.0 - a * t0 ** 2)
        elif fall_type == 2:
            # Slow-to-fast decrease: y = -a*(t - t0)^2 + (1.0 + a*t0^2)
            # Vertex position t0 placed in the first 5-20% of the series
            t0 = length * random.uniform(0.05, 0.2)
            a = random.uniform(0.0005, 0.002)
            y = -a * (t - t0) ** 2 + (1.0 + a * t0 ** 2)

    elif class_id == 2:  # Single-valley
        t0 = length * random.uniform(0.2, 0.8)        # Valley position
        sigma_left = random.uniform(5, 15)             # Left width (descent)
        sigma_right = random.uniform(5, 15)            # Right width (ascent)
        valley_depth = random.uniform(3, 10)           # Valley depth (delta)
        baseline = random.uniform(5, 15)               # Baseline level
        y = np.zeros(length)
        for i, ti in enumerate(t):
            if ti < t0:
                # Left half: descending into the valley
                y[i] = baseline - valley_depth * np.exp(-(ti - t0) ** 2 / (2 * sigma_left ** 2))
            else:
                # Right half: ascending out of the valley
                y[i] = baseline - valley_depth * np.exp(-(ti - t0) ** 2 / (2 * sigma_right ** 2))

    elif class_id == 3:  # Single-peak
        t0 = length * random.uniform(0.2, 0.8)        # Peak position (left- or right-skewed)
        sigma_left = random.uniform(5, 15)             # Left width (ascent, larger = gentler)
        sigma_right = random.uniform(5, 15)            # Right width (descent, larger = gentler)
        peak_height = random.uniform(3, 10)            # Peak height (h)
        baseline = random.uniform(0, 1)                # Baseline level
        y = np.zeros(length)
        for i, ti in enumerate(t):
            if ti < t0:
                # Left half: ascending to the peak
                y[i] = baseline + peak_height * np.exp(-(ti - t0) ** 2 / (2 * sigma_left ** 2))
            else:
                # Right half: descending from the peak
                y[i] = baseline + peak_height * np.exp(-(ti - t0) ** 2 / (2 * sigma_right ** 2))

    else:  # Multi-peak (3-5 peaks with controlled spacing)
        n_peaks = random.randint(3, 5)
        # Peak width scales with sequence length (5%-15% of total length)
        min_width = length * 0.05
        max_width = length * 0.15
        baseline = random.uniform(0, 2)
        y = np.full_like(t, baseline)

        # Peak centres are sampled within the central 60% of each equally partitioned segment
        centers = []
        segment_len = length / n_peaks
        for i in range(n_peaks):
            low = i * segment_len + segment_len * 0.2
            high = i * segment_len + segment_len * 0.8
            center = random.uniform(low, high)
            centers.append(center)

        # Amplitudes: first two peaks (base_amp * 0.8-1.4), remaining peaks (base_amp * 0.5-1.2)
        base_amp = random.uniform(2, 5)
        amplitudes = []
        for i in range(n_peaks):
            if i < 2:
                amp = base_amp * random.uniform(0.8, 1.4)
            else:
                amp = base_amp * random.uniform(0.5, 1.2)
            amplitudes.append(amp)

        for center, amp in zip(centers, amplitudes):
            width = random.uniform(min_width, max_width)
            y += amp * np.exp(-((t - center) ** 2) / (2 * width ** 2))

    return y


# ===================== Noise Generation Functions =====================
def ar1_noise(length, rho, std):
    """Generate AR(1) temporally correlated noise with autocorrelation rho."""
    noise = np.zeros(length)
    noise[0] = np.random.normal(0, std)
    for i in range(1, length):
        noise[i] = rho * noise[i - 1] + np.random.normal(0, std * np.sqrt(1 - rho ** 2))
    return noise


def multiplicative_noise(signal, log_std):
    """Multiplicative lognormal noise: y_noisy = y * epsilon, epsilon ~ Lognormal(0, log_std^2)."""
    mult = np.random.lognormal(0, log_std, size=len(signal))
    return signal * mult


def additive_noise(signal, std_ratio):
    """Additive Gaussian noise with standard deviation = std_ratio * std(signal)."""
    std = np.std(signal) * std_ratio
    return np.random.normal(0, std, size=len(signal))


def outlier_noise(signal, prob, factor_range=(2.0, 4.0)):
    """Random multiplicative spikes: with probability prob, a point is multiplied by U(2, 4)."""
    out = signal.copy()
    mask = np.random.rand(len(signal)) < prob
    n_out = np.sum(mask)
    if n_out > 0:
        factors = np.random.uniform(factor_range[0], factor_range[1], n_out)
        out[mask] *= factors
    return out


# ===================== Noise Combination Application =====================
def apply_noise_combination(signal, config):
    """
    Apply a sequence of noise operations to a clean signal.
    Supported keys: 'multiplicative', 'additive', 'ar', 'outlier'.
    """
    y = signal.copy()

    if config.get('multiplicative') is not None:
        mult_log_std = config['multiplicative']['log_std']
        y = multiplicative_noise(y, mult_log_std)

    add_noise_arr = None
    ar_noise_arr = None

    if config.get('additive') is not None:
        add_std_ratio = config['additive']['std_ratio']
        add_noise_arr = additive_noise(y, add_std_ratio)

    if config.get('ar') is not None:
        rho = config['ar']['rho']
        ar_std_ratio = config['ar']['std_ratio']
        ar_noise_arr = ar1_noise(len(y), rho, np.std(y) * ar_std_ratio)

    if add_noise_arr is not None or ar_noise_arr is not None:
        total_add = np.zeros_like(y)
        if add_noise_arr is not None:
            total_add += add_noise_arr
        if ar_noise_arr is not None:
            total_add += ar_noise_arr
        y = y + total_add

    if config.get('outlier') is not None:
        prob = config['outlier']['prob']
        y = outlier_noise(y, prob)

    return y


# ===================== Dataset Generation (with progress display) =====================
def generate_dataset(base_config, strength, config_name, output_dir):
    """
    Generate one synthetic dataset (5,000 time series with variable length
    drawn uniformly from [LEN_MIN, LEN_MAX]) for a given noise composition
    (base_config) and noise intensity (strength).
    """
    global CURRENT_DATASET
    CURRENT_DATASET += 1

    # Fill in the strength-specific parameters for each noise type
    config = {}
    for noise_type in base_config:
        config[noise_type] = STRENGTH_PARAMS[strength][noise_type].copy()

    full_name = f"{config_name}_{strength}"
    print(f"\n[{CURRENT_DATASET}/{TOTAL_DATASETS}] Generating {full_name} ...")

    all_vals, all_cls, all_sids, all_tids = [], [], [], []
    sample_id = 0
    total_samples = NUM_CLASSES * SAMPLES_PER_CLASS   # Total samples per dataset

    for class_id in range(NUM_CLASSES):
        for sample_idx in range(SAMPLES_PER_CLASS):
            # Overall progress within the current dataset
            current_sample = class_id * SAMPLES_PER_CLASS + sample_idx + 1
            progress = (current_sample / total_samples) * 100

            # Real-time progress update (overwrite current line)
            sys.stdout.write(f"\r   Progress: {current_sample}/{total_samples} ({progress:.1f}%)")
            sys.stdout.flush()

            # Generate one sample: variable length drawn from [LEN_MIN, LEN_MAX]
            length = random.randint(LEN_MIN, LEN_MAX)
            base = generate_base_curve(class_id, length)
            noisy = apply_noise_combination(base, config)

            for t in range(length):
                all_vals.append(noisy[t])
                all_cls.append(class_id)
                all_sids.append(sample_id)
                all_tids.append(t)
            sample_id += 1

    # Save the dataset
    df = pd.DataFrame({
        "class_label": all_cls,
        "sample_id": all_sids,
        "time_step_id": all_tids,
        "value": all_vals
    })
    save_path = os.path.join(output_dir, f"timeseries_{full_name}.csv")
    df.to_csv(save_path, index=False)
    print(f"\n   Saved to {save_path}")
    print(f"   Overall progress: {(CURRENT_DATASET / TOTAL_DATASETS) * 100:.1f}%")


# ===================== Main =====================
def main():
    print("=" * 50)
    print("Starting synthetic time series dataset generation")
    print(f"Total tasks: {TOTAL_DATASETS} datasets (3 noise compositions x 3 intensities)")
    print(f"Each dataset: {NUM_CLASSES} classes x {SAMPLES_PER_CLASS} samples/class")
    print("=" * 50)

    for config_name, base_config in BASE_CONFIGS.items():
        for strength in ['low', 'medium', 'high']:
            generate_dataset(base_config, strength, config_name, OUTPUT_DIR)

    # Write metadata describing the noise parameters used for each dataset
    metadata = []
    for config_name in BASE_CONFIGS.keys():
        for strength in ['low', 'medium', 'high']:
            row = {'dataset': f"{config_name}_{strength}", 'strength': strength}
            params = STRENGTH_PARAMS[strength]
            for noise_type in ['additive', 'ar', 'multiplicative', 'outlier']:
                if noise_type in params:
                    for k, v in params[noise_type].items():
                        row[f"{noise_type}_{k}"] = v
                else:
                    row[f"{noise_type}_present"] = False
            metadata.append(row)

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(OUTPUT_DIR, "strength_metadata.csv"), index=False)

    print(f"\nAll datasets generated successfully.")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Total files: {TOTAL_DATASETS} datasets + 1 metadata file")


if __name__ == "__main__":
=======
import numpy as np
import pandas as pd
import random
import os
import sys  # For real-time progress bar updates

# ===================== Global Configuration =====================
NUM_CLASSES = 5
SAMPLES_PER_CLASS = 1000      # Number of samples per class
LEN_MIN = 50
LEN_MAX = 100
SEED = 2
np.random.seed(SEED)
random.seed(SEED)

OUTPUT_DIR = "../../data/seed_2/length_50_100"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== Noise Combination and Strength Definitions =====================
# BASE_CONFIGS is defined first so that TOTAL_DATASETS can be computed correctly
BASE_CONFIGS = {
    "multiplicative": {
        'multiplicative': {}
    },
    "multiplicative_outlier": {
        'multiplicative': {},
        'outlier': {}
    },
    "additive_multiplicative": {
        'additive': {},
        'multiplicative': {}
    }
}

# Strength parameter definitions (low / medium / high)

STRENGTH_PARAMS = {
    "low": {
        'additive': {'std_ratio': 0.01},
        'ar': {'rho': 0.5, 'std_ratio': 0.01},
        'multiplicative': {'log_std': 0.05},
        'outlier': {'prob': 0.005}
    },
    "medium": {
        'additive': {'std_ratio': 0.04},
        'ar': {'rho': 0.7, 'std_ratio': 0.04},
        'multiplicative': {'log_std': 0.10},
        'outlier': {'prob': 0.01}
    },
    "high": {
        'additive': {'std_ratio': 0.10},
        'ar': {'rho': 0.9, 'std_ratio': 0.10},
        'multiplicative': {'log_std': 0.20},
        'outlier': {'prob': 0.02}
    }
}

# Total number of datasets to generate: 3 noise compositions x 3 intensities
TOTAL_DATASETS = len(BASE_CONFIGS) * 3
CURRENT_DATASET = 0  # Index of the dataset currently being generated


# ===================== Base Curve Generation =====================
def generate_base_curve(class_id, length):
    """
    Generate a clean (noise-free) trajectory for a given shape class.

    class_id:
        0 = linear increasing (three subtypes)
        1 = linear decreasing (three subtypes)
        2 = single-valley (asymmetric Gaussian)
        3 = single-peak (asymmetric Gaussian)
        4 = multi-peak (3-5 peaks)
    """
    t = np.arange(length, dtype=np.float64)
    y = np.zeros(length, dtype=np.float64)

    if class_id == 0:  # Linear increasing
        rise_type = random.choice([0, 1, 2])
        if rise_type == 0:
            # Straight-line increase: y = k*t + b
            k = random.uniform(0.005, 0.03)
            b = random.uniform(1, 10)
            y = k * t + b
        elif rise_type == 1:
            # Slow-to-fast increase: y = a*(t - t0)^2 + 1.0
            # Vertex position t0 placed in the first 5-20% of the series
            t0 = length * random.uniform(0.05, 0.2)
            a = random.uniform(0.0005, 0.002)
            y = a * (t - t0) ** 2 + 1.0
        elif rise_type == 2:
            # Fast-to-slow increase: y = -a*(t - t0)^2 + (a*t0^2 + 1.0)
            # Vertex position t0 placed in the last 80-95% of the series
            t0 = length * random.uniform(0.8, 0.95)
            a = random.uniform(0.0005, 0.002)
            y = -a * (t - t0) ** 2 + (a * t0 ** 2 + 1.0)

    elif class_id == 1:  # Linear decreasing
        fall_type = random.choice([0, 1, 2])
        if fall_type == 0:
            # Straight-line decrease: y = -k*t + b
            k = random.uniform(0.005, 0.03)
            b = k * (length - 1) + random.uniform(1, 10)
            y = -k * t + b
        elif fall_type == 1:
            # Fast-to-slow decrease: y = a*(t - t0)^2 + (1.0 - a*t0^2)
            # Vertex position t0 placed in the last 80-95% of the series
            t0 = length * random.uniform(0.8, 0.95)
            a = random.uniform(0.0005, 0.002)
            y = a * (t - t0) ** 2 + (1.0 - a * t0 ** 2)
        elif fall_type == 2:
            # Slow-to-fast decrease: y = -a*(t - t0)^2 + (1.0 + a*t0^2)
            # Vertex position t0 placed in the first 5-20% of the series
            t0 = length * random.uniform(0.05, 0.2)
            a = random.uniform(0.0005, 0.002)
            y = -a * (t - t0) ** 2 + (1.0 + a * t0 ** 2)

    elif class_id == 2:  # Single-valley
        t0 = length * random.uniform(0.2, 0.8)        # Valley position
        sigma_left = random.uniform(5, 15)             # Left width (descent)
        sigma_right = random.uniform(5, 15)            # Right width (ascent)
        valley_depth = random.uniform(3, 10)           # Valley depth (delta)
        baseline = random.uniform(5, 15)               # Baseline level
        y = np.zeros(length)
        for i, ti in enumerate(t):
            if ti < t0:
                # Left half: descending into the valley
                y[i] = baseline - valley_depth * np.exp(-(ti - t0) ** 2 / (2 * sigma_left ** 2))
            else:
                # Right half: ascending out of the valley
                y[i] = baseline - valley_depth * np.exp(-(ti - t0) ** 2 / (2 * sigma_right ** 2))

    elif class_id == 3:  # Single-peak
        t0 = length * random.uniform(0.2, 0.8)        # Peak position (left- or right-skewed)
        sigma_left = random.uniform(5, 15)             # Left width (ascent, larger = gentler)
        sigma_right = random.uniform(5, 15)            # Right width (descent, larger = gentler)
        peak_height = random.uniform(3, 10)            # Peak height (h)
        baseline = random.uniform(0, 1)                # Baseline level
        y = np.zeros(length)
        for i, ti in enumerate(t):
            if ti < t0:
                # Left half: ascending to the peak
                y[i] = baseline + peak_height * np.exp(-(ti - t0) ** 2 / (2 * sigma_left ** 2))
            else:
                # Right half: descending from the peak
                y[i] = baseline + peak_height * np.exp(-(ti - t0) ** 2 / (2 * sigma_right ** 2))

    else:  # Multi-peak (3-5 peaks with controlled spacing)
        n_peaks = random.randint(3, 5)
        # Peak width scales with sequence length (5%-15% of total length)
        min_width = length * 0.05
        max_width = length * 0.15
        baseline = random.uniform(0, 2)
        y = np.full_like(t, baseline)

        # Peak centres are sampled within the central 60% of each equally partitioned segment
        centers = []
        segment_len = length / n_peaks
        for i in range(n_peaks):
            low = i * segment_len + segment_len * 0.2
            high = i * segment_len + segment_len * 0.8
            center = random.uniform(low, high)
            centers.append(center)

        # Amplitudes: first two peaks (base_amp * 0.8-1.4), remaining peaks (base_amp * 0.5-1.2)
        base_amp = random.uniform(2, 5)
        amplitudes = []
        for i in range(n_peaks):
            if i < 2:
                amp = base_amp * random.uniform(0.8, 1.4)
            else:
                amp = base_amp * random.uniform(0.5, 1.2)
            amplitudes.append(amp)

        for center, amp in zip(centers, amplitudes):
            width = random.uniform(min_width, max_width)
            y += amp * np.exp(-((t - center) ** 2) / (2 * width ** 2))

    return y


# ===================== Noise Generation Functions =====================
def ar1_noise(length, rho, std):
    """Generate AR(1) temporally correlated noise with autocorrelation rho."""
    noise = np.zeros(length)
    noise[0] = np.random.normal(0, std)
    for i in range(1, length):
        noise[i] = rho * noise[i - 1] + np.random.normal(0, std * np.sqrt(1 - rho ** 2))
    return noise


def multiplicative_noise(signal, log_std):
    """Multiplicative lognormal noise: y_noisy = y * epsilon, epsilon ~ Lognormal(0, log_std^2)."""
    mult = np.random.lognormal(0, log_std, size=len(signal))
    return signal * mult


def additive_noise(signal, std_ratio):
    """Additive Gaussian noise with standard deviation = std_ratio * std(signal)."""
    std = np.std(signal) * std_ratio
    return np.random.normal(0, std, size=len(signal))


def outlier_noise(signal, prob, factor_range=(2.0, 4.0)):
    """Random multiplicative spikes: with probability prob, a point is multiplied by U(2, 4)."""
    out = signal.copy()
    mask = np.random.rand(len(signal)) < prob
    n_out = np.sum(mask)
    if n_out > 0:
        factors = np.random.uniform(factor_range[0], factor_range[1], n_out)
        out[mask] *= factors
    return out


# ===================== Noise Combination Application =====================
def apply_noise_combination(signal, config):
    """
    Apply a sequence of noise operations to a clean signal.
    Supported keys: 'multiplicative', 'additive', 'ar', 'outlier'.
    """
    y = signal.copy()

    if config.get('multiplicative') is not None:
        mult_log_std = config['multiplicative']['log_std']
        y = multiplicative_noise(y, mult_log_std)

    add_noise_arr = None
    ar_noise_arr = None

    if config.get('additive') is not None:
        add_std_ratio = config['additive']['std_ratio']
        add_noise_arr = additive_noise(y, add_std_ratio)

    if config.get('ar') is not None:
        rho = config['ar']['rho']
        ar_std_ratio = config['ar']['std_ratio']
        ar_noise_arr = ar1_noise(len(y), rho, np.std(y) * ar_std_ratio)

    if add_noise_arr is not None or ar_noise_arr is not None:
        total_add = np.zeros_like(y)
        if add_noise_arr is not None:
            total_add += add_noise_arr
        if ar_noise_arr is not None:
            total_add += ar_noise_arr
        y = y + total_add

    if config.get('outlier') is not None:
        prob = config['outlier']['prob']
        y = outlier_noise(y, prob)

    return y


# ===================== Dataset Generation (with progress display) =====================
def generate_dataset(base_config, strength, config_name, output_dir):
    """
    Generate one synthetic dataset (5,000 time series with variable length
    drawn uniformly from [LEN_MIN, LEN_MAX]) for a given noise composition
    (base_config) and noise intensity (strength).
    """
    global CURRENT_DATASET
    CURRENT_DATASET += 1

    # Fill in the strength-specific parameters for each noise type
    config = {}
    for noise_type in base_config:
        config[noise_type] = STRENGTH_PARAMS[strength][noise_type].copy()

    full_name = f"{config_name}_{strength}"
    print(f"\n[{CURRENT_DATASET}/{TOTAL_DATASETS}] Generating {full_name} ...")

    all_vals, all_cls, all_sids, all_tids = [], [], [], []
    sample_id = 0
    total_samples = NUM_CLASSES * SAMPLES_PER_CLASS   # Total samples per dataset

    for class_id in range(NUM_CLASSES):
        for sample_idx in range(SAMPLES_PER_CLASS):
            # Overall progress within the current dataset
            current_sample = class_id * SAMPLES_PER_CLASS + sample_idx + 1
            progress = (current_sample / total_samples) * 100

            # Real-time progress update (overwrite current line)
            sys.stdout.write(f"\r   Progress: {current_sample}/{total_samples} ({progress:.1f}%)")
            sys.stdout.flush()

            # Generate one sample: variable length drawn from [LEN_MIN, LEN_MAX]
            length = random.randint(LEN_MIN, LEN_MAX)
            base = generate_base_curve(class_id, length)
            noisy = apply_noise_combination(base, config)

            for t in range(length):
                all_vals.append(noisy[t])
                all_cls.append(class_id)
                all_sids.append(sample_id)
                all_tids.append(t)
            sample_id += 1

    # Save the dataset
    df = pd.DataFrame({
        "class_label": all_cls,
        "sample_id": all_sids,
        "time_step_id": all_tids,
        "value": all_vals
    })
    save_path = os.path.join(output_dir, f"timeseries_{full_name}.csv")
    df.to_csv(save_path, index=False)
    print(f"\n   Saved to {save_path}")
    print(f"   Overall progress: {(CURRENT_DATASET / TOTAL_DATASETS) * 100:.1f}%")


# ===================== Main =====================
def main():
    print("=" * 50)
    print("Starting synthetic time series dataset generation")
    print(f"Total tasks: {TOTAL_DATASETS} datasets (3 noise compositions x 3 intensities)")
    print(f"Each dataset: {NUM_CLASSES} classes x {SAMPLES_PER_CLASS} samples/class")
    print("=" * 50)

    for config_name, base_config in BASE_CONFIGS.items():
        for strength in ['low', 'medium', 'high']:
            generate_dataset(base_config, strength, config_name, OUTPUT_DIR)

    # Write metadata describing the noise parameters used for each dataset
    metadata = []
    for config_name in BASE_CONFIGS.keys():
        for strength in ['low', 'medium', 'high']:
            row = {'dataset': f"{config_name}_{strength}", 'strength': strength}
            params = STRENGTH_PARAMS[strength]
            for noise_type in ['additive', 'ar', 'multiplicative', 'outlier']:
                if noise_type in params:
                    for k, v in params[noise_type].items():
                        row[f"{noise_type}_{k}"] = v
                else:
                    row[f"{noise_type}_present"] = False
            metadata.append(row)

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(OUTPUT_DIR, "strength_metadata.csv"), index=False)

    print(f"\nAll datasets generated successfully.")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Total files: {TOTAL_DATASETS} datasets + 1 metadata file")


if __name__ == "__main__":
>>>>>>> b297a6585c7af3d1197b1a55708e8e1c6d1bda15
    main()