"""
Preprocess the SENSE-42 EEG dataset into per-participant .pt files for domain adaptation.

Data structure expected in data/sense_42/:
  - Parquet files: {PID}_seg{NN}_eeg_raw.parquet
      Each parquet has 32 EEG channels (columns) x N time samples (rows) at 1024 Hz.
  - Label files: {PID}_seg{NN}_eeg_raw.txt  (overall NASA-TLX workload score, 0-100)

Steps:
1. Read all valid parquet files and their matching label .txt files
2. Window the EEG data into fixed-length segments (default 512 samples = ~0.5s)
3. Discretize workload labels into 3 classes: low / medium / high
4. Group by participant and split 80% train / 20% test
5. Save as train_{participant_id}.pt and test_{participant_id}.pt

If no valid parquet files are found, generates synthetic EEG-like data for pipeline testing.

Usage:
    python preprocess_sense42.py
    python preprocess_sense42.py --window_size 1024 --step_size 512
    python preprocess_sense42.py --generate_synthetic  # Force synthetic data generation
"""

import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd
from collections import defaultdict

# === Configuration ===
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "sense_42")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "SENSE_42")

# 32 standard EEG channels from the SENSE-42 dataset
EEG_CHANNELS = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
    "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
    "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz"
]
NUM_CHANNELS = 32
SAMPLING_RATE = 1024  # Hz

# Workload label discretization thresholds
# NASA-TLX scores: 0-100 → 3 classes
LOW_THRESHOLD = 33.33       # 0 - 33.33 → class 0 (low)
HIGH_THRESHOLD = 66.67      # 66.67 - 100 → class 2 (high)
                            # 33.33 - 66.67 → class 1 (medium)

TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

# Default windowing parameters
DEFAULT_WINDOW_SIZE = 512   # samples (~0.5 sec at 1024 Hz)
DEFAULT_STEP_SIZE = 256     # 50% overlap


def discretize_label(score):
    """Convert continuous NASA-TLX score to discrete class."""
    if score <= LOW_THRESHOLD:
        return 0  # low workload
    elif score <= HIGH_THRESHOLD:
        return 1  # medium workload
    else:
        return 2  # high workload


def read_label(txt_path):
    """Read a NASA-TLX score from a .txt file. Returns None if unreadable."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return float(content)
    except (ValueError, UnicodeDecodeError, FileNotFoundError):
        return None


def read_parquet(parquet_path):
    """Read an EEG parquet file. Returns DataFrame or None if corrupt."""
    try:
        df = pd.read_parquet(parquet_path)
        if df.shape[1] == NUM_CHANNELS and df.shape[0] > 0:
            return df
        else:
            return None
    except Exception:
        return None


def window_signal(data, window_size, step_size):
    """
    Slide a window over multi-channel time series data.

    Args:
        data: np.ndarray of shape (num_channels, num_samples)
        window_size: int, number of samples per window
        step_size: int, stride between windows

    Returns:
        np.ndarray of shape (num_windows, num_channels, window_size)
    """
    num_channels, num_samples = data.shape
    if num_samples < window_size:
        return np.array([]).reshape(0, num_channels, window_size)

    windows = []
    for start in range(0, num_samples - window_size + 1, step_size):
        window = data[:, start:start + window_size]
        windows.append(window)

    return np.array(windows)


def load_real_data(window_size, step_size):
    """
    Load real SENSE-42 data from parquet files.

    Returns:
        participant_data: dict mapping participant_id to list of (windows, label_class) tuples
        stats: dict with loading statistics
    """
    participant_data = defaultdict(list)
    stats = {'total_parquet': 0, 'valid_parquet': 0, 'matched': 0,
             'total_windows': 0, 'skipped_corrupt': 0, 'skipped_no_label': 0}

    # Find all parquet files
    if not os.path.exists(RAW_DATA_DIR):
        print(f"  Data directory not found: {RAW_DATA_DIR}")
        return participant_data, stats

    parquet_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.parquet')])
    stats['total_parquet'] = len(parquet_files)

    for pf in parquet_files:
        # Parse filename: P001_seg01_eeg_raw.parquet
        base = pf[:-8]  # remove .parquet
        parts = base.split('_')
        participant_id = parts[0]  # e.g., "P001"

        # Read the corresponding label
        label_path = os.path.join(RAW_DATA_DIR, base + '.txt')
        score = read_label(label_path)
        if score is None:
            stats['skipped_no_label'] += 1
            continue

        label_class = discretize_label(score)

        # Read the parquet EEG data
        parquet_path = os.path.join(RAW_DATA_DIR, pf)
        df = read_parquet(parquet_path)
        if df is None:
            stats['skipped_corrupt'] += 1
            continue

        stats['valid_parquet'] += 1

        # Convert to numpy: shape (num_channels, num_samples)
        eeg_data = df.values.T  # (32, N)

        # Window the data
        windows = window_signal(eeg_data, window_size, step_size)
        if windows.shape[0] == 0:
            continue

        stats['matched'] += 1
        stats['total_windows'] += windows.shape[0]
        participant_data[participant_id].append((windows, label_class))

    return participant_data, stats


def generate_synthetic_data(num_participants=12, segments_per_participant=25,
                            samples_per_segment=10240, window_size=512, step_size=256):
    """
    Generate synthetic EEG-like data for pipeline testing.

    Creates realistic-looking multi-channel signals with:
    - Alpha/beta/gamma band oscillations
    - Per-participant variations (domain shift)
    - Workload-correlated features
    """
    print("\n  Generating synthetic EEG data for pipeline testing...")
    rng = np.random.RandomState(RANDOM_SEED)
    participant_data = defaultdict(list)

    for p_idx in range(1, num_participants + 1):
        pid = f"P{p_idx:03d}"

        # Per-participant "domain" characteristics
        alpha_freq = 10.0 + rng.randn() * 1.0   # 8-12 Hz alpha band
        beta_freq = 22.0 + rng.randn() * 3.0    # 13-30 Hz beta band
        noise_level = 0.3 + rng.rand() * 0.2
        amplitude_scale = 1.0 + rng.randn() * 0.2

        for seg_idx in range(segments_per_participant):
            # Random workload level for this segment
            score = rng.uniform(0, 80)  # NASA-TLX score
            label_class = discretize_label(score)

            # Generate multi-channel EEG signal
            t = np.arange(samples_per_segment) / SAMPLING_RATE
            eeg = np.zeros((NUM_CHANNELS, samples_per_segment))

            for ch in range(NUM_CHANNELS):
                # Base oscillations (workload affects beta power)
                alpha_power = 1.0 - score / 200.0  # alpha desynchronizes with workload
                beta_power = 0.5 + score / 100.0    # beta increases with workload

                phase = rng.uniform(0, 2 * np.pi)
                signal = (
                    alpha_power * np.sin(2 * np.pi * alpha_freq * t + phase) +
                    beta_power * np.sin(2 * np.pi * beta_freq * t + rng.uniform(0, 2 * np.pi)) +
                    0.3 * np.sin(2 * np.pi * 40.0 * t + rng.uniform(0, 2 * np.pi)) +  # gamma
                    noise_level * rng.randn(samples_per_segment)
                )
                eeg[ch] = amplitude_scale * signal * 1e-4  # ~100 uV scale

            # Window the data
            windows = window_signal(eeg, window_size, step_size)
            if windows.shape[0] > 0:
                participant_data[pid].append((windows, label_class))

    total_windows = sum(
        w.shape[0] for segs in participant_data.values() for w, _ in segs
    )
    print(f"  Generated: {num_participants} participants, "
          f"{segments_per_participant} segments each, "
          f"{total_windows} total windows")

    return participant_data


def save_participant_data(participant_data, output_dir, window_size):
    """
    Split each participant's data into train/test and save as .pt files.

    Args:
        participant_data: dict mapping participant_id to list of (windows, label) tuples
        output_dir: directory to save .pt files
        window_size: window size used (for logging)
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)

    # Map participant IDs to integer IDs for the TERSE framework
    # e.g., P001 → "1", P002 → "2", etc.
    pid_to_int = {}
    for pid in sorted(participant_data.keys()):
        int_id = str(int(pid[1:]))  # "P001" → "1", "P010" → "10"
        pid_to_int[pid] = int_id

    print(f"\n  Saving .pt files to: {output_dir}")
    print("-" * 60)

    for pid in sorted(participant_data.keys()):
        int_id = pid_to_int[pid]

        # Collect all windows and labels for this participant
        all_windows = []
        all_labels = []
        for windows, label in participant_data[pid]:
            all_windows.append(windows)
            all_labels.extend([label] * windows.shape[0])

        if not all_windows:
            continue

        all_windows = np.concatenate(all_windows, axis=0)  # (N, C, L)
        all_labels = np.array(all_labels)

        n_samples = all_windows.shape[0]
        if n_samples < 2:
            print(f"  {pid} (id={int_id}): Only {n_samples} sample(s), skipping")
            continue

        # Shuffle and split
        indices = rng.permutation(n_samples)
        n_train = max(1, int(n_samples * TRAIN_TEST_SPLIT_RATIO))

        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_data = {
            "samples": torch.from_numpy(all_windows[train_idx]).float(),
            "labels": torch.from_numpy(all_labels[train_idx]).long(),
        }
        test_data = {
            "samples": torch.from_numpy(all_windows[test_idx]).float(),
            "labels": torch.from_numpy(all_labels[test_idx]).long(),
        }

        # Save .pt files
        train_path = os.path.join(output_dir, f"train_{int_id}.pt")
        test_path = os.path.join(output_dir, f"test_{int_id}.pt")
        torch.save(train_data, train_path)
        torch.save(test_data, test_path)

        label_dist = dict(zip(*np.unique(all_labels, return_counts=True)))
        print(f"  {pid} (id={int_id:>3s}): {n_samples:>5d} windows -> "
              f"train={len(train_idx)}, test={len(test_idx)}  |  "
              f"shape: ({all_windows.shape[1]}, {all_windows.shape[2]})  |  "
              f"labels: {label_dist}")

    print("-" * 60)
    return pid_to_int


def main():
    parser = argparse.ArgumentParser(description="Preprocess SENSE-42 EEG dataset")
    parser.add_argument('--window_size', type=int, default=DEFAULT_WINDOW_SIZE,
                        help=f'Window size in samples (default: {DEFAULT_WINDOW_SIZE})')
    parser.add_argument('--step_size', type=int, default=DEFAULT_STEP_SIZE,
                        help=f'Step size / stride in samples (default: {DEFAULT_STEP_SIZE})')
    parser.add_argument('--generate_synthetic', action='store_true',
                        help='Force generation of synthetic data even if real data exists')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()

    print("=" * 60)
    print("SENSE-42 EEG Dataset Preprocessing")
    print("=" * 60)
    print(f"  Window size:  {args.window_size} samples ({args.window_size / SAMPLING_RATE * 1000:.0f} ms)")
    print(f"  Step size:    {args.step_size} samples ({args.step_size / SAMPLING_RATE * 1000:.0f} ms)")
    print(f"  Channels:     {NUM_CHANNELS}")
    print(f"  Classes:      3 (low / medium / high workload)")
    print(f"  Train/test:   {TRAIN_TEST_SPLIT_RATIO:.0%} / {1 - TRAIN_TEST_SPLIT_RATIO:.0%}")

    use_synthetic = args.generate_synthetic
    participant_data = {}

    if not use_synthetic:
        print(f"\nLoading real data from: {RAW_DATA_DIR}")
        participant_data, stats = load_real_data(args.window_size, args.step_size)
        print(f"  Total parquet files:   {stats['total_parquet']}")
        print(f"  Valid (readable):      {stats['valid_parquet']}")
        print(f"  Matched with labels:   {stats['matched']}")
        print(f"  Skipped (corrupt):     {stats['skipped_corrupt']}")
        print(f"  Skipped (no label):    {stats['skipped_no_label']}")
        print(f"  Total windows:         {stats['total_windows']}")
        print(f"  Participants:          {len(participant_data)}")

        if stats['total_windows'] < 100:
            print("\n  WARNING: Very few valid data windows found.")
            print("  The parquet files appear to be corrupted (truncated during extraction).")
            print("  Falling back to synthetic data for pipeline testing.")
            print("  To use real data, re-extract from the original BDF files.")
            use_synthetic = True

    if use_synthetic:
        participant_data = generate_synthetic_data(
            num_participants=12,
            segments_per_participant=25,
            samples_per_segment=10240,
            window_size=args.window_size,
            step_size=args.step_size
        )

    # Save per-participant .pt files
    pid_to_int = save_participant_data(participant_data, args.output_dir, args.window_size)

    # Print scenario mapping for configs
    print(f"\nParticipant ID mapping:")
    for pid, int_id in sorted(pid_to_int.items()):
        print(f"  {pid} -> {int_id}")

    # Verify saved files
    print(f"\nVerifying saved files in: {args.output_dir}")
    pt_files = sorted([f for f in os.listdir(args.output_dir) if f.endswith('.pt')])
    for f in pt_files[:10]:
        data = torch.load(os.path.join(args.output_dir, f), weights_only=True)
        print(f"  {f}: samples={data['samples'].shape}, labels={data['labels'].shape}")
    if len(pt_files) > 10:
        print(f"  ... and {len(pt_files) - 10} more files")

    print("\n" + "=" * 60)
    print("Done! .pt files saved.")
    print(f"\nTo train, run:")
    print(f"  cd trainers")
    print(f"  python train.py --dataset SENSE_42 --data_path ../data --device cpu")
    print("=" * 60)


if __name__ == "__main__":
    main()
