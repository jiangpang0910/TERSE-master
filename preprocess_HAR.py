"""
Preprocess the raw HAR dataset into per-subject .pt files for domain adaptation.

Steps:
1. Load all 9 inertial signal channels from both train/ and test/ directories
2. Load labels (convert from 1-indexed to 0-indexed) and subject IDs
3. Combine train+test data
4. For each needed subject, split their data into ~80% train / ~20% test
5. Save as train_{subject_id}.pt and test_{subject_id}.pt
"""

import os
import numpy as np
import torch

# === Configuration ===
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "HAR")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "HAR")

# Subject IDs needed by the scenarios in data_model_configs.py:
# scenarios = [("2","11"), ("6","23"), ("7","13"), ("9","18"), ("12","16")]
NEEDED_SUBJECTS = {2, 6, 7, 9, 11, 12, 13, 16, 18, 23}

# 9 inertial signal channel files (order matters for consistency)
SIGNAL_FILES = [
    "body_acc_x_{split}.txt",
    "body_acc_y_{split}.txt",
    "body_acc_z_{split}.txt",
    "body_gyro_x_{split}.txt",
    "body_gyro_y_{split}.txt",
    "body_gyro_z_{split}.txt",
    "total_acc_x_{split}.txt",
    "total_acc_y_{split}.txt",
    "total_acc_z_{split}.txt",
]

TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% train, 20% test
RANDOM_SEED = 42


def load_txt(filepath):
    """Load a whitespace-delimited text file into a numpy array."""
    return np.loadtxt(filepath)


def load_split(split):
    """
    Load one split (train or test) of the HAR dataset.
    Returns:
        signals: np.ndarray of shape (N, 9, 128) - 9 channels, 128 timesteps
        labels:  np.ndarray of shape (N,) - 0-indexed labels
        subjects: np.ndarray of shape (N,) - subject IDs
    """
    split_dir = os.path.join(RAW_DATA_DIR, split)
    inertial_dir = os.path.join(split_dir, "Inertial Signals")

    # Load 9 inertial signal channels
    channels = []
    for sig_file in SIGNAL_FILES:
        filepath = os.path.join(inertial_dir, sig_file.format(split=split))
        data = load_txt(filepath)  # shape: (N, 128)
        channels.append(data)

    # Stack channels: (9, N, 128) -> transpose to (N, 9, 128)
    signals = np.stack(channels, axis=0).transpose(1, 0, 2)  # (N, 9, 128)

    # Load labels (1-indexed) and convert to 0-indexed
    labels = load_txt(os.path.join(split_dir, f"y_{split}.txt")).astype(int) - 1

    # Load subject IDs
    subjects = load_txt(os.path.join(split_dir, f"subject_{split}.txt")).astype(int)

    print(f"  Loaded {split}: {signals.shape[0]} samples, "
          f"{signals.shape[1]} channels, {signals.shape[2]} timesteps")

    return signals, labels, subjects


def main():
    print("=" * 50)
    print("HAR Dataset Preprocessing")
    print("=" * 50)

    # Load both splits
    print("\nLoading raw data...")
    train_signals, train_labels, train_subjects = load_split("train")
    test_signals, test_labels, test_subjects = load_split("test")

    # Combine all data
    all_signals = np.concatenate([train_signals, test_signals], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    all_subjects = np.concatenate([train_subjects, test_subjects], axis=0)

    print(f"\nCombined: {all_signals.shape[0]} total samples")
    print(f"Unique subjects: {sorted(np.unique(all_subjects))}")
    print(f"Unique labels: {sorted(np.unique(all_labels))} (0-indexed)")
    print(f"Needed subjects: {sorted(NEEDED_SUBJECTS)}")

    # Process each needed subject
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)

    print(f"\nGenerating .pt files in: {OUTPUT_DIR}")
    print("-" * 50)

    for subject_id in sorted(NEEDED_SUBJECTS):
        mask = all_subjects == subject_id
        subj_signals = all_signals[mask]
        subj_labels = all_labels[mask]
        n_samples = subj_signals.shape[0]

        if n_samples == 0:
            print(f"  WARNING: Subject {subject_id} has no data!")
            continue

        # Shuffle and split into train/test
        indices = rng.permutation(n_samples)
        n_train = int(n_samples * TRAIN_TEST_SPLIT_RATIO)

        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_data = {
            "samples": torch.from_numpy(subj_signals[train_idx]).float(),
            "labels": torch.from_numpy(subj_labels[train_idx]).long(),
        }
        test_data = {
            "samples": torch.from_numpy(subj_signals[test_idx]).float(),
            "labels": torch.from_numpy(subj_labels[test_idx]).long(),
        }

        # Save .pt files
        train_path = os.path.join(OUTPUT_DIR, f"train_{subject_id}.pt")
        test_path = os.path.join(OUTPUT_DIR, f"test_{subject_id}.pt")
        torch.save(train_data, train_path)
        torch.save(test_data, test_path)

        print(f"  Subject {subject_id:>2d}: {n_samples:>4d} samples -> "
              f"train={len(train_idx)}, test={len(test_idx)}  |  "
              f"samples shape: {train_data['samples'].shape}  |  "
              f"label dist: {dict(zip(*np.unique(subj_labels, return_counts=True)))}")

    print("-" * 50)
    print("Done! .pt files saved.")


if __name__ == "__main__":
    main()

