"""
Preprocess the SENSE-42 EEG dataset into per-person, per-time-segment .pt files.

Data structure expected in data/sense42_raw_eeg_extracted_rawonly 2/:
  - Parquet files: {PID}_seg{NN}_eeg_raw.parquet (32 EEG channels x N time samples at 1024 Hz)
  - Label files:   {PID}_seg{NN}_eeg_raw.txt (overall NASA-TLX)
  - Subscale labels: {PID}_seg{NN}_eeg_raw_mental.txt, _frustration.txt,
                     _effort.txt, _performance.txt, _temporal.txt

Output structure (data/load_data_sense_42/):
  P001/
    seg01.pt   ->  {"samples": (N, 32, W), "labels": (6,), "labels_per_window": (N, 6)}
    seg02.pt
    ...
  P002/
    ...

Each .pt file stores one time segment for one person:
  - samples: windowed EEG tensor of shape (num_windows, 32, window_size)
  - labels: 1D tensor of 6 NASA-TLX scores [raw, mental, frustration, effort, performance, temporal]
  - labels_per_window: (num_windows, 6) — same label vector repeated per window

No train/test split is performed here; the user handles that separately.

Usage:
    python preprocess_sense42_per_segment.py
    python preprocess_sense42_per_segment.py --window_size 1024 --step_size 512
    python preprocess_sense42_per_segment.py --limit 10
"""

import os
import argparse
import re
import numpy as np
import torch
import polars as pl

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "sense42_raw_eeg_extracted_rawonly 2")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "load_data_sense_42")

NUM_CHANNELS = 32
SAMPLING_RATE = 1024
DEFAULT_WINDOW_SIZE = 512
DEFAULT_STEP_SIZE = 256

SUBSCALES = ("raw", "mental", "frustration", "effort", "performance", "temporal")


def read_label(txt_path):
    """Read a NASA-TLX score from a .txt file. Returns None if unreadable."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return float(f.read().strip())
    except (ValueError, UnicodeDecodeError, FileNotFoundError):
        return None


def read_parquet(parquet_path):
    """Read an EEG parquet file with polars. Returns numpy array (C, T) or None."""
    try:
        df = pl.read_parquet(parquet_path)
        if df.shape[1] == NUM_CHANNELS and df.shape[0] > 0:
            return df.to_numpy().T  # (32, N)
        return None
    except Exception:
        return None


def window_signal(data, window_size, step_size):
    """
    Slide a window over multi-channel time series data.

    Returns:
        np.ndarray of shape (num_windows, num_channels, window_size)
    """
    num_channels, num_samples = data.shape
    if num_samples < window_size:
        return np.array([]).reshape(0, num_channels, window_size)

    windows = []
    for start in range(0, num_samples - window_size + 1, step_size):
        windows.append(data[:, start:start + window_size])

    return np.array(windows)


def load_all_labels(data_dir, base):
    """
    Load all 6 NASA-TLX scores for a segment.

    Order: [raw, mental, frustration, effort, performance, temporal]
    Returns a (6,) float32 tensor, or None if any label is missing.
    """
    values = []
    for sub in SUBSCALES:
        if sub == "raw":
            path = os.path.join(data_dir, base + ".txt")
        else:
            path = os.path.join(data_dir, f"{base}_{sub}.txt")
        val = read_label(path)
        if val is None:
            return None
        values.append(val)
    return torch.tensor(values, dtype=torch.float32)


def parse_filename(parquet_name):
    """
    Extract participant ID and segment tag from a parquet filename.

    E.g. 'P001_seg01_eeg_raw.parquet' -> ('P001', 'seg01')
    Returns (pid, seg) or (None, None) on parse failure.
    """
    m = re.match(r"^(P\d+)_(seg\d+)_eeg_raw\.parquet$", parquet_name)
    if m:
        return m.group(1), m.group(2)
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SENSE-42 EEG into per-person per-segment .pt files")
    parser.add_argument('--input_dir', type=str, default=RAW_DATA_DIR,
                        help='Input directory with parquet + .txt label files')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output root directory for per-person folders')
    parser.add_argument('--window_size', type=int, default=DEFAULT_WINDOW_SIZE,
                        help=f'Window size in samples (default: {DEFAULT_WINDOW_SIZE})')
    parser.add_argument('--step_size', type=int, default=DEFAULT_STEP_SIZE,
                        help=f'Step size / stride in samples (default: {DEFAULT_STEP_SIZE})')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N parquet files (for testing)')
    args = parser.parse_args()

    print("=" * 60)
    print("SENSE-42 Per-Person Per-Segment Preprocessing")
    print("=" * 60)
    print(f"  Input dir:   {args.input_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Window size: {args.window_size} samples ({args.window_size / SAMPLING_RATE * 1000:.0f} ms)")
    print(f"  Step size:   {args.step_size} samples")
    print(f"  Labels:      {list(SUBSCALES)}")

    if not os.path.exists(args.input_dir):
        print(f"\n  ERROR: Input directory not found: {args.input_dir}")
        return

    parquet_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith('.parquet'))
    if args.limit is not None:
        parquet_files = parquet_files[:args.limit]
        print(f"  (Limited to first {args.limit} parquet files)")

    total = len(parquet_files)
    saved = 0
    skipped_parse = 0
    skipped_label = 0
    skipped_corrupt = 0
    skipped_empty = 0
    persons_seen = set()

    print(f"\nProcessing {total} parquet files ...")
    print("-" * 60)

    for pf in parquet_files:
        pid, seg = parse_filename(pf)
        if pid is None:
            skipped_parse += 1
            continue

        base = pf[:-8]  # strip .parquet

        labels = load_all_labels(args.input_dir, base)
        if labels is None:
            skipped_label += 1
            continue

        eeg = read_parquet(os.path.join(args.input_dir, pf))
        if eeg is None:
            skipped_corrupt += 1
            continue

        windows = window_signal(eeg, args.window_size, args.step_size)
        if windows.shape[0] == 0:
            skipped_empty += 1
            continue

        person_dir = os.path.join(args.output_dir, pid)
        os.makedirs(person_dir, exist_ok=True)

        samples = torch.from_numpy(windows).float()
        data = {
            "samples": samples,
            "labels": labels,
            "labels_per_window": labels.unsqueeze(0).expand(samples.shape[0], -1).clone(),
        }

        out_path = os.path.join(person_dir, f"{seg}.pt")
        torch.save(data, out_path)

        persons_seen.add(pid)
        saved += 1
        print(f"  {pid}/{seg}.pt  ->  {samples.shape[0]} windows, labels={labels.tolist()}")

    print("-" * 60)
    print(f"  Total parquet files:  {total}")
    print(f"  Saved .pt files:      {saved}")
    print(f"  Persons:              {len(persons_seen)}")
    print(f"  Skipped (parse):      {skipped_parse}")
    print(f"  Skipped (no label):   {skipped_label}")
    print(f"  Skipped (corrupt):    {skipped_corrupt}")
    print(f"  Skipped (empty):      {skipped_empty}")

    if saved > 0:
        print(f"\nVerifying a sample of saved files ...")
        count = 0
        for pid in sorted(persons_seen):
            person_dir = os.path.join(args.output_dir, pid)
            pt_files = sorted(f for f in os.listdir(person_dir) if f.endswith('.pt'))
            for f in pt_files[:2]:
                d = torch.load(os.path.join(person_dir, f), weights_only=True)
                print(f"  {pid}/{f}: samples={d['samples'].shape}, "
                      f"labels={d['labels'].shape}, "
                      f"labels_per_window={d['labels_per_window'].shape}")
                count += 1
            if count >= 8:
                remaining = saved - count
                if remaining > 0:
                    print(f"  ... and {remaining} more files")
                break

    print("\n" + "=" * 60)
    print("Done! One .pt per person per segment (no train/test split).")
    print("=" * 60)


if __name__ == "__main__":
    main()
