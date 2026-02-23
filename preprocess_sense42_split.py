"""
Build train_{id}.pt / test_{id}.pt from per-segment .pt files for TERSE.

Split strategy: participant-level (no within-person split).
  - Persons 1-16  → source (pretrain)
  - Persons 17-24 → target (adapt & evaluate)

For each person, ALL their data goes into both train_{id}.pt and test_{id}.pt
because the framework loads both for every scenario participant.

Classification labels are discretized from raw NASA-TLX (index 0):
  raw < 30  → 0 (low)
  30 ≤ raw < 50 → 1 (medium)
  raw ≥ 50  → 2 (high)

Usage:
    python preprocess_sense42_split.py
    python preprocess_sense42_split.py --task regression
"""

import os
import argparse
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEGMENT_DIR = os.path.join(SCRIPT_DIR, "data", "load_data_sense_42")
OUTPUT_DIR = SEGMENT_DIR

PERSON_IDS = list(range(1, 25))
LABEL_THRESHOLDS = [30, 50]


def discretize_label(raw_score, thresholds):
    """Map continuous NASA-TLX score to class index."""
    for i, t in enumerate(thresholds):
        if raw_score < t:
            return i
    return len(thresholds)


def load_person_segments(person_dir):
    """Load and concatenate all seg*.pt files for one person."""
    seg_files = sorted(f for f in os.listdir(person_dir) if f.endswith(".pt"))
    all_samples, all_raw_labels = [], []

    for sf in seg_files:
        d = torch.load(os.path.join(person_dir, sf), weights_only=True)
        samples = d["samples"]                 # (N, 32, W)
        raw_nasa_tlx = d["labels"][0].item()   # scalar: raw NASA-TLX
        n_windows = samples.shape[0]

        all_samples.append(samples)
        all_raw_labels.extend([raw_nasa_tlx] * n_windows)

    if not all_samples:
        return None, None

    return torch.cat(all_samples, dim=0), np.array(all_raw_labels, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build per-person train/test .pt files")
    parser.add_argument("--segment_dir", type=str, default=SEGMENT_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="classification: discretize labels; regression: keep raw score")
    args = parser.parse_args()

    print("=" * 60)
    print("SENSE-42 Participant-Level Split")
    print("=" * 60)
    print(f"  Segment dir: {args.segment_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Task:        {args.task}")
    print(f"  Persons:     {PERSON_IDS[0]}-{PERSON_IDS[-1]}")
    if args.task == "classification":
        print(f"  Thresholds:  {LABEL_THRESHOLDS}  →  low / medium / high")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    saved = 0

    for pid_num in PERSON_IDS:
        pid_str = f"P{pid_num:03d}"
        person_dir = os.path.join(args.segment_dir, pid_str)

        if not os.path.isdir(person_dir):
            print(f"  {pid_str}: SKIPPED (directory not found)")
            continue

        samples, raw_labels = load_person_segments(person_dir)
        if samples is None:
            print(f"  {pid_str}: SKIPPED (no segment files)")
            continue

        if args.task == "classification":
            labels = torch.tensor(
                [discretize_label(v, LABEL_THRESHOLDS) for v in raw_labels],
                dtype=torch.long,
            )
            class_counts = {i: int((labels == i).sum()) for i in range(len(LABEL_THRESHOLDS) + 1)}
            extra_info = f"class dist: {class_counts}"
        else:
            labels = torch.from_numpy(raw_labels).float()
            extra_info = f"label range: [{raw_labels.min():.1f}, {raw_labels.max():.1f}]"

        data = {"samples": samples, "labels": labels}

        for prefix in ("train", "test"):
            out_path = os.path.join(args.output_dir, f"{prefix}_{pid_num}.pt")
            torch.save(data, out_path)

        role = "SOURCE" if pid_num <= 16 else "TARGET"
        print(f"  {pid_str} (id={pid_num:>2d}, {role}): "
              f"{samples.shape[0]:>5d} windows, shape={tuple(samples.shape)}, {extra_info}")
        saved += 1

    print()
    print(f"Saved {saved} persons × 2 files = {saved * 2} .pt files")
    print("Done!")


if __name__ == "__main__":
    main()
