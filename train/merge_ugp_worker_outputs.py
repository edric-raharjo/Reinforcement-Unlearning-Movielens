#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import sys
from pathlib import Path

import pandas as pd


DEFAULT_BASE_ROOT = Path("D:/Bob_Skripsi_Do Not Delete")
DEFAULT_MERGED_ROOT = DEFAULT_BASE_ROOT / "results_ugp_analysis"
DEFAULT_WORKER_ROOTS = [
    DEFAULT_BASE_ROOT / f"results_ugp_analysis_worker{i}"
    for i in range(4)
]

METRIC_KEY_COLS = ["setting_id", "method", "K"]
PROGRESS_KEY_COLS = ["setting_id", "method"]
LOSS_KEY_COLS = ["setting_id", "method", "iter"]
SELECTION_KEY_COLS = ["method", "source_row_id"]


def read_existing_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def merge_csvs(worker_paths, out_path, dedupe_cols):
    frames = []
    for path in worker_paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return False
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return True


def copy_models(worker_roots, merged_root):
    merged_models_dir = merged_root / "models"
    merged_models_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for worker_root in worker_roots:
        worker_models_dir = worker_root / "models"
        if not worker_models_dir.exists():
            continue
        for model_path in worker_models_dir.glob("*.pt"):
            target = merged_models_dir / model_path.name
            if not target.exists():
                shutil.copy2(model_path, target)
                copied += 1
    return copied


def main():
    if len(sys.argv) > 1:
        merged_root = Path(sys.argv[1])
        worker_roots = [Path(p) for p in sys.argv[2:]] if len(sys.argv) > 2 else DEFAULT_WORKER_ROOTS
    else:
        merged_root = DEFAULT_MERGED_ROOT
        worker_roots = DEFAULT_WORKER_ROOTS

    print(f"Merged output root : {merged_root}")
    print("Worker roots:")
    for worker_root in worker_roots:
        print(f"  - {worker_root}")

    merged_metrics = merge_csvs(
        [root / "metrics" / "relearn_metrics.csv" for root in worker_roots],
        merged_root / "metrics" / "relearn_metrics.csv",
        METRIC_KEY_COLS,
    )
    merged_progress = merge_csvs(
        [root / "metrics" / "relearn_progress.csv" for root in worker_roots],
        merged_root / "metrics" / "relearn_progress.csv",
        PROGRESS_KEY_COLS,
    )
    merged_loss = merge_csvs(
        [root / "metrics" / "relearn_loss_log.csv" for root in worker_roots],
        merged_root / "metrics" / "relearn_loss_log.csv",
        LOSS_KEY_COLS,
    )
    merged_selection = merge_csvs(
        [root / "metrics" / "relearn_selection_summary.csv" for root in worker_roots],
        merged_root / "metrics" / "relearn_selection_summary.csv",
        SELECTION_KEY_COLS,
    )
    copied_models = copy_models(worker_roots, merged_root)

    print(f"relearn_metrics.csv merged      : {merged_metrics}")
    print(f"relearn_progress.csv merged     : {merged_progress}")
    print(f"relearn_loss_log.csv merged     : {merged_loss}")
    print(f"relearn_selection_summary merged: {merged_selection}")
    print(f"Models copied                   : {copied_models}")
    print("Merge complete.")


if __name__ == "__main__":
    main()
