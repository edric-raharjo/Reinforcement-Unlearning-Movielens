#!/usr/bin/env python
# coding: utf-8

import sys
import glob
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Must mirror GPU_Enabled_Combined.py exactly ────────────────────────────
TRAIN_LRS         = [1e-3, 1e-4, 1e-5]
GAMMAS            = [0.99, 0.98, 0.97]
HIDDEN_DIMS       = [128, 256, 512]
TRAIN_BATCH_SIZES = [1, 2, 4]

UNLEARN_LRS   = [1e-3, 1e-4, 1e-5]
UNLEARN_ITERS = [500, 1000, 1500, 2000]

LAMBDA_VALS = sorted(set(
    [round(0.1 * i, 1) for i in range(1, 11)] +
    [0.5 * i for i in range(3, 21)]
))  # 28 values

FIXED_METHODS = ["Ye_ApxI", "Ye_multi"]
SWEPT_METHODS = ["New_True_inf", "New_Max"]

TOP_PERCENT     = 0.05
TOP_SELECTION_K = 10
EVAL_OVERHEAD_S = 20
# ───────────────────────────────────────────────────────────────────────────

_TRAIN_PROG_COLS = ["t_lr", "gamma", "hidden_dim", "train_bs"]
_TRAIN_KEY_COLS  = ["train_lr", "gamma", "hidden_dim", "train_batch", "K"]
_PROG_COLS       = ["t_lr", "gamma", "hidden_dim", "train_bs", "u_lr", "u_iters", "lam", "method"]
_KEY_COLS        = ["train_lr", "gamma", "hidden_dim", "train_batch",
                    "unlearn_lr", "unlearn_iters", "lambda_retain", "method", "K"]

_FIXED_COMBOS  = len(UNLEARN_LRS) * len(UNLEARN_ITERS) * len(FIXED_METHODS)
_SWEPT_COMBOS  = len(UNLEARN_LRS) * len(UNLEARN_ITERS) * len(SWEPT_METHODS) * len(LAMBDA_VALS)
COMBOS_PER_CFG = _FIXED_COMBOS + _SWEPT_COMBOS


def fmt_float(v):
    try:
        return f"{float(v):.3g}"
    except Exception:
        return str(v)


def make_bar(frac, width=30):
    if frac is None or (isinstance(frac, float) and np.isnan(frac)):
        return "[?]" + "-" * (width - 3)
    frac = max(0.0, min(1.0, float(frac)))
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def pretty_td(seconds):
    if seconds is None or np.isnan(seconds):
        return "unknown"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h {m}m"
    d, h = divmod(h, 24)
    return f"{d}d {h}h"


def load_csv_glob(base, merged_name, worker_pattern, dedup_cols=None):
    merged_path = base / merged_name
    if merged_path.exists():
        df = pd.read_csv(merged_path)
        if dedup_cols:
            df = df.drop_duplicates(subset=dedup_cols, keep="last")
        return df, f"merged ({len(df):,} rows)"

    worker_files = sorted(glob.glob(str(base / worker_pattern)))
    if worker_files:
        df = pd.concat([pd.read_csv(f) for f in worker_files], ignore_index=True)
        if dedup_cols:
            df = df.drop_duplicates(subset=dedup_cols, keep="last")
        return df, f"{len(worker_files)} worker file(s) ({len(df):,} rows)"

    return pd.DataFrame(), "not found"


def count_train_progress(base):
    merged = base / "train_phase_progress.csv"
    if merged.exists():
        df = pd.read_csv(merged).drop_duplicates(subset=_TRAIN_PROG_COLS)
        return len(df), "merged"

    worker_files = sorted(glob.glob(str(base / "train_phase_progress_w*.csv")))
    if worker_files:
        df = pd.concat([pd.read_csv(f) for f in worker_files], ignore_index=True)
        df = df.drop_duplicates(subset=_TRAIN_PROG_COLS)
        return len(df), f"{len(worker_files)} worker file(s)"

    return 0, "not found"


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python progress_check.py <forget_percentage> [num_workers]\n"
            "Example: python progress_check.py 20 4"
        )

    forget_pct  = int(sys.argv[1])
    NUM_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # ← NEW

    base = Path(f"C:/Bob/results/{forget_pct}_percent")

    print(f"Checking  : {base}")
    print(f"Workers   : {NUM_WORKERS}")          # ← NEW
    print()
    if not base.exists():
        print("  Folder does not exist yet.")
        return

    # ------------------------------------------------------------------ #
    # PHASE 1 -- Training                                                  #
    # ------------------------------------------------------------------ #
    total_train = (
        len(TRAIN_LRS) * len(GAMMAS) * len(HIDDEN_DIMS) * len(TRAIN_BATCH_SIZES)
    )
    print("=" * 60)
    print("PHASE 1 -- Training")
    print("=" * 60)
    print(f"Total train configs expected : {total_train}")

    done_train, prog_src = count_train_progress(base)
    avg_train_time = np.nan

    if done_train > 0:
        frac = done_train / total_train
        print(
            f"Configs finished : {done_train}/{total_train} "
            f"({frac*100:.1f}%) {make_bar(frac)}  [{prog_src}]"
        )
    else:
        print("No train progress files yet -- training not started.")

    train_res_df, tr_src = load_csv_glob(
        base,
        "train_phase_results.csv",
        "train_phase_results_w*.csv",
        dedup_cols=_TRAIN_KEY_COLS,
    )
    if not train_res_df.empty:
        print(f"Train results source : {tr_src}")
        times = pd.to_numeric(
            train_res_df.get("train_time_s", pd.Series()), errors="coerce"
        ).dropna()
        if not times.empty:
            avg_train_time = times.mean()
            print(f"Avg train_time_s per config : {avg_train_time:.1f}s")

    remaining_train = max(0, total_train - done_train)
    if remaining_train > 0:
        if not np.isnan(avg_train_time):
            # ← Workers train configs in parallel (round-robin assignment)
            eta_s    = (remaining_train * avg_train_time) / NUM_WORKERS
            eta_when = datetime.now() + timedelta(seconds=eta_s)
            print(f"ETA (training done)  : {pretty_td(eta_s)} across {NUM_WORKERS} workers "
                  f"(~ {eta_when:%Y-%m-%d %H:%M:%S})")
        else:
            print("ETA (training done)  : unknown (no timing yet)")
    else:
        print("Training phase       : COMPLETE")

    # ------------------------------------------------------------------ #
    # Determine top configs                                                #
    # ------------------------------------------------------------------ #
    if train_res_df.empty or "K" not in train_res_df.columns:
        print("\nCannot determine top configs -- train results missing/incomplete.")
        return

    k10 = train_res_df[train_res_df["K"] == TOP_SELECTION_K].copy()
    if k10.empty:
        print(f"\nNo K={TOP_SELECTION_K} rows yet -- unlearning not started.")
        return

    rank_df = (
        k10.drop_duplicates(
            subset=["train_lr", "gamma", "hidden_dim", "train_batch"], keep="last"
        )
        .sort_values(
            ["base_retain_Hit", "train_lr", "gamma", "hidden_dim", "train_batch"],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    n_top          = max(1, int(len(rank_df) * TOP_PERCENT))
    top_configs_df = rank_df.head(n_top).copy().reset_index(drop=True)

    print(f"\nTop configs selected (top {TOP_PERCENT*100:.0f}% of {len(rank_df)}) : {n_top}")
    for i, row in top_configs_df.iterrows():
        assigned_worker = i % NUM_WORKERS  # ← show which worker owns this model
        print(
            f"  Model {i+1} [worker {assigned_worker}]: "
            f"train_lr={fmt_float(row['train_lr'])}, "
            f"gamma={fmt_float(row['gamma'])}, "
            f"hidden_dim={int(row['hidden_dim'])}, "
            f"train_batch={int(row['train_batch'])}, "
            f"base_retain_Hit@{TOP_SELECTION_K}={row['base_retain_Hit']:.4f}"
        )

    # ------------------------------------------------------------------ #
    # PHASE 2 -- Unlearning (global)                                       #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("PHASE 2 -- Unlearning")
    print("=" * 60)

    total_expected = n_top * COMBOS_PER_CFG
    print(
        f"Combos per top config : {COMBOS_PER_CFG}  "
        f"= {len(UNLEARN_LRS)} ulrs x {len(UNLEARN_ITERS)} iters "
        f"x ({len(FIXED_METHODS)} fixed-lambda + "
        f"{len(SWEPT_METHODS)} swept-lambda x {len(LAMBDA_VALS)} lambda)"
    )
    print(f"Total combos expected  : {total_expected}  ({n_top} models x {COMBOS_PER_CFG})")

    ul_prog_df, ul_src = load_csv_glob(
        base,
        "progress.csv",
        "progress_w*.csv",
        dedup_cols=_PROG_COLS,
    )

    if ul_prog_df.empty:
        print("No progress files yet -- unlearning not started.")
        return

    print(f"Unlearn progress source : {ul_src}")
    done_total = len(ul_prog_df)
    frac_total = done_total / total_expected if total_expected > 0 else 0.0
    print(
        f"Total combos finished  : {done_total}/{total_expected} "
        f"({frac_total*100:.1f}%) {make_bar(frac_total)}"
    )

    res_df, res_src = load_csv_glob(
        base,
        "tuning_full_results.csv",
        "tuning_full_results_w*.csv",
        dedup_cols=_KEY_COLS,
    )

    avg_ul_global = np.nan
    if not res_df.empty:
        print(f"Results source : {res_src}")
        if "unlearn_time_s" in res_df.columns:
            times = pd.to_numeric(res_df["unlearn_time_s"], errors="coerce").dropna()
            if not times.empty:
                avg_ul_global = times.mean() + EVAL_OVERHEAD_S
                print(
                    f"Global avg time/combo  : {times.mean():.2f}s (unlearn) "
                    f"+ {EVAL_OVERHEAD_S}s (eval) = {avg_ul_global:.2f}s"
                )
    else:
        print("No results files yet -- ETA uses no timing data.")

    remaining_global = max(0, total_expected - done_total)
    if remaining_global > 0 and not np.isnan(avg_ul_global):
        # ← Workers run top configs in parallel (round-robin), so wall-clock
        #   time is total_remaining / num_workers (assuming balanced load)
        eta_s    = (remaining_global * avg_ul_global) / NUM_WORKERS
        eta_when = datetime.now() + timedelta(seconds=eta_s)
        print(f"ETA (all unlearning done) : {pretty_td(eta_s)} across {NUM_WORKERS} workers "
              f"(~ {eta_when:%Y-%m-%d %H:%M:%S})")
    elif remaining_global > 0:
        print("ETA (all unlearning done) : unknown (no timing yet)")
    else:
        print("Unlearning phase : COMPLETE")

    # ------------------------------------------------------------------ #
    # Per-model progress + parallel chained ETA                            #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Per-top-model unlearning progress")
    print("=" * 60)

    now = datetime.now()

    # ← Each worker processes its assigned models sequentially.
    #   worker_cursors[w] tracks how many seconds worker w still has ahead.
    #   Model i is assigned to worker (i % NUM_WORKERS) — same round-robin
    #   as GPU_Enabled_Combined.py's Phase 2 loop.
    worker_cursors = {w: 0.0 for w in range(NUM_WORKERS)}

    for idx, row in top_configs_df.iterrows():
        t_lr        = row["train_lr"]
        gamma       = row["gamma"]
        hidden_dim  = row["hidden_dim"]
        train_bs    = row["train_batch"]
        this_worker = idx % NUM_WORKERS  # ← which worker owns this model

        mask_prog = (
            (ul_prog_df["t_lr"]         == t_lr)
            & (ul_prog_df["gamma"]      == gamma)
            & (ul_prog_df["hidden_dim"] == hidden_dim)
            & (ul_prog_df["train_bs"]   == train_bs)
        )
        done_model      = int(mask_prog.sum())
        frac_model      = done_model / COMBOS_PER_CFG if COMBOS_PER_CFG > 0 else 0.0
        remaining_model = max(0, COMBOS_PER_CFG - done_model)

        avg_ul_model = np.nan
        if not res_df.empty and "unlearn_time_s" in res_df.columns:
            mask_res = (
                (res_df["train_lr"]      == t_lr)
                & (res_df["gamma"]       == gamma)
                & (res_df["hidden_dim"]  == hidden_dim)
                & (res_df["train_batch"] == train_bs)
            )
            times_m = pd.to_numeric(
                res_df.loc[mask_res, "unlearn_time_s"], errors="coerce"
            ).dropna()
            if not times_m.empty:
                avg_ul_model = times_m.mean() + EVAL_OVERHEAD_S

        eff_avg = avg_ul_model if not np.isnan(avg_ul_model) else avg_ul_global
        src     = "per-model" if not np.isnan(avg_ul_model) else "global"

        model_remaining_sec = (
            remaining_model * eff_avg
            if eff_avg is not None and not np.isnan(eff_avg)
            else None
        )

        # ← ETA = now + how long this worker still has queued before this
        #   model + how long this model itself takes
        if model_remaining_sec is not None:
            eta_done_sec  = worker_cursors[this_worker] + model_remaining_sec
            eta_done_when = now + timedelta(seconds=eta_done_sec)
            eta_done_str  = eta_done_when.strftime("%Y-%m-%d %H:%M:%S")
            eta_rem_str   = pretty_td(eta_done_sec)
        else:
            eta_done_str = eta_rem_str = "unknown"

        print(f"\nModel {idx+1}  [assigned to worker {this_worker}]:")
        print(
            f"  Config   : train_lr={fmt_float(t_lr)}, gamma={fmt_float(gamma)}, "
            f"hidden_dim={int(hidden_dim)}, train_batch={int(train_bs)}"
        )
        print(
            f"  Progress : {done_model}/{COMBOS_PER_CFG} "
            f"({frac_model*100:.1f}%) {make_bar(frac_model)}"
        )

        if remaining_model == 0:
            print("  Status   : COMPLETE")
        elif model_remaining_sec is not None:
            print(
                f"  ETA done : {eta_rem_str} on worker {this_worker} "
                f"(~ {eta_done_str})  [{src} avg + {EVAL_OVERHEAD_S}s eval]"
            )
        else:
            print("  ETA done : unknown (no timing data yet)")

        # ← Advance only this model's worker cursor, not all workers
        if model_remaining_sec is not None:
            worker_cursors[this_worker] += model_remaining_sec

    # Overall wall-clock ETA = when the slowest worker finishes
    max_cursor = max(worker_cursors.values())
    if max_cursor > 0:
        overall_eta = now + timedelta(seconds=max_cursor)
        print(f"\nOverall wall-clock ETA : {pretty_td(max_cursor)} "
              f"(~ {overall_eta:%Y-%m-%d %H:%M:%S})  "
              f"[bottleneck: worker {max(worker_cursors, key=worker_cursors.get)}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
