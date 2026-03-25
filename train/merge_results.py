import os
import sys
import glob
import pandas as pd

FORGET_PERCENTAGE = int(sys.argv[1])
# FIX: Make the DEMO argument optional so it doesn't crash if omitted
DEMO = int(sys.argv[2]) 
DEMO = (DEMO == 2)
if DEMO:
      RESULTS_BASE = f"D:/Bob_Skripsi_Do Not Delete/results_demography/{FORGET_PERCENTAGE}_percent"
else:
      RESULTS_BASE = f"D:/Bob_Skripsi_Do Not Delete/results/{FORGET_PERCENTAGE}_percent"

_TRAIN_KEY_COLS = ["train_lr", "gamma", "hidden_dim", "train_batch", "K"]
_KEY_COLS = ["train_lr", "gamma", "hidden_dim", "train_batch",
             "unlearn_lr", "unlearn_iters", "lambda_retain", "method", "K"]

def merge(pattern, key_cols, out_path):
    files = glob.glob(os.path.join(RESULTS_BASE, pattern))
    if not files:
        print(f"  No files matched: {pattern}")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=key_cols, keep="last")
    df.to_csv(out_path, index=False)
    print(f"  Merged {len(files)} files → {out_path} ({len(df):,} rows)")

print("Merging Phase 1 results...")
merge("train_phase_results_w*.csv", _TRAIN_KEY_COLS,
      os.path.join(RESULTS_BASE, "train_phase_results.csv"))

print("Merging Phase 2 results...")
merge("tuning_full_results_w*.csv", _KEY_COLS,
      os.path.join(RESULTS_BASE, "tuning_full_results.csv"))

print("Merging unlearn loss logs...")
merge("unlearning_loss_log_w*.csv",
      ["train_lr","gamma","hidden_dim","train_batch","unlearn_lr","unlearn_iters","lambda_retain","method","iter"],
      os.path.join(RESULTS_BASE, "unlearning_loss_log.csv"))

print("Done.")
