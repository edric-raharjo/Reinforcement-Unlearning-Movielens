#!/usr/bin/env python
# coding: utf-8
# C:\Bob\tools\sequence_overlap.py

import hashlib
import os
import random
import sys
from pathlib import Path
from itertools import islice

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if len(sys.argv) < 4:
    raise ValueError(
        "Usage: python sequence_overlap.py <forget_pct> <max_retain_drop> <n_gram>\n"
        "Example: python sequence_overlap.py 20 0.09 3\n"
        "  max_retain_drop : upper bound on (base_retain_Hit - retain_Hit) at K=10\n"
        "  n_gram          : compute overlaps for 2-gram up to this n\n"
    )

FORGET_PERCENTAGE = int(sys.argv[1])
MAX_RETAIN_DROP   = float(sys.argv[2])
MAX_N             = int(sys.argv[3])
assert MAX_N >= 2, "n_gram must be >= 2"

METHODS = ["Ye_ApxI", "Ye_multi", "New_True_inf", "New_Max"]
EVAL_K  = 10

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR     = "C:/Bob/ml-1m"
RESULTS_BASE = Path(f"C:/Bob/results/{FORGET_PERCENTAGE}_percent")
MODELS_DIR   = RESULTS_BASE / "models"
ANALYSIS_DIR = RESULTS_BASE / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

for _candidate in [
    RESULTS_BASE / "tuning_full_results.csv",
    RESULTS_BASE / "tuning_full_results_w0.csv",
]:
    if _candidate.exists():
        RESULTS_PATH = _candidate
        break
else:
    raise FileNotFoundError(f"No tuning_full_results*.csv found in {RESULTS_BASE}")

# ---------------------------------------------------------------------------
# Seed + reproducibility (mirrors main script exactly)
# ---------------------------------------------------------------------------
SEED = 97620260313

def make_seed(*args):
    key = f"{SEED}|{'|'.join(str(a) for a in args)}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**31 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed % (2**31 - 1))

set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")

# ---------------------------------------------------------------------------
# Data loading (mirrors main script exactly)
# ---------------------------------------------------------------------------
def load_data(data_dir):
    ratings_df = pd.read_csv(
        os.path.join(data_dir, "ratings.dat"),
        sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies_df = pd.read_csv(
        os.path.join(data_dir, "movies.dat"),
        sep="::", engine="python",
        names=["movie_id", "title", "genres"],
        encoding="ISO-8859-1",
    )
    users_df = pd.read_csv(
        os.path.join(data_dir, "users.dat"),
        sep="::", engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
    )
    return ratings_df, movies_df, users_df

ratings_df, movies_df, users_df = load_data(DATA_DIR)
sample_users  = ratings_df["user_id"].unique().tolist()
pilot_ratings = ratings_df[ratings_df["user_id"].isin(sample_users)].copy()
pilot_ratings.sort_values(["user_id", "timestamp"], inplace=True)

# Feature engineering
pilot_users_df = users_df[users_df["user_id"].isin(sample_users)]
user_cats      = pilot_users_df[["user_id", "gender", "age", "occupation"]].copy()
oh             = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
user_cat_mat   = oh.fit_transform(user_cats[["gender", "age", "occupation"]])
user_feat_df   = pd.DataFrame(user_cat_mat, index=user_cats["user_id"])

all_genres    = sorted({g for s in movies_df["genres"].astype(str) for g in s.split("|")})
genre_to_idx  = {g: i for i, g in enumerate(all_genres)}
num_genres    = len(all_genres)

def movie_genre_vector(genres_str):
    v = np.zeros(num_genres, dtype=np.float32)
    for g in str(genres_str).split("|"):
        if g in genre_to_idx:
            v[genre_to_idx[g]] = 1.0
    return v

movies_df["genre_vec"] = movies_df["genres"].apply(movie_genre_vector)
movie_genre_map = {
    mid: movies_df.loc[movies_df["movie_id"] == mid, "genre_vec"].values[0]
    for mid in movies_df["movie_id"].unique()
}

# ---------------------------------------------------------------------------
# Seeded split (mirrors main script exactly)
# ---------------------------------------------------------------------------
sample_users = np.array(sample_users)
np.random.shuffle(sample_users)   # seeded by set_seed(SEED)

split_amt      = int(np.round(FORGET_PERCENTAGE / 100 * pilot_ratings["user_id"].nunique()))
forget_users   = sample_users[:split_amt]
retain_users   = sample_users[split_amt:]
forget_user_set = set(forget_users.tolist())
retain_user_set = set(retain_users.tolist())

pilot_ratings_all = ratings_df[ratings_df["user_id"].isin(sample_users)].copy()
pilot_ratings_all.sort_values(["user_id", "timestamp"], inplace=True)

trajectories_all = [
    {"user_id": uid, "movies": g["movie_id"].tolist(), "ratings": g["rating"].tolist()}
    for uid, g in pilot_ratings_all.groupby("user_id")
    if len(g) >= 5
]

forget_trajectories = [t for t in trajectories_all if t["user_id"] in forget_user_set]
retain_trajectories  = [t for t in trajectories_all if t["user_id"] in retain_user_set]
candidate_movies     = np.array(sorted(pilot_ratings_all["movie_id"].unique()))

print(f"Forget trajectories : {len(forget_trajectories)}")
print(f"Retain trajectories : {len(retain_trajectories)}")

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
state_dim = user_feat_df.shape[1] + num_genres

def build_state_fn(user_id, watched_movies):
    user_feat = user_feat_df.loc[user_id].values.astype(np.float32)
    pref_vec  = np.zeros(num_genres, dtype=np.float32)
    for mid in watched_movies:
        if mid in movie_genre_map:
            pref_vec += movie_genre_map[mid]
    s = pref_vec.sum()
    if s > 0:
        pref_vec /= s
    return np.concatenate([user_feat, pref_vec]).astype(np.float32)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.fc1    = nn.Linear(state_dim, hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, hidden_dim)
        self.fc3    = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.logits(x)

def load_net(path, hidden_dim):
    net = PolicyNet(state_dim, len(candidate_movies), hidden_dim).to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net

def evaluate_per_trajectory(policy_net, trajectories, K=10):
    results = []
    for traj in trajectories:
        uid    = traj["user_id"]
        movies = traj["movies"]
        if len(movies) < 5:
            continue
        split  = len(movies) // 2
        future = set(movies[split:])
        state  = build_state_fn(uid, movies[:split])
        st     = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(policy_net(st), dim=-1).squeeze(0).cpu().numpy()
        topk = candidate_movies[np.argsort(-probs)[:K]]
        results.append({"user_id": uid, "hit": int(any(m in future for m in topk))})
    return results

# ---------------------------------------------------------------------------
# N-gram helpers
# ---------------------------------------------------------------------------
def get_ngrams(movie_list, n):
    return set(
        tuple(islice(iter(movie_list[i:]), n))
        for i in range(len(movie_list) - n + 1)
    )

def get_all_ngrams_up_to(movie_list, max_n):
    return {n: get_ngrams(movie_list, n) for n in range(2, max_n + 1)}

# ---------------------------------------------------------------------------
# Build forget n-gram index ONCE (shared across all methods)
# ---------------------------------------------------------------------------
print(f"\nBuilding forget n-gram index (prefix only, 2..{MAX_N}-grams)...")

forget_ngrams_by_user = {}
global_forget_ngrams  = {n: set() for n in range(2, MAX_N + 1)}

for traj in forget_trajectories:
    uid    = traj["user_id"]
    prefix = traj["movies"][:len(traj["movies"]) // 2]
    ngs    = get_all_ngrams_up_to(prefix, MAX_N)
    forget_ngrams_by_user[uid] = ngs
    for n in range(2, MAX_N + 1):
        global_forget_ngrams[n].update(ngs[n])

print("  Forget n-gram sizes: " +
      ", ".join(f"{n}-gram={len(global_forget_ngrams[n]):,}" for n in range(2, MAX_N + 1)))

# ---------------------------------------------------------------------------
# Filename helpers (mirrors main script exactly)
# ---------------------------------------------------------------------------
def _fmt(v):
    if v < 0.01:
        return f"{v:.0e}".replace("-", "n").replace("+", "p")
    return str(v).replace(".", "d")

def trained_model_filename(t_lr, gamma, hidden_dim, train_bs):
    return (
        f"trained__tlr{_fmt(t_lr)}__g{_fmt(gamma)}"
        f"__h{hidden_dim}__bs{train_bs}.pt"
    )

def unlearned_model_filename(method, t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam):
    return (
        f"unlearn__{method}__tlr{_fmt(t_lr)}__g{_fmt(gamma)}"
        f"__h{hidden_dim}__bs{train_bs}__ulr{_fmt(u_lr)}"
        f"__ui{u_iters}__lam{_fmt(lam)}.pt"
    )

# ---------------------------------------------------------------------------
# Helper: flip label
# ---------------------------------------------------------------------------
def flip_label(hit_before, hit_after):
    if hit_before == 1 and hit_after == 0:
        return "flipped_harmful"
    if hit_before == 1 and hit_after == 1:
        return "stable_correct"
    if hit_before == 0 and hit_after == 0:
        return "stable_miss"
    return "recovered"

# ---------------------------------------------------------------------------
# Load results CSV once
# ---------------------------------------------------------------------------
print(f"\nLoading results from : {RESULTS_PATH}")
res_df = pd.read_csv(RESULTS_PATH)
k10    = res_df[res_df["K"] == EVAL_K].copy()
k10["retain_drop"] = k10["base_retain_Hit"] - k10["retain_Hit"]
k10["forget_drop"] = k10["base_forget_Hit"] - k10["forget_Hit"]

# ---------------------------------------------------------------------------
# Trained-model eval is the same for every method sharing the same train cfg
# We cache it to avoid re-running evaluate_per_trajectory repeatedly
# ---------------------------------------------------------------------------
trained_eval_cache = {}   # key: (t_lr, gamma, hidden_dim, train_bs) -> DataFrame

# ---------------------------------------------------------------------------
# Main loop — one analysis per method
# ---------------------------------------------------------------------------
all_summaries = []   # collect per-method summary rows for a combined CSV

for method in METHODS:
    print(f"\n{'='*60}")
    print(f"METHOD : {method}")
    print(f"{'='*60}")

    method_dir = ANALYSIS_DIR / method
    method_dir.mkdir(parents=True, exist_ok=True)

    # ── Select best run for this method ─────────────────────────────────
    method_k10 = k10[k10["method"] == method].copy()
    if method_k10.empty:
        print(f"  No results found for method={method} — skipping.")
        continue

    eligible = method_k10[method_k10["retain_drop"] <= MAX_RETAIN_DROP]
    if eligible.empty:
        print(
            f"  No runs satisfy retain_drop <= {MAX_RETAIN_DROP} for {method}.\n"
            f"  Min retain_drop: {method_k10['retain_drop'].min():.4f} — skipping."
        )
        continue

    best = eligible.sort_values("forget_drop", ascending=False).iloc[0]

    t_lr       = best["train_lr"]
    gamma      = best["gamma"]
    hidden_dim = int(best["hidden_dim"])
    train_bs   = int(best["train_batch"])
    u_lr       = best["unlearn_lr"]
    u_iters    = int(best["unlearn_iters"])
    lam        = best["lambda_retain"]

    print(f"  Best run selected:")
    print(f"    train_lr={t_lr}, gamma={gamma}, hidden_dim={hidden_dim}, train_batch={train_bs}")
    print(f"    unlearn_lr={u_lr}, unlearn_iters={u_iters}, lambda={lam}")
    print(f"    retain_drop={best['retain_drop']:.4f}, forget_drop={best['forget_drop']:.4f}")

    # Save selection info
    with open(method_dir / "selected_run_info.txt", "w") as f:
        f.write(f"method            : {method}\n")
        f.write(f"forget_percentage : {FORGET_PERCENTAGE}\n")
        f.write(f"max_retain_drop   : {MAX_RETAIN_DROP}\n")
        f.write(f"n_gram            : {MAX_N}\n\n")
        f.write(f"Selected run\n{'='*40}\n")
        for col in ["train_lr","gamma","hidden_dim","train_batch",
                    "unlearn_lr","unlearn_iters","lambda_retain","method",
                    "retain_drop","forget_drop",
                    "retain_Hit","forget_Hit","base_retain_Hit","base_forget_Hit"]:
            if col in best.index:
                f.write(f"  {col:<22} = {best[col]}\n")

    # ── Load trained model (cached) ──────────────────────────────────────
    train_key     = (t_lr, gamma, hidden_dim, train_bs)
    trained_path  = MODELS_DIR / trained_model_filename(t_lr, gamma, hidden_dim, train_bs)

    if not trained_path.exists():
        print(f"  Trained checkpoint not found: {trained_path.name} — skipping.")
        continue

    # ── Load unlearned model ─────────────────────────────────────────────
    unlearned_path = MODELS_DIR / unlearned_model_filename(
        method, t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam
    )
    if not unlearned_path.exists():
        print(f"  Unlearned checkpoint not found: {unlearned_path.name} — skipping.")
        continue

    print(f"  Loading trained   : {trained_path.name}")
    trained_net = load_net(trained_path, hidden_dim)

    print(f"  Loading unlearned : {unlearned_path.name}")
    unlearned_net = load_net(unlearned_path, hidden_dim)

    # ── Per-trajectory eval ──────────────────────────────────────────────
    if train_key not in trained_eval_cache:
        print("  Evaluating trained model (not cached)...")
        trained_eval_cache[train_key] = pd.DataFrame(
            evaluate_per_trajectory(trained_net, retain_trajectories, K=EVAL_K)
        ).rename(columns={"hit": "hit_before"})
    else:
        print("  Using cached trained model eval.")

    before_df = trained_eval_cache[train_key]

    print("  Evaluating unlearned model...")
    after_df = pd.DataFrame(
        evaluate_per_trajectory(unlearned_net, retain_trajectories, K=EVAL_K)
    ).rename(columns={"hit": "hit_after"})

    eval_df              = before_df.merge(after_df, on="user_id")
    eval_df["flip_label"] = eval_df.apply(
        lambda r: flip_label(r["hit_before"], r["hit_after"]), axis=1
    )
    eval_df.to_csv(method_dir / "retain_eval_per_traj.csv", index=False)

    print("  Flip label distribution:")
    print("  " + eval_df["flip_label"].value_counts().to_string().replace("\n", "\n  "))

    # ── N-gram overlap per retain trajectory ────────────────────────────
    overlap_rows = []

    for traj in retain_trajectories:
        uid    = traj["user_id"]
        movies = traj["movies"]
        split  = len(movies) // 2
        prefix = movies[:split]
        if len(prefix) < 2:
            continue

        retain_ngs = get_all_ngrams_up_to(prefix, MAX_N)
        row        = {"user_id": uid}

        for n in range(2, MAX_N + 1):
            r_set = retain_ngs[n]
            if len(r_set) == 0:
                row[f"overlap_{n}gram_pct"] = np.nan
            else:
                shared = len(r_set & global_forget_ngrams[n])
                row[f"overlap_{n}gram_pct"] = round(shared / len(r_set) * 100, 4)

        # Nearest forget neighbors by MAX_N-gram
        retain_main    = retain_ngs[MAX_N]
        per_user_overlaps = []
        for f_uid, f_ngs in forget_ngrams_by_user.items():
            f_set = f_ngs[MAX_N]
            if not retain_main or not f_set:
                continue
            pct = len(retain_main & f_set) / len(retain_main) * 100
            if pct > 0:
                per_user_overlaps.append((f_uid, pct))

        per_user_overlaps.sort(key=lambda x: -x[1])

        row["max_single_forget_overlap_pct"]          = round(per_user_overlaps[0][1], 4) if per_user_overlaps else 0.0
        row[f"num_forget_users_sharing_{MAX_N}gram"]  = len(per_user_overlaps)

        for rank, (f_uid, f_pct) in enumerate(per_user_overlaps[:3], start=1):
            row[f"neighbor_{rank}_user_id"]      = f_uid
            row[f"neighbor_{rank}_overlap_pct"]  = round(f_pct, 4)

        overlap_rows.append(row)

    overlap_df = pd.DataFrame(overlap_rows).merge(
        eval_df[["user_id", "hit_before", "hit_after", "flip_label"]],
        on="user_id", how="left"
    )
    overlap_df.to_csv(method_dir / "ngram_overlap.csv", index=False)

    # ── Flipped vs stable summary ────────────────────────────────────────
    overlap_cols = (
        [f"overlap_{n}gram_pct" for n in range(2, MAX_N + 1)] +
        ["max_single_forget_overlap_pct", f"num_forget_users_sharing_{MAX_N}gram"]
    )
    summary = (
        overlap_df.groupby("flip_label")[overlap_cols]
        .agg(["mean", "median", "std"])
        .round(4)
    )
    summary.to_csv(method_dir / "flipped_vs_stable_summary.csv")

    # ── Nearest forget neighbors for flipped cases ──────────────────────
    flipped_df   = overlap_df[overlap_df["flip_label"] == "flipped_harmful"].copy()
    neighbor_cols = (
        ["user_id", "flip_label"] +
        [c for c in overlap_df.columns if "neighbor_" in c or "overlap" in c or "num_forget" in c]
    )
    flipped_df[[c for c in neighbor_cols if c in flipped_df.columns]].to_csv(
        method_dir / "nearest_forget_neighbors.csv", index=False
    )

    print(f"  Saved all outputs → {method_dir}")

    # ── Collect into combined summary ────────────────────────────────────
    for flip_lbl in ["flipped_harmful", "stable_correct", "stable_miss", "recovered"]:
        subset = overlap_df[overlap_df["flip_label"] == flip_lbl]
        if subset.empty:
            continue
        row = {"method": method, "flip_label": flip_lbl, "n": len(subset)}
        for col in overlap_cols:
            row[f"{col}_mean"]   = round(subset[col].mean(), 4)
            row[f"{col}_median"] = round(subset[col].median(), 4)
        all_summaries.append(row)

    # Free GPU memory between methods
    del trained_net, unlearned_net
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Combined summary across all methods
# ---------------------------------------------------------------------------
if all_summaries:
    combined_df = pd.DataFrame(all_summaries)
    combined_df.to_csv(ANALYSIS_DIR / "all_methods_summary.csv", index=False)
    print(f"\n✓ Combined summary saved → {ANALYSIS_DIR / 'all_methods_summary.csv'}")

print(f"\n✓ All analysis complete. Results in: {ANALYSIS_DIR}")
