#!/usr/bin/env python
# coding: utf-8
# C:\Bob\tools\margin_analysis.py

import hashlib
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from scipy import stats

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if len(sys.argv) < 3:
    raise ValueError(
        "Usage: python margin_analysis.py <forget_pct> <max_retain_drop>\n"
        "Example: python margin_analysis.py 20 0.09\n"
    )

FORGET_PERCENTAGE = int(sys.argv[1])
MAX_RETAIN_DROP   = float(sys.argv[2])

METHODS     = ["Ye_ApxI", "Ye_multi", "New_True_inf", "New_Max"]
FLIP_LABELS = ["flipped_harmful", "stable_correct", "stable_miss", "recovered"]
EVAL_K      = 10

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR     = "C:/Bob/ml-1m"
RESULTS_BASE = Path(f"C:/Bob/results/{FORGET_PERCENTAGE}_percent")
MODELS_DIR   = RESULTS_BASE / "models"
ANALYSIS_DIR = RESULTS_BASE / "analysis"

TRAIN_RESULTS_MERGED = RESULTS_BASE / "train_phase_results.csv"
RESULTS_MERGED       = RESULTS_BASE / "tuning_full_results.csv"

for p in [TRAIN_RESULTS_MERGED, RESULTS_MERGED]:
    if not p.exists():
        raise FileNotFoundError(p)

# ---------------------------------------------------------------------------
# Seed + determinism
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
torch.use_deterministic_algorithms(True, warn_only=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")

# ---------------------------------------------------------------------------
# Data loading — mirrors main script exactly
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

pilot_users_df = users_df[users_df["user_id"].isin(sample_users)]
user_cats      = pilot_users_df[["user_id", "gender", "age", "occupation"]].copy()
oh             = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
user_feat_df   = pd.DataFrame(
    oh.fit_transform(user_cats[["gender", "age", "occupation"]]),
    index=user_cats["user_id"],
)

all_genres   = sorted({g for s in movies_df["genres"].astype(str) for g in s.split("|")})
genre_to_idx = {g: i for i, g in enumerate(all_genres)}
num_genres   = len(all_genres)

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

sample_users = np.array(sample_users)
np.random.shuffle(sample_users)
split_amt    = int(np.round(FORGET_PERCENTAGE / 100 * pilot_ratings["user_id"].nunique()))
forget_users = sample_users[:split_amt]
retain_users = sample_users[split_amt:]

pilot_ratings_all = ratings_df[ratings_df["user_id"].isin(sample_users)].copy()
pilot_ratings_all.sort_values(["user_id", "timestamp"], inplace=True)

trajectories_all    = [
    {"user_id": uid, "movies": g["movie_id"].tolist(), "ratings": g["rating"].tolist()}
    for uid, g in pilot_ratings_all.groupby("user_id")
    if len(g) >= 5
]
candidate_movies    = np.array(sorted(pilot_ratings_all["movie_id"].unique()))
forget_user_set     = set(forget_users.tolist())
retain_user_set     = set(retain_users.tolist())
forget_trajectories = [t for t in trajectories_all if t["user_id"] in forget_user_set]
retain_trajectories  = [t for t in trajectories_all if t["user_id"] in retain_user_set]

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

print(f"Retain trajectories : {len(retain_trajectories)}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------
def _fmt(v):
    if v < 0.01:
        return f"{v:.0e}".replace("-", "n").replace("+", "p")
    return str(v).replace(".", "d")

def trained_model_path(t_lr, gamma, hidden_dim, train_bs):
    return MODELS_DIR / (
        f"trained__tlr{_fmt(t_lr)}__g{_fmt(gamma)}"
        f"__h{hidden_dim}__bs{train_bs}.pt"
    )

def unlearned_model_path(t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam):
    return MODELS_DIR / (
        f"unlearn__{method}__tlr{_fmt(t_lr)}__g{_fmt(gamma)}"
        f"__h{hidden_dim}__bs{train_bs}__ulr{_fmt(u_lr)}"
        f"__ui{u_iters}__lam{_fmt(lam)}.pt"
    )

def flip_label(hit_before, hit_after):
    if hit_before == 1 and hit_after == 0: return "flipped_harmful"
    if hit_before == 1 and hit_after == 1: return "stable_correct"
    if hit_before == 0 and hit_after == 0: return "stable_miss"
    return "recovered"

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
_TRAIN_KEY = ["train_lr", "gamma", "hidden_dim", "train_batch", "K"]
_KEY_COLS  = ["train_lr", "gamma", "hidden_dim", "train_batch",
              "unlearn_lr", "unlearn_iters", "lambda_retain", "method", "K"]

train_res_df = pd.read_csv(TRAIN_RESULTS_MERGED).drop_duplicates(subset=_TRAIN_KEY, keep="last")
res_df       = pd.read_csv(RESULTS_MERGED).drop_duplicates(subset=_KEY_COLS, keep="last")

k10 = res_df[res_df["K"] == EVAL_K].copy()
k10["retain_drop"] = k10["base_retain_Hit"] - k10["retain_Hit"]
k10["forget_drop"] = k10["base_forget_Hit"] - k10["forget_Hit"]

# ---------------------------------------------------------------------------
# ===========================================================================
#
#   MARGIN METRICS — all computed from the trained model before unlearning.
#
#   Each metric is a scalar per retain user that attempts to capture
#   "how precarious was this user's correct recommendation?"
#
#   To add a new metric:
#     1. Add a function compute_<name>(probs, future, candidate_movies) -> float
#     2. Add its key to METRIC_REGISTRY below
#     3. Done — the loop handles the rest automatically
#
# ===========================================================================

def compute_top1_prob(probs, future, candidates):
    """
    Probability assigned to the single highest-ranked movie.
    Low = model was uncertain overall.
    """
    return float(probs.max())

def compute_hit_prob_sum(probs, future, candidates):
    """
    Total probability mass assigned to all future (relevant) movies.
    Low = relevant items buried deep in the ranking.
    """
    relevant_mask = np.isin(candidates, list(future))
    return float(probs[relevant_mask].sum())

def compute_hit_prob_max(probs, future, candidates):
    """
    Highest probability assigned to any single relevant movie.
    The 'confidence' of the best correct recommendation.
    """
    relevant_mask = np.isin(candidates, list(future))
    rel_probs = probs[relevant_mask]
    return float(rel_probs.max()) if rel_probs.size > 0 else 0.0

def compute_rank_of_best_hit(probs, future, candidates):
    """
    The rank (1-indexed) of the highest-probability relevant movie.
    Lower rank = better. A user with rank=1 would be flipped if that
    movie falls out of top-K; a user with rank=10 is already at the edge.
    """
    sorted_idx = np.argsort(-probs, kind="stable")
    for rank, idx in enumerate(sorted_idx):
        if candidates[idx] in future:
            return float(rank + 1)
    return float(len(candidates))

def compute_margin_best_hit_vs_k1(probs, future, candidates):
    """
    KEY METRIC — The core margin hypothesis.

    prob(best_relevant_movie) - prob(movie ranked K+1)

    A small or negative margin means the best relevant movie is barely
    inside the top-K. Any small weight perturbation can flip it out.

    This is specifically defined relative to EVAL_K.
    """
    sorted_idx  = np.argsort(-probs, kind="stable")
    topk_set    = set(candidates[sorted_idx[:EVAL_K]])
    kplus1_prob = float(probs[sorted_idx[EVAL_K]]) if len(sorted_idx) > EVAL_K else 0.0

    best_hit_prob = 0.0
    for idx in sorted_idx:
        if candidates[idx] in future:
            best_hit_prob = float(probs[idx])
            break

    return best_hit_prob - kplus1_prob

def compute_margin_worst_hit_vs_k1(probs, future, candidates):
    """
    prob(worst relevant movie inside top-K) - prob(movie ranked K+1)

    The minimum margin of any correct recommendation that made it
    into top-K. Even closer to zero = more precarious.
    """
    sorted_idx   = np.argsort(-probs, kind="stable")
    kplus1_prob  = float(probs[sorted_idx[EVAL_K]]) if len(sorted_idx) > EVAL_K else 0.0
    topk_hits    = [probs[idx] for idx in sorted_idx[:EVAL_K]
                    if candidates[idx] in future]
    if not topk_hits:
        # User was already a miss — margin is meaningless, return NaN
        return float("nan")
    return float(min(topk_hits)) - kplus1_prob

def compute_entropy(probs, future, candidates):
    """
    Entropy of the full recommendation distribution.
    High entropy = model is uncertain across many movies.
    A very uncertain model may be more susceptible to weight changes.
    """
    clipped = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))

def compute_topk_prob_mass(probs, future, candidates):
    """
    Total probability mass in the top-K recommendations.
    Low = model is spreading probability widely (uncertain).
    """
    sorted_idx = np.argsort(-probs, kind="stable")
    return float(probs[sorted_idx[:EVAL_K]].sum())

# Registry — add new metric functions here
METRIC_REGISTRY = {
    "top1_prob":               compute_top1_prob,
    "hit_prob_sum":            compute_hit_prob_sum,
    "hit_prob_max":            compute_hit_prob_max,
    "rank_of_best_hit":        compute_rank_of_best_hit,
    "margin_best_hit_vs_k1":   compute_margin_best_hit_vs_k1,   # ← KEY
    "margin_worst_hit_vs_k1":  compute_margin_worst_hit_vs_k1,  # ← KEY
    "entropy":                 compute_entropy,
    "topk_prob_mass":          compute_topk_prob_mass,
}

# ===========================================================================
# END METRIC DEFINITION ZONE
# ===========================================================================

# ---------------------------------------------------------------------------
# Core per-trajectory margin computation
# ---------------------------------------------------------------------------
def compute_all_margins(net, trajectories):
    """
    For every retain trajectory, compute all metrics in METRIC_REGISTRY.
    Returns a DataFrame with one row per user.
    """
    net.eval()
    rows = []
    with torch.no_grad():
        for traj in trajectories:
            uid    = traj["user_id"]
            movies = traj["movies"]
            if len(movies) < 5:
                continue
            split  = len(movies) // 2
            future = set(movies[split:])
            state  = build_state_fn(uid, movies[:split])
            st     = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            probs  = F.softmax(net(st), dim=-1).squeeze(0).cpu().numpy()

            row = {"user_id": uid}
            for metric_name, fn in METRIC_REGISTRY.items():
                row[metric_name] = fn(probs, future, candidate_movies)
            rows.append(row)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Flip labels from trained + unlearned model pair
# ---------------------------------------------------------------------------
def get_flip_labels(trained_net, unlearned_net, trajectories, K=10):
    rows = []
    trained_net.eval()
    unlearned_net.eval()
    with torch.no_grad():
        for traj in trajectories:
            uid    = traj["user_id"]
            movies = traj["movies"]
            if len(movies) < 5:
                continue
            split  = len(movies) // 2
            future = set(movies[split:])
            state  = build_state_fn(uid, movies[:split])
            st     = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            p_before = F.softmax(trained_net(st),   dim=-1).squeeze(0).cpu().numpy()
            p_after  = F.softmax(unlearned_net(st), dim=-1).squeeze(0).cpu().numpy()

            topk_b = candidate_movies[np.argsort(-p_before, kind="stable")[:K]]
            topk_a = candidate_movies[np.argsort(-p_after,  kind="stable")[:K]]

            hit_b  = int(any(m in future for m in topk_b))
            hit_a  = int(any(m in future for m in topk_a))
            rows.append({
                "user_id":    uid,
                "hit_before": hit_b,
                "hit_after":  hit_a,
                "flip_label": flip_label(hit_b, hit_a),
            })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Statistical tests for flipped_harmful vs stable_correct
# ---------------------------------------------------------------------------
def run_stats(df, metric_col):
    """
    For the KEY comparison: flipped_harmful vs stable_correct.

    Returns a dict with:
      - mean/median/std per flip label
      - Mann-Whitney U test p-value (non-parametric, no normality assumption)
      - Cohen's d effect size
      - direction flag
    """
    groups = {}
    for lbl in FLIP_LABELS:
        sub = df[df["flip_label"] == lbl][metric_col].dropna()
        groups[lbl] = sub.values

    result = {}

    # Descriptive stats for all groups
    for lbl in FLIP_LABELS:
        g = groups[lbl]
        result[f"{lbl}_mean"]   = round(float(np.mean(g)),   6) if len(g) else float("nan")
        result[f"{lbl}_median"] = round(float(np.median(g)), 6) if len(g) else float("nan")
        result[f"{lbl}_std"]    = round(float(np.std(g)),    6) if len(g) else float("nan")
        result[f"{lbl}_n"]      = len(g)

    # Mann-Whitney U: flipped_harmful vs stable_correct
    fh = groups["flipped_harmful"]
    sc = groups["stable_correct"]

    if len(fh) >= 5 and len(sc) >= 5:
        stat, pval = stats.mannwhitneyu(fh, sc, alternative="two-sided")
        result["mwu_stat"]   = round(float(stat), 2)
        result["mwu_pvalue"] = float(pval)
        result["significant_005"] = pval < 0.05
        result["significant_001"] = pval < 0.01

        # Cohen's d
        pooled_std = np.sqrt((np.std(fh)**2 + np.std(sc)**2) / 2) + 1e-10
        cohens_d   = (np.mean(fh) - np.mean(sc)) / pooled_std
        result["cohens_d"] = round(float(cohens_d), 4)

        # Direction: negative d = flipped_harmful has LOWER margin (hypothesis confirmed)
        result["hypothesis_direction"] = (
            "✅ confirmed (flipped_harmful < stable_correct)"
            if cohens_d < 0 else
            "❌ reversed  (flipped_harmful > stable_correct)"
        )
    else:
        result.update({
            "mwu_stat": float("nan"), "mwu_pvalue": float("nan"),
            "significant_005": False,  "significant_001": False,
            "cohens_d": float("nan"),  "hypothesis_direction": "insufficient data",
        })

    return result

# ---------------------------------------------------------------------------
# Summary table — same format as n-gram / latent analyses
# ---------------------------------------------------------------------------
def summarise(df, metric_cols):
    return (
        df.groupby("flip_label")[metric_cols]
        .agg(["mean", "median", "std"])
        .round(6)
        .reindex(FLIP_LABELS)
    )

# ===========================================================================
# MAIN LOOP — one analysis per method
# ===========================================================================
# Cache trained-model margins — method-independent, expensive to recompute
margin_cache = {}   # key: (t_lr, gamma, hidden_dim, train_bs) -> DataFrame

all_method_stats = []

for method in METHODS:
    print(f"\n{'='*60}\nMETHOD : {method}\n{'='*60}")

    method_dir = ANALYSIS_DIR / method / "margin"
    method_dir.mkdir(parents=True, exist_ok=True)

    # ── Select best run ────────────────────────────────────────────────
    eligible = k10[(k10["method"] == method) & (k10["retain_drop"] <= MAX_RETAIN_DROP)]
    if eligible.empty:
        print(f"  No eligible runs — skipping")
        continue

    best       = eligible.sort_values("forget_drop", ascending=False).iloc[0]
    t_lr       = best["train_lr"];    gamma      = best["gamma"]
    hidden_dim = int(best["hidden_dim"]); train_bs = int(best["train_batch"])
    u_lr       = best["unlearn_lr"];  u_iters    = int(best["unlearn_iters"])
    lam        = best["lambda_retain"]

    t_path  = trained_model_path(t_lr, gamma, hidden_dim, train_bs)
    ul_path = unlearned_model_path(t_lr, gamma, hidden_dim, train_bs,
                                   method, u_lr, u_iters, lam)

    missing = [label for p, label in [(t_path,"trained"), (ul_path,"unlearned")]
               if not p.exists()]
    if missing:
        print(f"  Missing checkpoints: {missing} — skipping")
        continue

    print(f"  tlr={t_lr} g={gamma} h={hidden_dim} bs={train_bs} "
          f"ulr={u_lr} ui={u_iters} lam={lam}")

    # ── Load models ────────────────────────────────────────────────────
    trained_net = PolicyNet(state_dim, len(candidate_movies), hidden_dim).to(DEVICE)
    trained_net.load_state_dict(torch.load(t_path,  map_location=DEVICE))
    trained_net.eval()

    unlearned_net = PolicyNet(state_dim, len(candidate_movies), hidden_dim).to(DEVICE)
    unlearned_net.load_state_dict(torch.load(ul_path, map_location=DEVICE))
    unlearned_net.eval()

    # ── Step 1: Compute margins from trained model (cached) ───────────
    train_key = (t_lr, gamma, hidden_dim, train_bs)
    if train_key not in margin_cache:
        print("  Computing pre-unlearning margins (all metrics)...")
        margin_cache[train_key] = compute_all_margins(trained_net, retain_trajectories)
        print(f"  Done — {len(margin_cache[train_key])} users")
    else:
        print("  Using cached margins.")
    margin_df = margin_cache[train_key].copy()

    # ── Step 2: Flip labels from THIS method's unlearned model ────────
    print("  Computing flip labels...")
    flip_df = get_flip_labels(trained_net, unlearned_net, retain_trajectories, K=EVAL_K)

    print("  Flip label distribution:")
    print("  " + flip_df["flip_label"].value_counts().to_string().replace("\n", "\n  "))

    # ── Step 3: Merge ─────────────────────────────────────────────────
    combined = margin_df.merge(
        flip_df[["user_id", "hit_before", "hit_after", "flip_label"]],
        on="user_id", how="left"
    )
    combined.to_csv(method_dir / "margin_per_user.csv", index=False)

    # ── Step 4: Summary table ─────────────────────────────────────────
    metric_cols = list(METRIC_REGISTRY.keys())
    summary_df  = summarise(combined, metric_cols)
    summary_df.to_csv(method_dir / "margin_summary.csv")

    print(f"\n  Margin summary:")
    print("  " + summary_df.to_string().replace("\n", "\n  "))

    # ── Step 5: Statistical tests — every metric, key comparison ─────
    print(f"\n  Statistical tests (flipped_harmful vs stable_correct):")
    stats_rows = []
    for metric in metric_cols:
        s = run_stats(combined, metric)
        s["method"] = method
        s["metric"] = metric
        stats_rows.append(s)

        d_str  = f"{s.get('cohens_d', float('nan')):+.4f}"
        p_str  = f"{s.get('mwu_pvalue', float('nan')):.4e}"
        sig    = "**" if s.get("significant_001") else ("*" if s.get("significant_005") else "  ")
        dirn   = s.get("hypothesis_direction", "")
        print(f"    {metric:<30} d={d_str}  p={p_str} {sig}  {dirn}")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(method_dir / "margin_stats.csv", index=False)
    all_method_stats.append(stats_df)

    # ── Step 6: Focused view — only hit users (before=1) ─────────────
    # margin_worst_hit_vs_k1 is NaN for stable_miss / recovered users
    # (they had no hit before unlearning, so margin is undefined).
    # This focused table shows only users who were hitting before — i.e.
    # the REAL comparison: flipped_harmful (hit→miss) vs stable_correct (hit→hit)
    hit_users = combined[combined["hit_before"] == 1].copy()
    hit_summary = summarise(hit_users, metric_cols)
    hit_summary.to_csv(method_dir / "margin_summary_hit_users_only.csv")

    print(f"\n  Summary (hit users only — flipped_harmful vs stable_correct):")
    # Show just the two key margin metrics for brevity
    key_metrics = ["margin_best_hit_vs_k1", "margin_worst_hit_vs_k1",
                   "rank_of_best_hit", "hit_prob_max", "entropy"]
    key_cols = [(m, s) for m in key_metrics for s in ["mean", "median", "std"]]
    available = [c for c in key_cols if c in hit_summary.columns]
    print("  " + hit_summary[available].to_string().replace("\n", "\n  "))

    del trained_net, unlearned_net
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Combined stats across all methods
# ---------------------------------------------------------------------------
if all_method_stats:
    combined_stats = pd.concat(all_method_stats, ignore_index=True)
    combined_stats.to_csv(ANALYSIS_DIR / "all_methods_margin_stats.csv", index=False)

    print(f"\n{'='*60}")
    print("CROSS-METHOD SUMMARY — margin_best_hit_vs_k1")
    print(f"{'='*60}")
    key = combined_stats[combined_stats["metric"] == "margin_best_hit_vs_k1"][
        ["method", "flipped_harmful_mean", "stable_correct_mean",
         "cohens_d", "mwu_pvalue", "significant_005", "hypothesis_direction"]
    ]
    print(key.to_string(index=False))

    print(f"\nCROSS-METHOD SUMMARY — margin_worst_hit_vs_k1")
    key2 = combined_stats[combined_stats["metric"] == "margin_worst_hit_vs_k1"][
        ["method", "flipped_harmful_mean", "stable_correct_mean",
         "cohens_d", "mwu_pvalue", "significant_005", "hypothesis_direction"]
    ]
    print(key2.to_string(index=False))

    print(f"\n✓ All stats saved → {ANALYSIS_DIR / 'all_methods_margin_stats.csv'}")

print(f"\n✓ Margin analysis complete. Results in: {ANALYSIS_DIR}")
