#!/usr/bin/env python
# coding: utf-8
# C:\Bob\tools\latent_analysis.py

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

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if len(sys.argv) < 3:
    raise ValueError(
        "Usage: python latent_analysis.py <forget_pct> <max_retain_drop>\n"
        "Example: python latent_analysis.py 20 0.09\n"
    )

FORGET_PERCENTAGE = int(sys.argv[1])
MAX_RETAIN_DROP   = float(sys.argv[2])

METHODS    = ["Ye_ApxI", "Ye_multi", "New_True_inf", "New_Max"]
FLIP_LABELS = ["flipped_harmful", "stable_correct", "stable_miss", "recovered"]
EVAL_K     = 10

# ---------------------------------------------------------------------------
# Paths — always read merged files
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
# Seed + determinism — mirrors GPU_Enabled_Combine.py exactly
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

# Feature engineering
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

# Seeded split — mirrors main script exactly
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

print(f"Forget trajectories : {len(forget_trajectories)}")
print(f"Retain trajectories  : {len(retain_trajectories)}")

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
# Activation extractor — hooks into fc2 and fc3 simultaneously
# ---------------------------------------------------------------------------
def extract_activations(net, trajectories, layers=("fc2", "fc3")):
    """
    Returns dict: {layer_name: np.ndarray of shape (N_users, hidden_dim)}
    plus a list of user_ids in the same order.

    Uses forward hooks so we never touch the model internals.
    Adding more layers = just extend the `layers` tuple.
    """
    net.eval()
    handles    = []
    captured   = {l: [] for l in layers}

    def make_hook(layer_name):
        def hook(module, input, output):
            # output is post-ReLU already (ReLU is applied inside forward())
            # We want the pre-activation here — the raw fc output before ReLU.
            # But since forward() calls F.relu(self.fcN(x)) inline,
            # the hook on fcN captures the LINEAR output (before ReLU).
            # That's actually BETTER for cosine sim — ReLU zeros kill direction.
            captured[layer_name].append(output.detach().cpu().numpy())
        return hook

    for layer_name in layers:
        module = getattr(net, layer_name)
        handles.append(module.register_forward_hook(make_hook(layer_name)))

    user_ids = []
    with torch.no_grad():
        for traj in trajectories:
            uid    = traj["user_id"]
            movies = traj["movies"]
            if len(movies) < 5:
                continue
            split  = len(movies) // 2
            state  = build_state_fn(uid, movies[:split])
            st     = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            net(st)
            user_ids.append(uid)

    for h in handles:
        h.remove()

    result = {
        layer: np.vstack(captured[layer])   # shape: (N, hidden_dim)
        for layer in layers
    }
    return user_ids, result

# ---------------------------------------------------------------------------
# ===========================================================================
#
#   ████████╗ ██████╗ ██╗   ██╗ ██████╗██╗  ██╗    ██████╗
#      ██╔══╝██╔═══██╗██║   ██║██╔════╝██║  ██║    ╚════██╗
#      ██║   ██║   ██║██║   ██║██║     ███████║      ▄███╔╝
#      ██║   ██║   ██║██║   ██║██║     ██╔══██║      ▀▀══╝
#      ██║   ╚██████╔╝╚██████╔╝╚██████╗██║  ██║    ██╗
#      ╚═╝    ╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝    ╚═╝
#
#   CHOICE 2 — What represents the "Forget Set"?
#
#   To swap strategies, change ONLY this class.
#   The rest of the script never calls forget activations directly —
#   it only calls `similarity_fn(retain_vec)` which returns a scalar.
#
#   Current: Option C — Nearest Forget Neighbor (max cosine similarity)
#   Alternatives are provided below as commented-out classes.
#
# ===========================================================================

class ForgetSetSimilarity:
    """
    Option C — Nearest Forget Neighbor.

    For each retain user vector, returns the MAX cosine similarity
    over all forget user vectors.

    Hypothesis: flipped_harmful users will have a higher max cosine
    similarity to the forget set than stable_correct users.
    """

    name = "nearest_forget_neighbor_cosine"

    def fit(self, forget_vecs: np.ndarray):
        """
        Precompute L2-normalised forget matrix for fast batch cosine sim.
        forget_vecs : shape (N_forget, hidden_dim)
        """
        norms = np.linalg.norm(forget_vecs, axis=1, keepdims=True) + 1e-10
        self.forget_normed = forget_vecs / norms          # (N_forget, D)

    def score(self, retain_vec: np.ndarray) -> float:
        """
        retain_vec : shape (hidden_dim,)
        Returns   : float — max cosine similarity to any forget user
        """
        norm = np.linalg.norm(retain_vec) + 1e-10
        r    = retain_vec / norm                          # (D,)
        sims = self.forget_normed @ r                     # (N_forget,)
        return float(sims.max())

    def score_batch(self, retain_vecs: np.ndarray) -> np.ndarray:
        """
        Vectorised version for the whole retain set at once.
        retain_vecs : shape (N_retain, D)
        Returns     : shape (N_retain,)
        """
        norms = np.linalg.norm(retain_vecs, axis=1, keepdims=True) + 1e-10
        R     = retain_vecs / norms                       # (N_retain, D)
        sims  = R @ self.forget_normed.T                  # (N_retain, N_forget)
        return sims.max(axis=1)                           # (N_retain,)


# ---------------------------------------------------------------------------
# ALTERNATIVE — uncomment to swap to Option A (forget centroid)
# ---------------------------------------------------------------------------
# class ForgetSetSimilarity:
#     """
#     Option A — Forget Centroid.
#     Returns cosine similarity to the mean of all forget activations.
#     """
#     name = "centroid_cosine"
#
#     def fit(self, forget_vecs: np.ndarray):
#         centroid = forget_vecs.mean(axis=0)
#         norm     = np.linalg.norm(centroid) + 1e-10
#         self.centroid_normed = centroid / norm       # (D,)
#
#     def score_batch(self, retain_vecs: np.ndarray) -> np.ndarray:
#         norms = np.linalg.norm(retain_vecs, axis=1, keepdims=True) + 1e-10
#         R     = retain_vecs / norms
#         return (R @ self.centroid_normed)            # (N_retain,)

# ---------------------------------------------------------------------------
# ALTERNATIVE — uncomment to swap to Option B (mean cosine to all forget)
# ---------------------------------------------------------------------------
# class ForgetSetSimilarity:
#     """
#     Option B — Mean Cosine to All Forget Users.
#     Returns the average cosine similarity across all forget users.
#     """
#     name = "mean_forget_cosine"
#
#     def fit(self, forget_vecs: np.ndarray):
#         norms = np.linalg.norm(forget_vecs, axis=1, keepdims=True) + 1e-10
#         self.forget_normed = forget_vecs / norms     # (N_forget, D)
#
#     def score_batch(self, retain_vecs: np.ndarray) -> np.ndarray:
#         norms = np.linalg.norm(retain_vecs, axis=1, keepdims=True) + 1e-10
#         R     = retain_vecs / norms
#         sims  = R @ self.forget_normed.T             # (N_retain, N_forget)
#         return sims.mean(axis=1)                     # (N_retain,)

# ===========================================================================
# END TOUCH ZONE
# ===========================================================================

# ---------------------------------------------------------------------------
# Filename helpers — mirrors main script exactly
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
# Load best run per method — same logic as sequence_overlap.py
# ---------------------------------------------------------------------------
_TRAIN_KEY = ["train_lr", "gamma", "hidden_dim", "train_batch", "K"]
_KEY_COLS  = ["train_lr","gamma","hidden_dim","train_batch",
              "unlearn_lr","unlearn_iters","lambda_retain","method","K"]

train_res_df = pd.read_csv(TRAIN_RESULTS_MERGED).drop_duplicates(subset=_TRAIN_KEY, keep="last")
res_df       = pd.read_csv(RESULTS_MERGED).drop_duplicates(subset=_KEY_COLS, keep="last")

k10 = res_df[res_df["K"] == EVAL_K].copy()
k10["retain_drop"] = k10["base_retain_Hit"] - k10["retain_Hit"]
k10["forget_drop"] = k10["base_forget_Hit"] - k10["forget_Hit"]

# ---------------------------------------------------------------------------
# Evaluate retain trajectories with a given model (for flip labels)
# ---------------------------------------------------------------------------
def get_flip_labels(trained_net, unlearned_net, trajectories, K=10):
    rows = []
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
            p_before = F.softmax(trained_net(st),   dim=-1).squeeze(0).cpu().numpy()
            p_after  = F.softmax(unlearned_net(st), dim=-1).squeeze(0).cpu().numpy()

        topk_before = candidate_movies[np.argsort(-p_before, kind="stable")[:K]]
        topk_after  = candidate_movies[np.argsort(-p_after,  kind="stable")[:K]]

        hit_before = int(any(m in future for m in topk_before))
        hit_after  = int(any(m in future for m in topk_after))
        rows.append({"user_id": uid, "hit_before": hit_before,
                     "hit_after": hit_after,
                     "flip_label": flip_label(hit_before, hit_after)})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Summary table helper — same format as sequence_overlap.py
# ---------------------------------------------------------------------------
def summarise(df, metric_col):
    return (
        df.groupby("flip_label")[metric_col]
        .agg(["mean", "median", "std"])
        .round(6)
        .reindex(FLIP_LABELS)
    )

# ===========================================================================
# MAIN LOOP — one analysis per method
# ===========================================================================
sim_fn     = ForgetSetSimilarity()
all_layers = ("fc2", "fc3")

# We cache trained-model activations since they are method-independent
trained_acts_cache = {}   # key: (t_lr, gamma, hidden_dim, train_bs) -> (user_ids, acts_dict)

for method in METHODS:
    print(f"\n{'='*60}\nMETHOD : {method}\n{'='*60}")

    method_dir = ANALYSIS_DIR / method / "latent"
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

    for p, label in [(t_path, "trained"), (ul_path, "unlearned")]:
        if not p.exists():
            print(f"  Missing {label} checkpoint: {p.name} — skipping")
            continue

    # ── Load models ────────────────────────────────────────────────────
    trained_net = PolicyNet(state_dim, len(candidate_movies), hidden_dim).to(DEVICE)
    trained_net.load_state_dict(torch.load(t_path,  map_location=DEVICE))
    trained_net.eval()

    unlearned_net = PolicyNet(state_dim, len(candidate_movies), hidden_dim).to(DEVICE)
    unlearned_net.load_state_dict(torch.load(ul_path, map_location=DEVICE))
    unlearned_net.eval()

    # ── Step 1: Extract activations from trained model ─────────────────
    # Forget activations
    train_key = (t_lr, gamma, hidden_dim, train_bs)
    if train_key not in trained_acts_cache:
        print("  Extracting trained-model activations (retain + forget)...")
        r_uids, r_acts = extract_activations(trained_net, retain_trajectories,  layers=all_layers)
        f_uids, f_acts = extract_activations(trained_net, forget_trajectories,  layers=all_layers)
        trained_acts_cache[train_key] = {
            "retain_uids": r_uids, "retain_acts": r_acts,
            "forget_uids": f_uids, "forget_acts": f_acts,
        }
        print(f"    retain={len(r_uids)}  forget={len(f_uids)}  "
              f"fc3_dim={r_acts['fc3'].shape[1]}")
    else:
        print("  Using cached trained-model activations.")

    cached     = trained_acts_cache[train_key]
    r_uids     = cached["retain_uids"]
    r_acts     = cached["retain_acts"]
    f_acts     = cached["forget_acts"]

    # ── Step 2: Flip labels from THIS method's unlearned model ─────────
    print("  Computing flip labels...")
    flip_df = get_flip_labels(trained_net, unlearned_net, retain_trajectories, K=EVAL_K)
    flip_df.to_csv(method_dir / "flip_labels.csv", index=False)

    print("  Flip label distribution:")
    print("  " + flip_df["flip_label"].value_counts().to_string().replace("\n", "\n  "))

    # ── Step 3: Compute similarity scores per layer ────────────────────
    # One score column per (layer, similarity_strategy) combination.
    # Currently only ForgetSetSimilarity() is active, but you can
    # instantiate multiple sim_fn variants here if you want to compare.

    score_cols = []
    overlap_rows = {"user_id": r_uids}

    for layer in all_layers:
        col_name = f"{sim_fn.name}_{layer}"
        score_cols.append(col_name)

        print(f"  Computing {col_name}...")
        sim_fn.fit(f_acts[layer])                          # fit on forget vecs for this layer
        scores = sim_fn.score_batch(r_acts[layer])         # (N_retain,)
        overlap_rows[col_name] = scores

    scores_df = pd.DataFrame(overlap_rows)
    scores_df  = scores_df.merge(
        flip_df[["user_id", "hit_before", "hit_after", "flip_label"]],
        on="user_id", how="left"
    )
    scores_df.to_csv(method_dir / "latent_scores.csv", index=False)

    # ── Step 4: Summary table — same format as n-gram analysis ────────
    all_summaries = []
    for col in score_cols:
        summary = summarise(scores_df, col)
        summary.columns = pd.MultiIndex.from_tuples(
            [(col, stat) for stat in summary.columns]
        )
        all_summaries.append(summary)

    summary_df = pd.concat(all_summaries, axis=1)
    summary_df.to_csv(method_dir / "latent_summary.csv")

    print(f"\n  Latent summary ({sim_fn.name}):")
    print("  " + summary_df.to_string().replace("\n", "\n  "))

    # ── Step 5: Per-label nearest-neighbor details ─────────────────────
    # For flipped_harmful users: who is their nearest forget neighbor?
    # Useful for manual inspection.
    nn_rows = []
    flipped_idx = scores_df[scores_df["flip_label"] == "flipped_harmful"].index

    sim_fn.fit(f_acts["fc3"])   # re-fit on fc3

    for idx in flipped_idx:
        uid        = scores_df.loc[idx, "user_id"]
        r_vec      = r_acts["fc3"][r_uids.index(uid)]
        norm       = np.linalg.norm(r_vec) + 1e-10
        r_normed   = r_vec / norm
        sims       = sim_fn.forget_normed @ r_normed       # (N_forget,)
        top3_idx   = np.argsort(-sims)[:3]
        for rank, fi in enumerate(top3_idx):
            nn_rows.append({
                "retain_user_id":        uid,
                "flip_label":            "flipped_harmful",
                "rank":                  rank + 1,
                "forget_user_id":        cached["forget_uids"][fi],
                "cosine_sim":            round(float(sims[fi]), 6),
                f"{sim_fn.name}_fc3":   round(float(scores_df.loc[idx, f"{sim_fn.name}_fc3"]), 6),
            })

    nn_df = pd.DataFrame(nn_rows)
    nn_df.to_csv(method_dir / "flipped_nearest_forget_neighbors.csv", index=False)

    print(f"\n  Saved → {method_dir}")

    del trained_net, unlearned_net
    torch.cuda.empty_cache()

print(f"\n✓ Latent analysis complete. Results in: {ANALYSIS_DIR}")
