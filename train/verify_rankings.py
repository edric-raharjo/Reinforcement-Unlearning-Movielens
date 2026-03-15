#!/usr/bin/env python
# coding: utf-8
"""
eval_combined.py <forget_pct>

Evaluates all trained base models on the combined (forget + retain) dataset.
Ranks them by Hit@10 and compares the ranking against the retain-only Hit@10.
"""

import hashlib
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CLI & Setup
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    raise ValueError("Usage: python eval_combined.py <forget_pct>")

FORGET_PERCENTAGE = int(sys.argv[1])
K_VAL = 10

SEED = 97620260313
def set_seed(seed):
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**31 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed % (2**31 - 1))

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR = "C:/Bob/ml-1m"
RESULTS_BASE = f"C:/Bob/results/{FORGET_PERCENTAGE}_percent"
MODELS_DIR = os.path.join(RESULTS_BASE, "models")
TRAIN_RESULTS_PATH = os.path.join(RESULTS_BASE, "train_phase_results.csv")
COMBINED_EVAL_PATH = os.path.join(RESULTS_BASE, "combined_eval_results.csv")

print(f"Device      : {DEVICE}")
print(f"Forget %    : {FORGET_PERCENTAGE}")
print(f"Train File  : {TRAIN_RESULTS_PATH}")

# ---------------------------------------------------------------------------
# Data loading & Feature Engineering (Identical to training script)
# ---------------------------------------------------------------------------
def load_data(data_dir):
    ratings_df = pd.read_csv(os.path.join(data_dir, "ratings.dat"), sep="::", engine="python", names=["user_id", "movie_id", "rating", "timestamp"])
    movies_df = pd.read_csv(os.path.join(data_dir, "movies.dat"), sep="::", engine="python", names=["movie_id", "title", "genres"], encoding="ISO-8859-1")
    users_df = pd.read_csv(os.path.join(data_dir, "users.dat"), sep="::", engine="python", names=["user_id", "gender", "age", "occupation", "zip"])
    return ratings_df, movies_df, users_df

ratings_df, movies_df, users_df = load_data(DATA_DIR)
sample_users = ratings_df["user_id"].unique().tolist()
pilot_ratings_all = ratings_df[ratings_df["user_id"].isin(sample_users)].copy()
pilot_ratings_all.sort_values(["user_id", "timestamp"], inplace=True)

pilot_users_df = users_df[users_df["user_id"].isin(sample_users)]
oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
user_cat_mat = oh.fit_transform(pilot_users_df[["gender", "age", "occupation"]])
user_feat_df = pd.DataFrame(user_cat_mat, index=pilot_users_df["user_id"])

all_genres = sorted({g for s in movies_df["genres"].astype(str) for g in s.split("|")})
genre_to_idx = {g: i for i, g in enumerate(all_genres)}
num_genres = len(all_genres)

def movie_genre_vector(genres_str):
    v = np.zeros(num_genres, dtype=np.float32)
    for g in str(genres_str).split("|"):
        if g in genre_to_idx:
            v[genre_to_idx[g]] = 1.0
    return v

movies_df["genre_vec"] = movies_df["genres"].apply(movie_genre_vector)
movie_genre_map = {mid: movies_df.loc[movies_df["movie_id"] == mid, "genre_vec"].values[0] for mid in movies_df["movie_id"].unique()}

state_dim = user_feat_df.shape[1] + num_genres

def build_state_fn(user_id, watched_movies):
    user_feat = user_feat_df.loc[user_id].values.astype(np.float32)
    pref_vec = np.zeros(num_genres, dtype=np.float32)
    for mid in watched_movies:
        if mid in movie_genre_map:
            pref_vec += movie_genre_map[mid]
    s = pref_vec.sum()
    if s > 0:
        pref_vec /= s
    return np.concatenate([user_feat, pref_vec]).astype(np.float32)

candidate_movies = np.array(sorted(pilot_ratings_all["movie_id"].unique()))
num_actions = len(candidate_movies)

# COMBINED TRAJECTORIES (Forget + Retain together)
trajectories_all = [
    {"user_id": uid, "movies": g["movie_id"].tolist(), "ratings": g["rating"].tolist()}
    for uid, g in pilot_ratings_all.groupby("user_id") if len(g) >= 5
]
print(f"Total Combined Trajectories: {len(trajectories_all)}")

# ---------------------------------------------------------------------------
# PolicyNet & Evaluation setup
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.logits(x)

def evaluate_policy(policy_net, trajectories, K=10):
    hits = []
    for traj in trajectories:
        uid = traj["user_id"]
        movies = traj["movies"]
        if len(movies) < 5: continue
        split = len(movies) // 2
        future = set(movies[split:])
        state = build_state_fn(uid, movies[:split])
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(policy_net(state_t), dim=-1).squeeze(0).cpu().numpy()
        topk_movies = candidate_movies[np.argsort(-probs, kind="stable")[:K]]
        hits.append(int(any(m in future for m in topk_movies)))
    return float(np.mean(hits)) if hits else 0.0

def _fmt(v):
    if v < 0.01: return f"{v:.0e}".replace("-", "n").replace("+", "p")
    return str(v).replace(".", "d")

def trained_model_path(t_lr, gamma, hidden_dim, train_bs):
    return os.path.join(MODELS_DIR, f"trained__tlr{_fmt(t_lr)}__g{_fmt(gamma)}__h{hidden_dim}__bs{train_bs}.pt")

# ===========================================================================
# Execution: Load results, test combined, compare rankings
# ===========================================================================
if not os.path.exists(TRAIN_RESULTS_PATH):
    raise FileNotFoundError(f"Cannot find {TRAIN_RESULTS_PATH}. Have you run Phase 1?")

train_df = pd.read_csv(TRAIN_RESULTS_PATH)
# Filter strictly for the K value we care about comparing
train_df = train_df[train_df["K"] == K_VAL].copy()

# Deduplicate in case there are multiple entries for the same config
key_cols = ["train_lr", "gamma", "hidden_dim", "train_batch"]
train_df = train_df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)

results = []

print("\nEvaluating base models on COMBINED dataset...")
for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    t_lr = row["train_lr"]
    gamma = row["gamma"]
    h_dim = int(row["hidden_dim"])
    t_bs = int(row["train_batch"])
    retain_hit = row["base_retain_Hit"]
    
    model_file = trained_model_path(t_lr, gamma, h_dim, t_bs)
    if not os.path.exists(model_file):
        print(f"\nMissing model file: {model_file} - skipping.")
        continue
        
    net = PolicyNet(state_dim, num_actions, hidden_dim=h_dim).to(DEVICE)
    net.load_state_dict(torch.load(model_file, map_location=DEVICE))
    net.eval()
    
    combined_hit = evaluate_policy(net, trajectories_all, K=K_VAL)
    
    results.append({
        "train_lr": t_lr,
        "gamma": gamma,
        "hidden_dim": h_dim,
        "train_batch": t_bs,
        "base_retain_Hit": retain_hit,
        "combined_Hit": combined_hit
    })

eval_df = pd.DataFrame(results)

# Rank the models
eval_df["rank_retain"] = eval_df["base_retain_Hit"].rank(ascending=False, method="min").astype(int)
eval_df["rank_combined"] = eval_df["combined_Hit"].rank(ascending=False, method="min").astype(int)

# Calculate rank difference
eval_df["rank_shift"] = eval_df["rank_retain"] - eval_df["rank_combined"]

# Sort the final output by the combined hit performance
eval_df = eval_df.sort_values("combined_Hit", ascending=False).reset_index(drop=True)
eval_df.to_csv(COMBINED_EVAL_PATH, index=False)

print(f"\n✅ Evaluation complete. Saved to {COMBINED_EVAL_PATH}")

# Print a quick summary of the top 5
print("\nTop 5 Models (by Combined Hit@10):")
print(eval_df[["train_lr", "gamma", "hidden_dim", "train_batch", "combined_Hit", "rank_retain", "rank_shift"]].head(5).to_string())

# Spearman correlation to check overall ranking similarity
correlation = eval_df["base_retain_Hit"].corr(eval_df["combined_Hit"], method="spearman")
print(f"\nSpearman Rank Correlation (Retain vs. Combined): {correlation:.4f}")
if correlation > 0.9:
    print("Conclusion: Rankings are highly identical. 'base_retain_Hit' is an excellent proxy.")
elif correlation > 0.7:
    print("Conclusion: Rankings are similar, but with some shifting.")
else:
    print("Conclusion: Rankings diverge significantly. You may need to reconsider your selection metric.")