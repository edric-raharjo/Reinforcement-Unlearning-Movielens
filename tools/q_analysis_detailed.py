#!/usr/bin/env python
# coding: utf-8
"""
q_analysis_detailed.py  <forget_pct>  <max_retain_drop_pp>  [num_top_models]

1. Filters models based on performance constraints.
2. Identifies specific 'Flipped' movies for Groups H and D.
3. Calculates Anchored vs Global Delta Q.
4. Calculates Historical Jaccard Overlap (D vs H, C vs H).
5. Outputs: q_summary.csv and q_detailed_movies.csv
"""

import os, sys, warnings, random
from collections import Counter
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
warnings.filterwarnings("ignore")

# CONFIGURATION
TOP_SELECTION_K = 10
SEED = 97620260313

if len(sys.argv) < 3:
    raise ValueError("Usage: python q_analysis_detailed.py <forget_pct> <max_retain_drop_pp> [num_top_models]")

FORGET_PERCENTAGE  = int(sys.argv[1])
MAX_RETAIN_DROP_PP = float(sys.argv[2])
NUM_TOP_MODELS     = int(sys.argv[3]) if len(sys.argv) > 3 else None

DATA_DIR     = "C:/Bob/ml-1m"
RESULTS_BASE = f"C:/Bob/results/{FORGET_PERCENTAGE}_percent"
MODELS_DIR   = os.path.join(RESULTS_BASE, "models")
ANALYZE_DIR  = os.path.join(RESULTS_BASE, "analyze", "diagnose")
os.makedirs(ANALYZE_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(RESULTS_BASE, "tuning_full_results.csv")
TRAIN_PATH   = os.path.join(RESULTS_BASE, "train_phase_results.csv")
SUMMARY_CSV  = os.path.join(ANALYZE_DIR, "q_summary.csv")
DETAILED_CSV = os.path.join(ANALYZE_DIR, "q_detailed_movies.csv")

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METHODS = ["Ye_ApxI", "Ye_multi", "New_True_inf", "New_Max"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**31 - 1))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed % (2**31 - 1))

set_seed(SEED)

# ===========================================================================
# DATA LOADING
# ===========================================================================
def load_data(data_dir):
    r_df = pd.read_csv(os.path.join(data_dir, "ratings.dat"), sep="::", engine="python", names=["user_id", "movie_id", "rating", "timestamp"])
    m_df = pd.read_csv(os.path.join(data_dir, "movies.dat"), sep="::", engine="python", names=["movie_id", "title", "genres"], encoding="ISO-8859-1")
    u_df = pd.read_csv(os.path.join(data_dir, "users.dat"), sep="::", engine="python", names=["user_id", "gender", "age", "occupation", "zip"])
    return r_df, m_df, u_df

ratings_df, movies_df, users_df = load_data(DATA_DIR)
movie_id_to_title = dict(zip(movies_df.movie_id, movies_df.title))
movie_id_to_genres = dict(zip(movies_df.movie_id, movies_df.genres))

# ... [State Building Logic matches diagnose.py] ...
# (Assuming standard build_state_fn exists as per previous iterations)
# Building user features and trajectories
pilot_users_df = users_df[users_df["user_id"].isin(ratings_df["user_id"].unique())]
oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
user_feat_mat = oh.fit_transform(pilot_users_df[["gender", "age", "occupation"]])
user_feat_df = pd.DataFrame(user_feat_mat, index=pilot_users_df["user_id"])

all_genres = sorted({g for s in movies_df["genres"].astype(str) for g in s.split("|")})
genre_to_idx = {g: i for i, g in enumerate(all_genres)}
num_genres = len(all_genres)

def movie_genre_vector(genres_str):
    v = np.zeros(num_genres, dtype=np.float32)
    for g in str(genres_str).split("|"):
        if g in genre_to_idx: v[genre_to_idx[g]] = 1.0
    return v

movies_df["genre_vec"] = movies_df["genres"].apply(movie_genre_vector)
movie_genre_map = {mid: movies_df.loc[movies_df["movie_id"] == mid, "genre_vec"].values[0] for mid in movies_df["movie_id"].unique()}
state_dim = user_feat_df.shape[1] + num_genres

def build_state_fn(user_id, watched_movies):
    user_feat = user_feat_df.loc[user_id].values.astype(np.float32)
    pref_vec = np.zeros(num_genres, dtype=np.float32)
    for mid in watched_movies:
        if mid in movie_genre_map: pref_vec += movie_genre_map[mid]
    s = pref_vec.sum()
    if s > 0: pref_vec /= s
    return np.concatenate([user_feat, pref_vec]).astype(np.float32)

# ===========================================================================
# TRAJECTORIES & SPLITS
# ===========================================================================
sample_users_arr = ratings_df["user_id"].unique()
np.random.shuffle(sample_users_arr)
split_amt = int(np.round(FORGET_PERCENTAGE / 100 * len(sample_users_arr)))
forget_users = set(sample_users_arr[:split_amt])

user_data = {}
for uid, g in tqdm(ratings_df.groupby("user_id"), desc="Processing Trajectories"):
    if len(g) < 5: continue
    g = g.sort_values("timestamp")
    movies = g["movie_id"].tolist()
    mid = len(movies) // 2
    user_data[uid] = {
        "history": set(movies[:mid]),
        "future": set(movies[mid:]),
        "state": build_state_fn(uid, movies[:mid]),
        "is_forget": uid in forget_users
    }

candidate_movies = np.array(sorted(ratings_df["movie_id"].unique()))
num_actions = len(candidate_movies)
movie_to_idx = {mid: i for i, mid in enumerate(candidate_movies)}

# ===========================================================================
# MODELS & EVAL
# ===========================================================================
class PolicyNet(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, num_actions)
    def forward(self, x):
        return self.logits(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))

def _fmt(v): return f"{v:.0e}".replace("-", "n").replace("+", "p") if v < 0.01 else str(v).replace(".", "d")

# Selection Logic (Summary level)
results_df = pd.read_csv(RESULTS_PATH)
results_k = results_df[results_df["K"] == TOP_SELECTION_K].copy()
results_k["retain_drop"] = results_k["base_retain_Hit"] - results_k["retain_Hit"]
results_k["forget_drop"] = results_k["base_forget_Hit"] - results_k["forget_Hit"]

best_rows = {}
for method in METHODS:
    mdf = results_k[results_k["method"] == method].copy()
    if mdf.empty: continue
    constrained = mdf[mdf["retain_drop"] < (MAX_RETAIN_DROP_PP/100.0)].copy()
    if constrained.empty: constrained = mdf.copy()
    best_rows[method] = constrained.sort_values("forget_drop", ascending=False).iloc[0]

# Analysis
detailed_records = []
summary_records = []

for method, row in best_rows.items():
    print(f"Analyzing {method}...")
    t_lr, gamma = row["train_lr"], row["gamma"]
    hidden_dim, train_bs = int(row["hidden_dim"]), int(row["train_batch"])
    u_lr, u_iters, lam = row["unlearn_lr"], int(row["unlearn_iters"]), row["lambda_retain"]

    base_p = os.path.join(MODELS_DIR, f"trained__tlr{_fmt(t_lr)}__g{_fmt(gamma)}__h{hidden_dim}__bs{train_bs}.pt")
    unl_p = os.path.join(MODELS_DIR, f"unlearn__{method}__tlr{_fmt(t_lr)}__g{_fmt(gamma)}__h{hidden_dim}__bs{train_bs}__ulr{_fmt(u_lr)}__ui{u_iters}__lam{_fmt(lam)}.pt")

    net_b = PolicyNet(state_dim, num_actions, hidden_dim).to(DEVICE)
    net_b.load_state_dict(torch.load(base_p, map_location=DEVICE))
    net_b.eval()

    net_u = PolicyNet(state_dim, num_actions, hidden_dim).to(DEVICE)
    net_u.load_state_dict(torch.load(unl_p, map_location=DEVICE))
    net_u.eval()

    # Get states and predictions
    uids = list(user_data.keys())
    states = np.stack([user_data[u]["state"] for u in uids])
    st_tensor = torch.tensor(states, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        b_logits = net_b(st_tensor).cpu().numpy()
        u_logits = net_u(st_tensor).cpu().numpy()

    flipped_h, flipped_d, ok_c = [], [], []
    for i, u in enumerate(uids):
        future = user_data[u]["future"]
        b_topk = set(candidate_movies[np.argsort(-b_logits[i])[:TOP_SELECTION_K]])
        u_topk = set(candidate_movies[np.argsort(-u_logits[i])[:TOP_SELECTION_K]])
        
        b_hit = len(b_topk.intersection(future)) > 0
        u_hit = len(u_topk.intersection(future)) > 0
        
        flipped = b_hit and not u_hit
        target_movies = list(b_topk.intersection(future)) if flipped else []
        
        entry = {"uid": u, "flipped": flipped, "target_movies": target_movies, "idx": i}
        if user_data[u]["is_forget"] and flipped: flipped_h.append(entry)
        elif not user_data[u]["is_forget"]:
            if flipped: flipped_d.append(entry)
            else: ok_c.append(entry)

    # Calculate Global Baseline per movie from Group C
    ok_indices = [e["idx"] for e in ok_c]
    
    # Process Detailed User-Movie Logs
    for group_label, group_list in [("H", flipped_h), ("D", flipped_d)]:
        for user_entry in group_list:
            u_idx = user_entry["idx"]
            h_uid = user_entry["uid"]
            
            # Historical Jaccard (Max similarity to any Flipped Forget User)
            h_set = user_data[h_uid]["history"]
            max_j = 0
            if group_label == "D":
                max_j = max([len(h_set & user_data[fh["uid"]]["history"]) / len(h_set | user_data[fh["uid"]]["history"]) for fh in flipped_h]) if flipped_h else 0
            
            for mid in user_entry["target_movies"]:
                m_idx = movie_to_idx[mid]
                anchored_dq = u_logits[u_idx, m_idx] - b_logits[u_idx, m_idx]
                global_dq = np.mean(u_logits[ok_indices, m_idx] - b_logits[ok_indices, m_idx]) if ok_indices else 0
                
                detailed_records.append({
                    "Method": method, "User_Group": group_label, "User_ID": h_uid,
                    "Movie_ID": mid, "Movie_Title": movie_id_to_title.get(mid, "Unknown"),
                    "Genres": movie_id_to_genres.get(mid, ""),
                    "Max_Jaccard": max_j, "Anchored_DQ": anchored_dq, "Global_DQ": global_dq
                })

    # Summary Stats
    summary_records.append({
        "Method": method, "|D| Flipped": len(flipped_d),
        "Avg_Max_Jaccard_D": np.mean([r["Max_Jaccard"] for r in detailed_records if r["Method"]==method and r["User_Group"]=="D"]),
        "Avg_Max_Jaccard_C": np.mean([max([len(user_data[c["uid"]]["history"] & user_data[fh["uid"]]["history"]) / len(user_data[c["uid"]]["history"] | user_data[fh["uid"]]["history"]) for fh in flipped_h]) if flipped_h else 0 for c in ok_c]),
        "Avg_Anchored_DQ_H": np.mean([r["Anchored_DQ"] for r in detailed_records if r["Method"]==method and r["User_Group"]=="H"]),
        "Avg_Global_DQ_H": np.mean([r["Global_DQ"] for r in detailed_records if r["Method"]==method and r["User_Group"]=="H"]),
        "Avg_Anchored_DQ_D": np.mean([r["Anchored_DQ"] for r in detailed_records if r["Method"]==method and r["User_Group"]=="D"]),
        "Avg_Global_DQ_D": np.mean([r["Global_DQ"] for r in detailed_records if r["Method"]==method and r["User_Group"]=="D"])
    })

pd.DataFrame(detailed_records).to_csv(DETAILED_CSV, index=False)
pd.DataFrame(summary_records).to_csv(SUMMARY_CSV, index=False)
print(f"Done. Outputs in {ANALYZE_DIR}")