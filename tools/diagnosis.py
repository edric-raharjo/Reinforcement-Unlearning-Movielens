#!/usr/bin/env python
# coding: utf-8
"""
diagnosis.py  <forget_pct>  <max_retain_drop_pp>  [results_csv_name] [num_top_models]

Geometric diagnosis: are damaged retain users geometrically closer to forget users?
Produces an interactive HTML dashboard at:
    C:/Bob/results_deterministic/<forget_pct>_percent/diagnosis_dashboard_{MAX_RETAIN_DROP_PP}.html
"""

import copy, hashlib, os, sys, warnings
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial.distance import cdist
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIGURATION
# ===========================================================================
# Choose from: "cosine", "euclidean", "mahalanobis"
DISTANCE_METRIC = "cosine" 

SIMILARITY_BINS = 5  # Number of bins for the distribution histograms

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if len(sys.argv) < 3:
    raise ValueError("Usage: python diagnosis.py <forget_pct> <max_retain_drop_pp> [results_csv_name] [num_top_models]")

FORGET_PERCENTAGE    = int(sys.argv[1])
MAX_RETAIN_DROP_PP   = float(sys.argv[2])          # e.g. 5.0  → < 5 pp
# RESULTS_CSV_NAME     = sys.argv[3] if len(sys.argv) > 3 else "tuning_full_results.csv"
RESULTS_CSV_NAME     = "tuning_full_results.csv"
NUM_TOP_MODELS       = int(sys.argv[3]) if len(sys.argv) > 3 else None

# Metric direction logic
if DISTANCE_METRIC == "cosine":
    METRIC_TYPE = "similarity"
    MWU_ALT = "greater"  # Damaged sim > OK sim
else:
    METRIC_TYPE = "distance"
    MWU_ALT = "less"     # Damaged dist < OK dist

# ---------------------------------------------------------------------------
# Seed & Paths
# ---------------------------------------------------------------------------
SEED = 97620260313

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

DATA_DIR     = "C:/Bob/ml-1m"
    RESULTS_BASE = f"C:/Bob/results/{FORGET_PERCENTAGE}_percent" if FORGET_PERCENTAGE in [1, 20] else f"D:/Bob_Skripsi_Do Not Delete/results/{FORGET_PERCENTAGE}_percent"
MODELS_DIR   = os.path.join(RESULTS_BASE, "models")
ANALYZE_DIR = f"D:/Bob_Skripsi_Do Not Delete/Analysis/Normal/{FORGET_PERCENTAGE}_percent"
os.makedirs(ANALYZE_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(RESULTS_BASE, RESULTS_CSV_NAME)
TRAIN_PATH   = os.path.join(RESULTS_BASE, "train_phase_results.csv")
DIAG_HTML = os.path.join(ANALYZE_DIR, f"diagnosis_dashboard_{MAX_RETAIN_DROP_PP}.html")
DIAG_CSV = os.path.join(ANALYZE_DIR, f"diagnosis_stats_{MAX_RETAIN_DROP_PP}.csv")

TOP_SELECTION_K = 5
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METHODS         = ["Ye_ApxI", "Ye_multi", "New_True_inf", "New_Max"]

print(f"Device        : {DEVICE}")
print(f"Forget %      : {FORGET_PERCENTAGE}")
print(f"Retain Limit  : < {MAX_RETAIN_DROP_PP} pp")
print(f"Metric        : {DISTANCE_METRIC.upper()}")
print(f"K             : {TOP_SELECTION_K}")

# ---------------------------------------------------------------------------
# Data loading & Feature engineering
# ---------------------------------------------------------------------------
def load_data(data_dir):
    ratings_df = pd.read_csv(os.path.join(data_dir, "ratings.dat"), sep="::", engine="python", names=["user_id", "movie_id", "rating", "timestamp"])
    movies_df = pd.read_csv(os.path.join(data_dir, "movies.dat"), sep="::", engine="python", names=["movie_id", "title", "genres"], encoding="ISO-8859-1")
    users_df = pd.read_csv(os.path.join(data_dir, "users.dat"), sep="::", engine="python", names=["user_id", "gender", "age", "occupation", "zip"])
    return ratings_df, movies_df, users_df

ratings_df, movies_df, users_df = load_data(DATA_DIR)
sample_users  = ratings_df["user_id"].unique().tolist()
pilot_ratings = ratings_df[ratings_df["user_id"].isin(sample_users)].copy()
pilot_ratings.sort_values(["user_id", "timestamp"], inplace=True)

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
movie_genre_map = {mid: movies_df.loc[movies_df["movie_id"] == mid, "genre_vec"].values[0] for mid in movies_df["movie_id"].unique()}
state_dim = user_feat_df.shape[1] + num_genres

def build_state_fn(user_id, watched_movies):
    user_feat = user_feat_df.loc[user_id].values.astype(np.float32)
    pref_vec  = np.zeros(num_genres, dtype=np.float32)
    for mid in watched_movies:
        if mid in movie_genre_map:
            pref_vec += movie_genre_map[mid]
    s = pref_vec.sum()
    if s > 0: pref_vec /= s
    return np.concatenate([user_feat, pref_vec]).astype(np.float32)

# ---------------------------------------------------------------------------
# Forget / Retain split & Trajectories
# ---------------------------------------------------------------------------
sample_users_arr = np.array(sample_users)
np.random.shuffle(sample_users_arr)

split_amt      = int(np.round(FORGET_PERCENTAGE / 100 * pilot_ratings["user_id"].nunique()))
forget_users   = sample_users_arr[:split_amt]
retain_users   = sample_users_arr[split_amt:]

pilot_ratings_all = ratings_df[ratings_df["user_id"].isin(sample_users)].copy()
pilot_ratings_all.sort_values(["user_id", "timestamp"], inplace=True)

trajectories_all = [
    {"user_id": uid, "movies": g["movie_id"].tolist(), "ratings": g["rating"].tolist()}
    for uid, g in pilot_ratings_all.groupby("user_id") if len(g) >= 5
]
forget_user_set     = set(forget_users.tolist())
retain_user_set     = set(retain_users.tolist())
forget_trajectories = [t for t in trajectories_all if t["user_id"] in forget_user_set]
retain_trajectories = [t for t in trajectories_all if t["user_id"] in retain_user_set]

candidate_movies = np.array(sorted(pilot_ratings_all["movie_id"].unique()))
num_actions      = len(candidate_movies)

# ---------------------------------------------------------------------------
# PolicyNet & File Paths
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

def _fmt(v): return f"{v:.0e}".replace("-", "n").replace("+", "p") if v < 0.01 else str(v).replace(".", "d")

def trained_model_path(t_lr, gamma, hidden_dim, train_bs):
    return os.path.join(MODELS_DIR, f"trained__tlr{_fmt(t_lr)}__g{_fmt(gamma)}__h{hidden_dim}__bs{train_bs}.pt")

def unlearned_model_path(t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam):
    return os.path.join(MODELS_DIR, f"unlearn__{method}__tlr{_fmt(t_lr)}__g{_fmt(gamma)}__h{hidden_dim}__bs{train_bs}__ulr{_fmt(u_lr)}__ui{u_iters}__lam{_fmt(lam)}.pt")

# ===========================================================================
# PHASE 1 — State vectors
# ===========================================================================
print("\n" + "="*60)
print("PHASE 1 — Computing user state vectors (midpoint)")
print("="*60)

def compute_user_states(trajectories, desc="states"):
    states = {}
    for traj in tqdm(trajectories, desc=desc):
        if len(traj["movies"]) < 5: continue
        mid = len(traj["movies"]) // 2
        states[traj["user_id"]] = build_state_fn(traj["user_id"], traj["movies"][:mid])
    return states

forget_states = compute_user_states(forget_trajectories, "Forget states")
retain_states = compute_user_states(retain_trajectories, "Retain states")

forget_uids = list(forget_states.keys())
retain_uids = list(retain_states.keys())
forget_mat  = np.stack([forget_states[u] for u in forget_uids])   
retain_mat  = np.stack([retain_states[u] for u in retain_uids])   

retain_uid_to_idx = {uid: i for i, uid in enumerate(retain_uids)}
forget_uid_to_idx = {uid: i for i, uid in enumerate(forget_uids)}
forget_core_uids_per_method = {}   

# ===========================================================================
# PHASE 2 — Strict Model Selection
# ===========================================================================
print("\n" + "="*60)
print(f"PHASE 2 — Selecting best models (max_retain_drop < {MAX_RETAIN_DROP_PP} pp)")
print("="*60)

results_df = pd.read_csv(RESULTS_PATH)

if NUM_TOP_MODELS is not None and os.path.exists(TRAIN_PATH):
    train_df = pd.read_csv(TRAIN_PATH)
    sort_metric = "base_combined_Hit" if "base_combined_Hit" in train_df.columns else "base_retain_Hit"
    for c in ["K", "train_lr", "gamma", "hidden_dim", "train_batch", sort_metric]:
        if c in train_df.columns: train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
            
    rank_df = train_df[train_df["K"] == TOP_SELECTION_K].sort_values(
        [sort_metric, "train_lr", "gamma", "hidden_dim", "train_batch"],
        ascending=[False, True, True, True, True], kind="mergesort"
    ).reset_index(drop=True)
    
    keys = ["train_lr", "gamma", "hidden_dim", "train_batch"]
    base_configs_df = rank_df.head(NUM_TOP_MODELS)[keys].drop_duplicates()
    results_df = results_df.merge(base_configs_df, on=keys, how="inner")

results_k = results_df[results_df["K"] == TOP_SELECTION_K].copy()
results_k["retain_drop"] = results_k["base_retain_Hit"] - results_k["retain_Hit"]
results_k["forget_drop"] = results_k["base_forget_Hit"] - results_k["forget_Hit"]

constraint = MAX_RETAIN_DROP_PP / 100.0
best_rows  = {}

for method in METHODS:
    mdf = results_k[results_k["method"] == method].copy()
    if mdf.empty: continue
    
    # STRICT FILTER MATCHING DASHBOARD
    constrained = mdf[mdf["retain_drop"] < constraint].copy()
    if constrained.empty:
        print(f"  ⚠ {method}: no config within constraint — using unconstrained best")
        constrained = mdf.copy()
        
    constrained = constrained.sort_values(by="forget_drop", ascending=False, na_position="last")
    best_rows[method] = constrained.iloc[0]

def evaluate_per_user(policy_net, trajectories, K=10, desc="eval"):
    hits = {}
    for traj in tqdm(trajectories, desc=desc, leave=False):
        uid = traj["user_id"]
        movies = traj["movies"]
        if len(movies) < 5: continue
        mid = len(movies) // 2
        future = set(movies[mid:])
        state = build_state_fn(uid, movies[:mid])
        st = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(policy_net(st), dim=-1).squeeze(0).cpu().numpy()
        topk = candidate_movies[np.argsort(-probs, kind="stable")[:K]]
        hits[uid] = int(any(m in future for m in topk))
    return hits

per_method_data = {}

for method, row in best_rows.items():
    t_lr, gamma = row["train_lr"], row["gamma"]
    hidden_dim, train_bs = int(row["hidden_dim"]), int(row["train_batch"])
    u_lr, u_iters, lam = row["unlearn_lr"], int(row["unlearn_iters"]), row["lambda_retain"]

    net_base = PolicyNet(state_dim, num_actions, hidden_dim=hidden_dim).to(DEVICE)
    net_base.load_state_dict(torch.load(trained_model_path(t_lr, gamma, hidden_dim, train_bs), map_location=DEVICE))
    net_base.eval()

    net_ul = PolicyNet(state_dim, num_actions, hidden_dim=hidden_dim).to(DEVICE)
    net_ul.load_state_dict(torch.load(unlearned_model_path(t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam), map_location=DEVICE))
    net_ul.eval()

    r_before = evaluate_per_user(net_base, retain_trajectories, desc=f"  {method} retain base")
    r_after  = evaluate_per_user(net_ul, retain_trajectories, desc=f"  {method} retain unl")
    f_before = evaluate_per_user(net_base, forget_trajectories, desc=f"  {method} forget base")
    f_after  = evaluate_per_user(net_ul, forget_trajectories, desc=f"  {method} forget unl")

    # Retain Df
    r_uids = [u for u in r_before if u in r_after and u in retain_uid_to_idx]
    df_r = pd.DataFrame({"uid": r_uids, "hit_before": [r_before[u] for u in r_uids], "hit_after": [r_after[u] for u in r_uids]})
    df_r["delta"]   = df_r["hit_after"] - df_r["hit_before"]  # Negative delta = damage/drop
    df_r["damaged"] = df_r["delta"] < 0
    df_r["group"]   = df_r["delta"].apply(lambda d: "Damaged" if d < 0 else "OK")

    # Forget Df
    f_uids = [u for u in f_before if u in f_after and u in forget_uid_to_idx]
    df_f = pd.DataFrame({"uid": f_uids, "hit_before": [f_before[u] for u in f_uids], "hit_after": [f_after[u] for u in f_uids]})
    df_f["delta"]   = df_f["hit_after"] - df_f["hit_before"]

    n_core = max(1, int(round(0.10 * len(df_f)))) if len(df_f) > 0 else 0
    forget_core_uids_per_method[method] = df_f.sort_values("delta").head(n_core)["uid"].tolist() if n_core else []

    per_method_data[method] = {"retain": df_r, "forget": df_f, "row": row}
    del net_base, net_ul
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ===========================================================================
# PHASE 3 — Geometric Proximity Calculation
# ===========================================================================
print("\n" + "="*60)
print(f"PHASE 3 — Calculating Geometric Proximity ({DISTANCE_METRIC})")
print("="*60)

all_states = np.vstack([forget_mat, retain_mat])
inv_cov = None
if DISTANCE_METRIC == "mahalanobis":
    cov = np.cov(all_states, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

for method, d in per_method_data.items():
    core_uids = forget_core_uids_per_method.get(method, [])
    df_r = d["retain"].copy()

    # ✅ FIX: Initialize columns so Phase 5 never hits KeyError
    df_r["centroid_val"]    = np.nan
    df_r["pairwise_max_val"] = np.nan

    if not core_uids:
        per_method_data[method]["retain"] = df_r  # ✅ save back before skipping
        continue

    core_idx = [forget_uid_to_idx[u] for u in core_uids if u in forget_uid_to_idx]
    if not core_idx:
        per_method_data[method]["retain"] = df_r  # ✅ save back before skipping
        continue

    core_mat = forget_mat[core_idx]
    df_r = d["retain"].copy()

    if DISTANCE_METRIC == "cosine":
        c_norm = core_mat / (np.linalg.norm(core_mat, axis=1, keepdims=True) + 1e-10)
        r_norm = retain_mat / (np.linalg.norm(retain_mat, axis=1, keepdims=True) + 1e-10)
        
        centroid = c_norm.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        
        df_r["centroid_val"] = [float(r_norm[retain_uid_to_idx[u]] @ centroid) for u in df_r["uid"]]
        df_r["pairwise_max_val"] = [float(np.max(r_norm[retain_uid_to_idx[u]] @ c_norm.T)) for u in df_r["uid"]]

    elif DISTANCE_METRIC == "euclidean":
        centroid = core_mat.mean(axis=0)
        c_dists = cdist(retain_mat, [centroid], 'euclidean').flatten()
        p_dists = cdist(retain_mat, core_mat, 'euclidean').min(axis=1)
        
        df_r["centroid_val"] = [float(c_dists[retain_uid_to_idx[u]]) for u in df_r["uid"]]
        df_r["pairwise_max_val"] = [float(p_dists[retain_uid_to_idx[u]]) for u in df_r["uid"]]
        
    elif DISTANCE_METRIC == "mahalanobis":
        centroid = core_mat.mean(axis=0)
        c_dists = cdist(retain_mat, [centroid], 'mahalanobis', VI=inv_cov).flatten()
        p_dists = cdist(retain_mat, core_mat, 'mahalanobis', VI=inv_cov).min(axis=1)
        
        df_r["centroid_val"] = [float(c_dists[retain_uid_to_idx[u]]) for u in df_r["uid"]]
        df_r["pairwise_max_val"] = [float(p_dists[retain_uid_to_idx[u]]) for u in df_r["uid"]]

    per_method_data[method]["retain"] = df_r

# ===========================================================================
# PHASE 4 — Mann-Whitney U test
# ===========================================================================
print("\n" + "="*60)
print(f"PHASE 4 — Statistical tests (H1: Damaged is CLOSER than OK)")
print("="*60)

stat_rows = []
for method, d in per_method_data.items():
    df_r = d["retain"]
    for col, label in [("centroid_val", "Centroid"), ("pairwise_max_val", "Pairwise")]:
        dmg = df_r[df_r["damaged"]][col].dropna().values
        ok  = df_r[~df_r["damaged"]][col].dropna().values
        if len(dmg) < 3 or len(ok) < 3: continue
        
        u_stat, p_val = stats.mannwhitneyu(dmg, ok, alternative=MWU_ALT)
        corr_r, p_corr = stats.pearsonr(df_r[col].dropna(), df_r.loc[df_r[col].notna(), "delta"])
        
        stat_rows.append({
            "Method": method,
            "Proximity Type": label,
            "N Damaged": len(dmg),
            "N OK": len(ok),
            "Mean (Damaged)": round(dmg.mean(), 4),
            "Mean (OK)": round(ok.mean(), 4),
            "Δ Mean": round(dmg.mean() - ok.mean(), 4),
            "U Statistic": round(u_stat, 1),
            "p-value": round(p_val, 4),
            "Significant (p<.05)": "✓" if p_val < 0.05 else "✗",
            "Pearson r (val~delta)": round(corr_r, 4),
            "Pearson p": round(p_corr, 4),
        })

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(DIAG_CSV, index=False)

# ===========================================================================
# PHASE 5 — HTML UI & Plot Generation
# ===========================================================================
PANEL   = "#ffffff"
TEXT    = "#0f172a"
GRIDCOL = "#e2e8f0"
COLORS  = {"Damaged": "#dc2626", "OK": "#2563eb", "Forget": "#d97706"} 

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(all_states)
forget_coords = coords[:len(forget_mat)]
retain_coords = coords[len(forget_mat):]

def make_pca_fig():
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"PCA — {m}" for m in METHODS],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )

    # --- Dummy damaged trace to guarantee legend entry ---
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],  # no visible point
            mode="markers",
            marker=dict(color=COLORS["Damaged"], size=6, opacity=0.0),
            name="Retain — Damaged",
            legendgroup="Damaged",
            showlegend=True,
        ),
        row=1, col=1,
    )

    for idx, method in enumerate(METHODS):
        r, c = divmod(idx, 2)
        df_r = per_method_data[method]["retain"]
        d_map = dict(zip(df_r["uid"], df_r["damaged"]))
        d_mask  = np.array([d_map.get(u, False) for u in retain_uids])
        ok_mask = ~d_mask

        fig.add_trace(
            go.Scatter(
                x=forget_coords[:, 0],
                y=forget_coords[:, 1],
                mode="markers",
                marker=dict(color=COLORS["Forget"], size=4, opacity=0.4),
                name="Forget users",
                legendgroup="Forget",
                showlegend=(idx == 0),
            ),
            row=r+1, col=c+1,
        )

        fig.add_trace(
            go.Scatter(
                x=retain_coords[ok_mask, 0],
                y=retain_coords[ok_mask, 1],
                mode="markers",
                marker=dict(color=COLORS["OK"], size=4, opacity=0.4),
                name="Retain — OK",
                legendgroup="OK",
                showlegend=(idx == 0),
            ),
            row=r+1, col=c+1,
        )

        # real damaged points, no need to showlegend again
        fig.add_trace(
            go.Scatter(
                x=retain_coords[d_mask, 0],
                y=retain_coords[d_mask, 1],
                mode="markers",
                marker=dict(color=COLORS["Damaged"], size=6, opacity=0.85),
                name="Retain — Damaged",
                legendgroup="Damaged",
                showlegend=False,
            ),
            row=r+1, col=c+1,
        )

    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font_color=TEXT,
        height=650,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            bgcolor=PANEL,
            bordercolor=GRIDCOL,
            borderwidth=1,
            groupclick="togglegroup",  # keep this from previous fix
        ),
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig

def make_binned_distribution_fig(sim_col, title):
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{m}" for m in METHODS], horizontal_spacing=0.08, vertical_spacing=0.14)
    
    # Calculate global min/max for dynamic binning
    all_vals = []
    for d in per_method_data.values():
        if sim_col in d["retain"].columns:
            all_vals.extend(d["retain"][sim_col].dropna().tolist())
    
    v_min = min(all_vals) if all_vals else 0.0
    v_max = max(all_vals) if all_vals else 1.0
    
    # If distance is exactly 0 everywhere, pad it to avoid div by zero
    if v_max == v_min: v_max = v_min + 1.0
    bin_size = (v_max - v_min) / SIMILARITY_BINS

    for idx, method in enumerate(METHODS):
        r, c = divmod(idx, 2)
        if method not in per_method_data: continue
        df_r = per_method_data[method]["retain"]
        
        dmg_v = df_r[df_r["damaged"]][sim_col].dropna().values
        ok_v  = df_r[~df_r["damaged"]][sim_col].dropna().values
        
        # Self-calculation for core forget
        core_uids = forget_core_uids_per_method.get(method, [])
        fgt_v = np.array([])
        if len(core_uids) >= 2:
            core_idx = [forget_uid_to_idx.get(u) for u in core_uids if u in forget_uid_to_idx]
            core_mat = forget_mat[core_idx]
            
            if DISTANCE_METRIC == "cosine":
                c_norm = core_mat / (np.linalg.norm(core_mat, axis=1, keepdims=True) + 1e-10)
                sims = c_norm @ c_norm.T
                fgt_self = [sims[j][sims[j] != sims[j,j]].max() if len(sims[j][sims[j] != sims[j,j]]) > 0 else sims[j].max() for j in range(len(c_norm))]
            elif DISTANCE_METRIC == "euclidean":
                dists = cdist(core_mat, core_mat, 'euclidean')
                fgt_self = [dists[j][dists[j] != 0].min() if len(dists[j][dists[j] != 0]) > 0 else 0.0 for j in range(len(core_mat))]
            elif DISTANCE_METRIC == "mahalanobis":
                dists = cdist(core_mat, core_mat, 'mahalanobis', VI=inv_cov)
                fgt_self = [dists[j][dists[j] != 0].min() if len(dists[j][dists[j] != 0]) > 0 else 0.0 for j in range(len(core_mat))]
            fgt_v = np.array(fgt_self)

        for vals, label, color in [(fgt_v, "Core Forget (10%)", COLORS["Forget"]), (ok_v, "Retain — OK", COLORS["OK"]), (dmg_v, "Retain — Damaged", COLORS["Damaged"])]:
            if len(vals) == 0: continue
            fig.add_trace(go.Histogram(x=vals, name=label, marker_color=color, opacity=0.75, xbins=dict(start=v_min, end=v_max, size=bin_size), histnorm='probability', showlegend=(idx == 0), legendgroup=label), row=r+1, col=c+1)

    fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color=TEXT, height=650, barmode="group", margin=dict(l=40, r=40, t=60, b=40), legend=dict(bgcolor=PANEL, bordercolor=GRIDCOL, borderwidth=1))
    tick_vals = np.linspace(v_min, v_max, SIMILARITY_BINS + 1)
    fig.update_xaxes(gridcolor=GRIDCOL, zerolinecolor=GRIDCOL, tickvals=tick_vals)
    return fig

fig_pca      = make_pca_fig()
fig_dist_c   = make_binned_distribution_fig("centroid_val", f"Centroid Proximity ({DISTANCE_METRIC})")
fig_dist_p   = make_binned_distribution_fig("pairwise_max_val", f"Pairwise Proximity ({DISTANCE_METRIC})")

summary_rows = []
for method in METHODS:
    if method not in per_method_data: continue
    d = per_method_data[method]
    df_r = d["retain"]
    n_d = df_r["damaged"].sum()
    n_ok = (~df_r["damaged"]).sum()
    dmg = df_r[df_r["damaged"]]
    ok = df_r[~df_r["damaged"]]
    # ✅ FIX: Guard column existence AND n_d > 0
    has_c = "centroid_val"    in df_r.columns and df_r["centroid_val"].notna().any()
    has_p = "pairwise_max_val" in df_r.columns and df_r["pairwise_max_val"].notna().any()
    summary_rows.append({
        "Method": method,
        "Forget Drop (pp)": f"{d['row']['forget_drop']*100:.2f}",
        "Retain Drop (pp)": f"{d['row']['retain_drop']*100:.2f}",
        "# Damaged Retain": int(n_d),
        "# OK Retain": int(n_ok),
        "Damaged %": f"{n_d/(n_d+n_ok)*100:.1f}%" if (n_d+n_ok) > 0 else "0.0%",
        "Mean Centroid (Dmg)":  f"{dmg['centroid_val'].mean():.4f}"    if (n_d > 0 and has_c) else "N/A",
        "Mean Centroid (OK)":   f"{ok['centroid_val'].mean():.4f}"     if has_c              else "N/A",
        "Mean Pairwise (Dmg)":  f"{dmg['pairwise_max_val'].mean():.4f}" if (n_d > 0 and has_p) else "N/A",
        "Mean Pairwise (OK)":   f"{ok['pairwise_max_val'].mean():.4f}" if has_p              else "N/A",
    })
summary_df = pd.DataFrame(summary_rows)

def get_cell_style(col_name, val, row):
    base_style = ""
    def safe_float(v):
        try: return float(v)
        except: return -float('inf') if METRIC_TYPE == "similarity" else float('inf')

    if col_name == "Significant (p<.05)" and val == "✓": return base_style + "background-color:#fef08a;color:#854d0e;font-weight:bold;"
        
    def check_better(dmg_v, ok_v):
        if METRIC_TYPE == "similarity" and dmg_v > ok_v: return True
        if METRIC_TYPE == "distance" and dmg_v < ok_v: return True
        return False

    if col_name in ["Mean (Damaged)", "Mean (OK)"]:
        dmg_val, ok_val = safe_float(row.get("Mean (Damaged)")), safe_float(row.get("Mean (OK)"))
        if col_name == "Mean (Damaged)" and check_better(dmg_val, ok_val): return base_style + "background-color:#dcfce7;color:#166534;font-weight:600;"
        if col_name == "Mean (OK)" and not check_better(dmg_val, ok_val): return base_style + "background-color:#dcfce7;color:#166534;font-weight:600;"

    if col_name in ["Mean Centroid (Dmg)", "Mean Centroid (OK)"]:
        dmg_val, ok_val = safe_float(row.get("Mean Centroid (Dmg)")), safe_float(row.get("Mean Centroid (OK)"))
        if col_name == "Mean Centroid (Dmg)" and check_better(dmg_val, ok_val): return base_style + "background-color:#dcfce7;color:#166534;font-weight:600;"
        if col_name == "Mean Centroid (OK)" and not check_better(dmg_val, ok_val): return base_style + "background-color:#dcfce7;color:#166534;font-weight:600;"

    if col_name in ["Mean Pairwise (Dmg)", "Mean Pairwise (OK)"]:
        dmg_val, ok_val = safe_float(row.get("Mean Pairwise (Dmg)")), safe_float(row.get("Mean Pairwise (OK)"))
        if col_name == "Mean Pairwise (Dmg)" and check_better(dmg_val, ok_val): return base_style + "background-color:#dcfce7;color:#166534;font-weight:600;"
        if col_name == "Mean Pairwise (OK)" and not check_better(dmg_val, ok_val): return base_style + "background-color:#dcfce7;color:#166534;font-weight:600;"
    return base_style

def df_to_html(df, title=""):
    header = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td style='{get_cell_style(col, row[col], row)}'>{row[col]}</td>" for col in df.columns)
        rows_html += f"<tr>{cells}</tr>"
    return f"<div class='section'><h2>{title}</h2><div style='overflow-x:auto'><table class='styled-table'><thead><tr>{header}</tr></thead><tbody>{rows_html}</tbody></table></div></div>"

def wrap_chart(title, pio_html):
    return f"<div class='section'><h2>{title}</h2>{pio_html}</div>"

css_block = """
<style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: 'Inter', system-ui, sans-serif; font-size: 15px; background: #f1f5f9; color: #0f172a; margin: 0; padding: 40px 0 72px; }
    .container { max-width: 1440px; margin: 0 auto; padding: 0 64px; }
    h1 { font-size: 32px; font-weight: 800; letter-spacing: -0.6px; margin: 0 0 6px; }
    h2 { font-size: 20px; font-weight: 700; margin: 0 0 14px; letter-spacing: -0.2px; }
    .page-subtitle { color: #64748b; font-size: 15px; margin: 0 0 30px; }
    .section, .meta { background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 22px 26px; margin-bottom: 22px; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
    .meta { font-size: 14px; line-height: 2.1; }
    .meta b { color: #334155; }
    .styled-table { width: 100%; border-collapse: collapse; font-size: 14px; }
    .styled-table th, .styled-table td { border-bottom: 1px solid #e2e8f0; padding: 11px 10px; text-align: left; vertical-align: middle; }
    .styled-table th { background: #f8fafc; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #475569; }
    ul { margin: 8px 0 0; padding-left: 20px; }
    li { margin-bottom: 5px; font-size: 14px; color: #475569; }
    p { margin: 0 0 8px 0; color: #475569; font-size: 14px; line-height: 1.6; }
    code { font-family: monospace; background: #f1f5f9; padding: 2px 4px; border-radius: 4px; font-size: 13px; color: #dc2626; }
</style>
"""

filter_text = f"Top {NUM_TOP_MODELS} Base Models (by base_combined_Hit@{TOP_SELECTION_K})" if NUM_TOP_MODELS else "All Base Models (Unfiltered)"

header_html = f"""
    <h1>Geometric Diagnosis Dashboard</h1>
    <p class="page-subtitle">
        {FORGET_PERCENTAGE}% forget split &nbsp;·&nbsp; K = {TOP_SELECTION_K} &nbsp;·&nbsp; Bins = {SIMILARITY_BINS}
    </p>

    <div class="meta">
        <div><b>Folder:</b> C:/Bob/results/{FORGET_PERCENTAGE}_percent</div>
        <div><b>Model Selection:</b> Top run per method strictly complying with retain_drop &lt; {MAX_RETAIN_DROP_PP} pp.</div>
        <div><b>Base Model Filter:</b> <span style="color:#0f172a; font-weight:600;">{filter_text}</span></div>
        <div><b>Distance Metric:</b> <span style="color:#0f172a; font-weight:600;">{DISTANCE_METRIC.upper()} ({METRIC_TYPE})</span></div>
    </div>
"""

explanation_html = f"""
    <div class="section">
        <h2>🔬 Geometric Diagnosis Hypothesis</h2>
        <p><strong>Hypothesis:</strong> The retain users who lost performance (Damaged) are geometrically <strong>closer</strong> to the forget users in the shared feature space than the unaffected retain users (OK).</p>
        <p><strong>State representation:</strong> <code>build_state_fn</code> at trajectory midpoint = <code>[one-hot demographics | normalised genre preference vector]</code> (dim = {state_dim}).</p>
        <ul>
            <li><strong>Centroid proximity</strong> — Distance of each retain user to the <i>mean</i> forget user state.</li>
            <li><strong>Pairwise proximity</strong> — Distance to the <i>closest individual</i> forget user.</li>
        </ul>
        <p><strong>Statistical test:</strong> Mann-Whitney U (one-sided, H₁: damaged is closer than OK). Pearson r measures the linear correlation between distance and Hit delta.</p>
    </div>
"""

pearson_explanation_html = f"""
    <div class="section" style="border-left: 4px solid #2563eb;">
        <h2>💡 Understanding Pearson Correlation ({METRIC_TYPE})</h2>
        <p>
            <strong>Pearson r (Correlation Coefficient):</strong> Measures the linear relationship between a user's proximity to the forget group and their drop in performance (Hit Delta). 
            Because a drop in performance is expressed as a <em>negative</em> delta:
        </p>
        <ul>
            <li>For <strong>similarity metrics (e.g. Cosine)</strong>: Expected <strong>negative r</strong> (higher similarity correlates with more negative delta).</li>
            <li>For <strong>distance metrics (e.g. Euclidean)</strong>: Expected <strong>positive r</strong> (lower distance correlates with more negative delta).</li>
        </ul>
        <p>
            <strong>Pearson p (p-value):</strong> Indicates if the observed correlation is statistically significant. 
        </p>
        <ul>
            <li><strong>Significant (p &lt; 0.05):</strong> The correlation is strong enough that it is highly unlikely to be random chance. We can confidently say proximity to forget users affects performance.</li>
            <li><strong>Not Significant (p &ge; 0.05):</strong> There is not enough statistical evidence to confirm a relationship.</li>
        </ul>
    </div>
"""

html_parts = [
    "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width,initial-scale=1'/>",
    f"<title>Unlearning Diagnosis — {FORGET_PERCENTAGE}%</title>",
    "<link rel='preconnect' href='https://fonts.googleapis.com'><link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap' rel='stylesheet'>",
    css_block, "</head><body><div class='container'>", 
    header_html, explanation_html,
    df_to_html(summary_df, "📊 Best Model Per Method — Summary"),
    df_to_html(stat_df, "📈 Statistical Tests (Mann-Whitney U + Pearson r)"),
    pearson_explanation_html,
    wrap_chart("🗺 PCA Projection — State Space by Group", pio.to_html(fig_pca, full_html=False, include_plotlyjs="cdn")),
    wrap_chart("📊 Centroid Proximity Distributions (Binned)", pio.to_html(fig_dist_c, full_html=False, include_plotlyjs=False)),
    wrap_chart("📊 Pairwise Proximity Distributions (Binned)", pio.to_html(fig_dist_p, full_html=False, include_plotlyjs=False)),
    "</div></body></html>"
]

with open(DIAG_HTML, "w", encoding="utf-8") as f: f.write("\n".join(html_parts))
print(f"\n✅ Dashboard saved → {DIAG_HTML}")