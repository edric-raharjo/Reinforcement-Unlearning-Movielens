#!/usr/bin/env python
# coding: utf-8
"""
diagnosis.py  <forget_pct>  <max_retain_drop_pp>  [results_csv_name]

Geometric diagnosis: are damaged retain users closer to forget users in state space?
Produces an interactive HTML dashboard at:
    C:/Bob/results_deterministic/<forget_pct>_percent/diagnosis_dashboard.html
"""

import copy, hashlib, os, pickle, sys, warnings
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
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if len(sys.argv) < 3:
    raise ValueError("Usage: python diagnosis.py <forget_pct> <max_retain_drop_pp> [results_csv_name]")

FORGET_PERCENTAGE    = int(sys.argv[1])
MAX_RETAIN_DROP_PP   = float(sys.argv[2])          # e.g. 4.0  → ≤4 pp
RESULTS_CSV_NAME     = sys.argv[3] if len(sys.argv) > 3 else "tuning_full_results.csv"

SIMILARITY_BINS      = 5  # Splits data into 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 bins

# ---------------------------------------------------------------------------
# Seed — identical to training script
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR     = "C:/Bob/ml-1m"
RESULTS_BASE = f"C:/Bob/results/{FORGET_PERCENTAGE}_percent"
MODELS_DIR   = os.path.join(RESULTS_BASE, "models")
ANALYZE_DIR  = os.path.join(RESULTS_BASE, "analyze", "diagnose")
os.makedirs(ANALYZE_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(RESULTS_BASE, RESULTS_CSV_NAME)
DIAG_HTML    = os.path.join(ANALYZE_DIR, "diagnosis_dashboard.html")
DIAG_CSV     = os.path.join(ANALYZE_DIR, "diagnosis_stats.csv")

TOP_SELECTION_K = 10
KS              = [1, 5, 10]
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METHODS         = ["Ye_ApxI", "Ye_multi", "New_True_inf", "New_Max"]

print(f"Device        : {DEVICE}")
print(f"Forget %      : {FORGET_PERCENTAGE}")
print(f"Max retain drop: {MAX_RETAIN_DROP_PP} pp")
print(f"Results CSV   : {RESULTS_PATH}")

# ---------------------------------------------------------------------------
# Data loading — identical to training script
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

# ---------------------------------------------------------------------------
# Feature engineering — identical to training script
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Forget / Retain split
# ---------------------------------------------------------------------------
sample_users_arr = np.array(sample_users)
np.random.shuffle(sample_users_arr)

split_amt      = int(np.round(FORGET_PERCENTAGE / 100 * pilot_ratings["user_id"].nunique()))
forget_users   = sample_users_arr[:split_amt]
retain_users   = sample_users_arr[split_amt:]

print(f"\nForget users : {len(forget_users)}")
print(f"Retain users : {len(retain_users)}")

pilot_ratings_all = ratings_df[ratings_df["user_id"].isin(sample_users)].copy()
pilot_ratings_all.sort_values(["user_id", "timestamp"], inplace=True)

trajectories_all = [
    {"user_id": uid, "movies": g["movie_id"].tolist(), "ratings": g["rating"].tolist()}
    for uid, g in pilot_ratings_all.groupby("user_id")
    if len(g) >= 5
]
forget_user_set     = set(forget_users.tolist())
retain_user_set     = set(retain_users.tolist())
forget_trajectories = [t for t in trajectories_all if t["user_id"] in forget_user_set]
retain_trajectories = [t for t in trajectories_all if t["user_id"] in retain_user_set]

candidate_movies = np.array(sorted(pilot_ratings_all["movie_id"].unique()))
num_actions      = len(candidate_movies)

# ---------------------------------------------------------------------------
# PolicyNet
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

def _fmt(v):
    if v < 0.01:
        return f"{v:.0e}".replace("-", "n").replace("+", "p")
    return str(v).replace(".", "d")

def trained_model_path(t_lr, gamma, hidden_dim, train_bs):
    name = f"trained__tlr{_fmt(t_lr)}__g{_fmt(gamma)}__h{hidden_dim}__bs{train_bs}.pt"
    return os.path.join(MODELS_DIR, name)

def unlearned_model_path(t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam):
    name = (
        f"unlearn__{method}__tlr{_fmt(t_lr)}__g{_fmt(gamma)}"
        f"__h{hidden_dim}__bs{train_bs}__ulr{_fmt(u_lr)}"
        f"__ui{u_iters}__lam{_fmt(lam)}.pt"
    )
    return os.path.join(MODELS_DIR, name)

# ===========================================================================
# PHASE 1 — State vectors at trajectory midpoint
# ===========================================================================
print("\n" + "="*60)
print("PHASE 1 — Computing user state vectors (midpoint)")
print("="*60)

def compute_user_states(trajectories, desc="states"):
    states = {}
    for traj in tqdm(trajectories, desc=desc):
        uid    = traj["user_id"]
        movies = traj["movies"]
        if len(movies) < 5:
            continue
        mid = len(movies) // 2
        states[uid] = build_state_fn(uid, movies[:mid])
    return states

forget_states = compute_user_states(forget_trajectories, "Forget states")
retain_states = compute_user_states(retain_trajectories, "Retain states")

forget_uids = list(forget_states.keys())
retain_uids = list(retain_states.keys())
forget_mat  = np.stack([forget_states[u] for u in forget_uids])   # (F, D)
retain_mat  = np.stack([retain_states[u] for u in retain_uids])   # (R, D)

retain_uid_to_idx = {uid: i for i, uid in enumerate(retain_uids)}
forget_uid_to_idx = {uid: i for i, uid in enumerate(forget_uids)}

forget_core_uids_per_method = {}   

# ===========================================================================
# PHASE 2 — Best model per method + per-user Hit delta
# ===========================================================================
print("\n" + "="*60)
print(f"PHASE 2 — Selecting best models (max_retain_drop ≤ {MAX_RETAIN_DROP_PP} pp)")
print("="*60)

results_df = pd.read_csv(RESULTS_PATH)
results_k  = results_df[results_df["K"] == TOP_SELECTION_K].copy()
results_k["forget_drop"] = results_k["base_forget_Hit"] - results_k["forget_Hit"]
results_k["retain_drop"] = results_k["base_retain_Hit"] - results_k["retain_Hit"]

constraint = MAX_RETAIN_DROP_PP / 100.0
best_rows  = {}

for method in METHODS:
    sub = results_k[
        (results_k["method"] == method) &
        (results_k["retain_drop"] <= constraint)
    ]
    if sub.empty:
        print(f"  ⚠ {method}: no config within constraint — using unconstrained best")
        sub = results_k[results_k["method"] == method]
    if sub.empty:
        print(f"  ✗ {method}: not found in results CSV — skipping")
        continue
    row = sub.loc[sub["forget_drop"].idxmax()]
    best_rows[method] = row

def evaluate_per_user(policy_net, trajectories, K=10, desc="eval"):
    hits = {}
    for traj in tqdm(trajectories, desc=desc, leave=False):
        uid    = traj["user_id"]
        movies = traj["movies"]
        if len(movies) < 5:
            continue
        mid    = len(movies) // 2
        future = set(movies[mid:])
        state  = build_state_fn(uid, movies[:mid])
        st     = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(policy_net(st), dim=-1).squeeze(0).cpu().numpy()
        topk   = candidate_movies[np.argsort(-probs, kind="stable")[:K]]
        hits[uid] = int(any(m in future for m in topk))
    return hits

per_method_data = {}

for method, row in best_rows.items():
    t_lr       = row["train_lr"];  gamma    = row["gamma"]
    hidden_dim = int(row["hidden_dim"]);  train_bs = int(row["train_batch"])
    u_lr       = row["unlearn_lr"];  u_iters  = int(row["unlearn_iters"])
    lam        = row["lambda_retain"]

    base_path    = trained_model_path(t_lr, gamma, hidden_dim, train_bs)
    unlearn_path = unlearned_model_path(t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam)

    net_base = PolicyNet(state_dim, num_actions, hidden_dim=hidden_dim).to(DEVICE)
    net_base.load_state_dict(torch.load(base_path, map_location=DEVICE))
    net_base.eval()

    net_ul = PolicyNet(state_dim, num_actions, hidden_dim=hidden_dim).to(DEVICE)
    net_ul.load_state_dict(torch.load(unlearn_path, map_location=DEVICE))
    net_ul.eval()

    r_before = evaluate_per_user(net_base, retain_trajectories, K=TOP_SELECTION_K, desc=f"  {method} retain base")
    r_after  = evaluate_per_user(net_ul,   retain_trajectories, K=TOP_SELECTION_K, desc=f"  {method} retain unl")
    f_before = evaluate_per_user(net_base, forget_trajectories, K=TOP_SELECTION_K, desc=f"  {method} forget base")
    f_after  = evaluate_per_user(net_ul,   forget_trajectories, K=TOP_SELECTION_K, desc=f"  {method} forget unl")

    r_uids = [u for u in r_before if u in r_after and u in retain_uid_to_idx]
    df_r   = pd.DataFrame({
        "uid":        r_uids,
        "hit_before": [r_before[u] for u in r_uids],
        "hit_after":  [r_after[u]  for u in r_uids],
        "delta":      [r_after[u] - r_before[u] for u in r_uids],
    })
    df_r["group"]   = df_r["delta"].apply(lambda d: "Damaged" if d < 0 else "OK")
    df_r["damaged"] = df_r["delta"] < 0

    f_uids = [u for u in f_before if u in f_after and u in forget_uid_to_idx]
    df_f   = pd.DataFrame({
        "uid":        f_uids,
        "hit_before": [f_before[u] for u in f_uids],
        "hit_after":  [f_after[u]  for u in f_uids],
        "delta":      [f_after[u] - f_before[u] for u in f_uids],
    })
    df_f["damaged"] = df_f["delta"] < 0

    if len(df_f) > 0:
        n_core = max(1, int(round(0.10 * len(df_f))))
        df_f_sorted = df_f.sort_values("delta")
        core_uids = df_f_sorted.head(n_core)["uid"].tolist()
    else:
        core_uids = []
    forget_core_uids_per_method[method] = core_uids

    per_method_data[method] = {"retain": df_r, "forget": df_f, "row": row}

    del net_base, net_ul
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ===========================================================================
# PHASE 3 — Cosine similarities
# ===========================================================================
print("\n" + "="*60)
print("PHASE 3 — Cosine similarity (centroid + pairwise max)")
print("="*60)

forget_norm = forget_mat / (np.linalg.norm(forget_mat, axis=1, keepdims=True) + 1e-10)
retain_norm = retain_mat / (np.linalg.norm(retain_mat, axis=1, keepdims=True) + 1e-10)

CHUNK = 256
def pairwise_max_sim(query_norm, key_norm, desc="pairwise max"):
    results = []
    for i in tqdm(range(0, len(query_norm), CHUNK), desc=desc):
        chunk  = query_norm[i : i + CHUNK]
        sims   = chunk @ key_norm.T
        results.append(sims.max(axis=1))
    return np.concatenate(results)

for method, d in per_method_data.items():
    core_uids = forget_core_uids_per_method.get(method, [])
    if not core_uids:
        continue

    core_idx   = [forget_uid_to_idx[u] for u in core_uids if u in forget_uid_to_idx]
    if not core_idx:
        continue

    core_mat   = forget_mat[core_idx]
    core_norm  = core_mat / (np.linalg.norm(core_mat, axis=1, keepdims=True) + 1e-10)

    core_centroid = core_norm.mean(axis=0)
    core_centroid = core_centroid / (np.linalg.norm(core_centroid) + 1e-10)
    retain_centroid_sims_m = retain_norm @ core_centroid

    retain_pairwise_max_m  = pairwise_max_sim(retain_norm, core_norm, f"Retain pairwise max ({method})")

    df_r = d["retain"].copy()
    df_r["centroid_sim"]     = df_r["uid"].apply(
        lambda u: float(retain_centroid_sims_m[retain_uid_to_idx[u]]) if u in retain_uid_to_idx else np.nan
    )
    df_r["pairwise_max_sim"] = df_r["uid"].apply(
        lambda u: float(retain_pairwise_max_m[retain_uid_to_idx[u]]) if u in retain_uid_to_idx else np.nan
    )
    per_method_data[method]["retain"] = df_r

# ===========================================================================
# PHASE 4 — Mann-Whitney U test
# ===========================================================================
print("\n" + "="*60)
print("PHASE 4 — Statistical tests (Mann-Whitney U)")
print("="*60)

stat_rows = []
for method, d in per_method_data.items():
    df_r = d["retain"]
    for sim_col, label in [("centroid_sim", "Centroid"), ("pairwise_max_sim", "Pairwise Max")]:
        dmg  = df_r[df_r["damaged"]][sim_col].dropna().values
        ok   = df_r[~df_r["damaged"]][sim_col].dropna().values
        if len(dmg) < 3 or len(ok) < 3:
            continue
        u_stat, p_val = stats.mannwhitneyu(dmg, ok, alternative="greater")
        corr_r, p_corr = stats.pearsonr(
            df_r[sim_col].dropna(),
            df_r.loc[df_r[sim_col].notna(), "delta"]
        )
        row_s = {
            "Method":                method,
            "Similarity Type":       label,
            "N Damaged":             len(dmg),
            "N OK":                  len(ok),
            "Mean Sim (Damaged)":    round(dmg.mean(), 4),
            "Mean Sim (OK)":         round(ok.mean(), 4),
            "Δ Mean":                round(dmg.mean() - ok.mean(), 4),
            "U Statistic":           round(u_stat, 1),
            "p-value":               round(p_val, 4),
            "Significant (p<.05)":   "✓" if p_val < 0.05 else "✗",
            "Pearson r (sim~delta)": round(corr_r, 4),
            "Pearson p":             round(p_corr, 4),
        }
        stat_rows.append(row_s)

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(DIAG_CSV, index=False)

# ===========================================================================
# PHASE 5 — HTML UI & Plot Generation (Aligned with dashboard_fair)
# ===========================================================================
print("\n" + "="*60)
print("PHASE 5 — Building HTML dashboard")
print("="*60)

# Colors tailored for Plotly charts matching the clean light dashboard look
PANEL   = "#ffffff"
TEXT    = "#0f172a"
GRIDCOL = "#e2e8f0"
COLORS  = {"Damaged": "#dc2626", "OK": "#2563eb", "Forget": "#d97706"} 

all_states  = np.vstack([forget_mat, retain_mat])
pca         = PCA(n_components=2, random_state=42)
coords      = pca.fit_transform(all_states)
forget_coords = coords[:len(forget_mat)]
retain_coords = coords[len(forget_mat):]
var_exp       = pca.explained_variance_ratio_

def make_pca_fig():
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"PCA — {m}" for m in METHODS],
                        horizontal_spacing=0.08, vertical_spacing=0.12)

    for idx, method in enumerate(METHODS):
        r, c   = divmod(idx, 2)
        df_r   = per_method_data[method]["retain"]
        uid2dmg = dict(zip(df_r["uid"], df_r["damaged"]))
        d_mask  = np.array([uid2dmg.get(u, False) for u in retain_uids])
        ok_mask = ~d_mask

        fig.add_trace(go.Scatter(
            x=forget_coords[:, 0], y=forget_coords[:, 1],
            mode="markers", marker=dict(color=COLORS["Forget"], size=4, opacity=0.4),
            name="Forget users", showlegend=(idx == 0), legendgroup="forget",
        ), row=r+1, col=c+1)

        fig.add_trace(go.Scatter(
            x=retain_coords[ok_mask, 0], y=retain_coords[ok_mask, 1],
            mode="markers", marker=dict(color=COLORS["OK"], size=4, opacity=0.4),
            name="Retain — OK", showlegend=(idx == 0), legendgroup="retain_ok",
        ), row=r+1, col=c+1)

        fig.add_trace(go.Scatter(
            x=retain_coords[d_mask, 0], y=retain_coords[d_mask, 1],
            mode="markers", marker=dict(color=COLORS["Damaged"], size=6, opacity=0.85),
            name="Retain — Damaged", showlegend=(idx == 0), legendgroup="retain_dmg",
        ), row=r+1, col=c+1)

    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color=TEXT, height=650,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor=PANEL, bordercolor=GRIDCOL, borderwidth=1),
        font=dict(family="Inter, system-ui, sans-serif")
    )
    fig.update_xaxes(gridcolor=GRIDCOL, zerolinecolor=GRIDCOL)
    fig.update_yaxes(gridcolor=GRIDCOL, zerolinecolor=GRIDCOL)
    return fig

def make_binned_distribution_fig(sim_col, title):
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{m}" for m in METHODS],
                        horizontal_spacing=0.08, vertical_spacing=0.14)
    bin_size = 1.0 / SIMILARITY_BINS
    for idx, method in enumerate(METHODS):
        r, c = divmod(idx, 2)
        if method not in per_method_data:
            continue
        df_r  = per_method_data[method]["retain"]
        dmg_v = df_r[df_r["damaged"]][sim_col].dropna().values
        ok_v  = df_r[~df_r["damaged"]][sim_col].dropna().values
        core_uids = forget_core_uids_per_method.get(method, [])
        fgt_v = np.array([])
        if len(core_uids) >= 2:
            core_idx = [forget_uid_to_idx.get(u) for u in core_uids if u in forget_uid_to_idx]
            if len(core_idx) >= 2:
                core_mat = forget_mat[core_idx]
                core_norm = core_mat / (np.linalg.norm(core_mat, axis=1, keepdims=True) + 1e-10)
                sims = core_norm @ core_norm.T
                fgt_self_sims = np.zeros(len(core_norm))
                for j in range(len(core_norm)):
                    other_sims = sims[j][sims[j] != sims[j,j]]
                    fgt_self_sims[j] = other_sims.max() if len(other_sims) > 0 else sims[j].max()
                fgt_v = fgt_self_sims
        for vals, label, color in [
            (fgt_v, "Core Forget (10%)", COLORS["Forget"]),
            (ok_v,  "Retain — OK",       COLORS["OK"]),
            (dmg_v, "Retain — Damaged",  COLORS["Damaged"]),
        ]:
            if len(vals) == 0:
                continue
            fig.add_trace(go.Histogram(
                x=vals, name=label,
                marker_color=color, opacity=0.75,
                xbins=dict(start=0.0, end=1.0, size=bin_size),
                histnorm='probability', 
                showlegend=(idx == 0), legendgroup=label,
            ), row=r+1, col=c+1)
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color=TEXT,
        height=650, barmode="group",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor=PANEL, bordercolor=GRIDCOL, borderwidth=1),
        font=dict(family="Inter, system-ui, sans-serif")
    )
    tick_vals = np.linspace(0.0, 1.0, SIMILARITY_BINS + 1)
    fig.update_xaxes(gridcolor=GRIDCOL, zerolinecolor=GRIDCOL, range=[-0.05, 1.05], tickvals=tick_vals)
    fig.update_yaxes(gridcolor=GRIDCOL, zerolinecolor=GRIDCOL, title_text="Probability Density")
    return fig

fig_pca      = make_pca_fig()
fig_dist_c   = make_binned_distribution_fig("centroid_sim", "Centroid Similarity Distribution")
fig_dist_p   = make_binned_distribution_fig("pairwise_max_sim", "Pairwise Max Similarity Distribution")

# Summary table per method
summary_rows = []
for method in METHODS:
    if method not in per_method_data:
        continue
    d    = per_method_data[method]
    row  = d["row"]
    df_r = d["retain"]
    n_d  = df_r["damaged"].sum()
    n_ok = (~df_r["damaged"]).sum()
    dmg  = df_r[df_r["damaged"]]
    ok   = df_r[~df_r["damaged"]]
    summary_rows.append({
        "Method":              method,
        "Forget Drop (pp)":    f"{row['forget_drop']*100:.2f}",
        "Retain Drop (pp)":    f"{row['retain_drop']*100:.2f}",
        "# Damaged Retain":    int(n_d),
        "# OK Retain":         int(n_ok),
        "Damaged %":           f"{n_d/(n_d+n_ok)*100:.1f}%",
        "Mean CentSim (Dmg)":  f"{dmg['centroid_sim'].mean():.4f}" if n_d > 0 else "N/A",
        "Mean CentSim (OK)":   f"{ok['centroid_sim'].mean():.4f}",
        "Mean PairSim (Dmg)":  f"{dmg['pairwise_max_sim'].mean():.4f}" if n_d > 0 else "N/A",
        "Mean PairSim (OK)":   f"{ok['pairwise_max_sim'].mean():.4f}",
    })
summary_df = pd.DataFrame(summary_rows)

def get_cell_style(col_name, val, row):
    base_style = ""
    
    def safe_float(v):
        try: return float(v)
        except: return -float('inf')

    # Statistical Tests conditional formatting
    if col_name == "Significant (p<.05)" and val == "✓":
        return base_style + "background-color:#fef08a;color:#854d0e;font-weight:bold;"
        
    if col_name in ["Mean Sim (Damaged)", "Mean Sim (OK)"]:
        dmg_val = safe_float(row.get("Mean Sim (Damaged)"))
        ok_val = safe_float(row.get("Mean Sim (OK)"))
        if col_name == "Mean Sim (Damaged)" and dmg_val > ok_val:
            return base_style + "background-color:#dcfce7;color:#166534;font-weight:600;"
        if col_name == "Mean Sim (OK)" and ok_val > dmg_val:
            return base_style + "background-color:#dcfce7;color:#166534;font-weight:600;"

    # Summary conditional formatting
    if col_name in ["Mean CentSim (Dmg)", "Mean CentSim (OK)"]:
        dmg_val = safe_float(row.get("Mean CentSim (Dmg)"))
        ok_val = safe_float(row.get("Mean CentSim (OK)"))
        if col_name == "Mean CentSim (Dmg)" and dmg_val > ok_val:
            return base_style + "background-color:#dcfce7;color:#166534;font-weight:600;"
        if col_name == "Mean CentSim (OK)" and ok_val > dmg_val:
            return base_style + "background-color:#dcfce7;color:#166534;font-weight:600;"

    if col_name in ["Mean PairSim (Dmg)", "Mean PairSim (OK)"]:
        dmg_val = safe_float(row.get("Mean PairSim (Dmg)"))
        ok_val = safe_float(row.get("Mean PairSim (OK)"))
        if col_name == "Mean PairSim (Dmg)" and dmg_val > ok_val:
            return base_style + "background-color:#dcfce7;color:#166534;font-weight:600;"
        if col_name == "Mean PairSim (OK)" and ok_val > dmg_val:
            return base_style + "background-color:#dcfce7;color:#166534;font-weight:600;"

    return base_style

def df_to_html(df, title=""):
    header = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            val = row[col]
            style = get_cell_style(col, val, row)
            cells += f"<td style='{style}'>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"
    return f"""
    <div class="section">
        <h2>{title}</h2>
        <div style='overflow-x:auto'>
            <table class="styled-table">
                <thead><tr>{header}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </div>
    """

def wrap_chart(title, pio_html):
    return f"""
    <div class="section">
        <h2>{title}</h2>
        {pio_html}
    </div>
    """

# Modern CSS matching dashboard_fair.py
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

header_html = f"""
    <h1>Geometric Diagnosis Dashboard</h1>
    <p class="page-subtitle">
        {FORGET_PERCENTAGE}% forget split &nbsp;·&nbsp; K = {TOP_SELECTION_K} &nbsp;·&nbsp; Bins = {SIMILARITY_BINS}
    </p>

    <div class="meta">
        <div><b>Folder:</b> C:/Bob/results/{FORGET_PERCENTAGE}_percent</div>
        <div><b>Model Selection:</b> Top run per method complying with max retain drop &le; {MAX_RETAIN_DROP_PP} pp.</div>
    </div>
"""

explanation_html = f"""
    <div class="section">
        <h2>🔬 Geometric Diagnosis Hypothesis</h2>
        <p><strong>Hypothesis:</strong> The retain users who lost performance (Damaged) are geometrically closer to the forget users in the shared feature space than the unaffected retain users (OK).</p>
        <p><strong>State representation:</strong> <code>build_state_fn</code> at trajectory midpoint = <code>[one-hot demographics | normalised genre preference vector]</code> (dim = {state_dim}).</p>
        <ul>
            <li><strong>Centroid similarity</strong> — cosine sim of each retain user to the <i>mean</i> forget user state.</li>
            <li><strong>Pairwise max similarity</strong> — cosine sim to the <i>closest individual</i> forget user.</li>
        </ul>
        <p><strong>Statistical test:</strong> Mann-Whitney U (one-sided, H₁: damaged &gt; undamaged similarity). Pearson r measures the linear correlation between similarity and Hit delta.</p>
    </div>
"""

pearson_explanation_html = f"""
    <div class="section" style="border-left: 4px solid #2563eb;">
        <h2>💡 Understanding Pearson Correlation</h2>
        <p>
            <strong>Pearson r (Correlation Coefficient):</strong> Measures the linear relationship between a user's similarity to the forget group and their drop in performance (Hit Delta). 
            Because a drop in performance is expressed as a <em>negative</em> delta, a <strong>negative r value</strong> means that <em>higher similarity</em> to the forget group correlates with a <em>larger performance drop</em>.
        </p>
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
    "<!DOCTYPE html>",
    "<html lang='en'>",
    "<head>",
    "    <meta charset='utf-8'/>",
    "    <meta name='viewport' content='width=device-width,initial-scale=1'/>",
    f"    <title>Unlearning Diagnosis — {FORGET_PERCENTAGE}%</title>",
    "    <link rel='preconnect' href='https://fonts.googleapis.com'>",
    "    <link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap' rel='stylesheet'>",
    css_block,
    "</head>",
    "<body>",
    "<div class='container'>", 
    header_html,
    explanation_html,
    df_to_html(summary_df, "📊 Best Model Per Method — Summary"),
    df_to_html(stat_df,    "📈 Statistical Tests (Mann-Whitney U + Pearson r)"),
    pearson_explanation_html,
    wrap_chart("🗺 PCA Projection — State Space by Group", pio.to_html(fig_pca, full_html=False, include_plotlyjs="cdn")),
    wrap_chart("📊 Centroid Similarity Distributions (Binned)", pio.to_html(fig_dist_c, full_html=False, include_plotlyjs=False)),
    wrap_chart("📊 Pairwise Max Similarity Distributions (Binned)", pio.to_html(fig_dist_p, full_html=False, include_plotlyjs=False)),
    "</div>",
    "</body></html>",
]

with open(DIAG_HTML, "w", encoding="utf-8") as f:
    f.write("\n".join(html_parts))

print(f"\n✅ Dashboard saved → {DIAG_HTML}")