#!/usr/bin/env python
# coding: utf-8
# C:\Bob\tools\verify.py

import hashlib
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import sys
import time
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
        "Usage: python verify.py <forget_pct> <max_retain_drop> [tolerance]\n"
        "Example: python verify.py 20 0.09\n"
        "         python verify.py 20 0.09 1e-6\n"
        "  max_retain_drop : filter threshold used in sequence_overlap.py\n"
        "  tolerance       : max allowed absolute diff for metrics (default 1e-8)\n"
    )

FORGET_PERCENTAGE = int(sys.argv[1])
MAX_RETAIN_DROP   = float(sys.argv[2])
TOLERANCE         = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8

METHODS = ["Ye_ApxI", "Ye_multi", "New_True_inf", "New_Max"]
EVAL_K  = 10
KS      = [1, 5, 10]

# ---------------------------------------------------------------------------
# Paths  — always read merged files, never per-worker files
# ---------------------------------------------------------------------------
DATA_DIR     = "C:/Bob/ml-1m"
RESULTS_BASE = Path(f"D:/Bob_Skripsi_Do Not Delete/results_demography/{FORGET_PERCENTAGE}_percent")
OUT_BASE = Path(f"D:/Bob_Skripsi_Do Not Delete/Analysis/Demography/{FORGET_PERCENTAGE}_percent")
OUT_BASE.mkdir(parents=True, exist_ok=True)

import sys
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger(OUT_BASE / f"verify_{MAX_RETAIN_DROP}.txt")

MODELS_DIR   = RESULTS_BASE / "models"

# Merged Phase-1 results (written by merge script, contains all workers)
TRAIN_RESULTS_MERGED = RESULTS_BASE / "train_phase_results.csv"
if not TRAIN_RESULTS_MERGED.exists():
    raise FileNotFoundError(f"Merged train results not found: {TRAIN_RESULTS_MERGED}")

# Merged Phase-2 results
RESULTS_MERGED = RESULTS_BASE / "tuning_full_results.csv"
if not RESULTS_MERGED.exists():
    raise FileNotFoundError(f"Merged unlearn results not found: {RESULTS_MERGED}")

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓ PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗ FAIL{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")
def info(msg): print(f"  {CYAN}ℹ{RESET}      {msg}")
def sep(title=""):
    width = 62
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{BOLD}{'─'*pad} {title} {'─'*(width-pad-len(title)-1)}{RESET}")
    else:
        print(f"\n{'─'*width}")

pass_count = 0
fail_count = 0

def record(passed, msg):
    global pass_count, fail_count
    if passed:
        ok(msg)
        pass_count += 1
    else:
        fail(msg)
        fail_count += 1
    return passed

# ---------------------------------------------------------------------------
# Constants  — must mirror GPU_Enabled_Combine.py exactly
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

# Mirror the exact determinism flags from main script
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.use_deterministic_algorithms(True, warn_only=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Data loading
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

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(ratings_df, movies_df, users_df, sample_users):
    pilot_users_df = users_df[users_df["user_id"].isin(sample_users)]
    user_cats      = pilot_users_df[["user_id", "gender", "age", "occupation"]].copy()
    oh             = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    user_cat_mat   = oh.fit_transform(user_cats[["gender", "age", "occupation"]])
    user_feat_df   = pd.DataFrame(user_cat_mat, index=user_cats["user_id"])

    all_genres   = sorted({g for s in movies_df["genres"].astype(str) for g in s.split("|")})
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    num_genres   = len(all_genres)

    def movie_genre_vector(genres_str):
        v = np.zeros(num_genres, dtype=np.float32)
        for g in str(genres_str).split("|"):
            if g in genre_to_idx:
                v[genre_to_idx[g]] = 1.0
        return v

    movies_df = movies_df.copy()
    movies_df["genre_vec"] = movies_df["genres"].apply(movie_genre_vector)
    movie_genre_map = {
        mid: movies_df.loc[movies_df["movie_id"] == mid, "genre_vec"].values[0]
        for mid in movies_df["movie_id"].unique()
    }
    return user_feat_df, movie_genre_map, num_genres

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
# Evaluation  — matches GPU_Enabled_Combine.py exactly
# NOTE: uses global build_state_fn and candidate_movies (mirrors main script)
# NOTE: argsort uses kind="stable" to match main script
# ---------------------------------------------------------------------------
def evaluate_policy(policy_net, trajectories, K=10):
    hits, ndcgs = [], []
    _candidates = np.array(candidate_movies)
    policy_net.eval()
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
        topk = _candidates[np.argsort(-probs, kind="stable")[:K]]   # ← kind="stable"
        hits.append(int(any(m in future for m in topk)))
        dcg  = sum(1.0 / np.log2(r + 2) for r, m in enumerate(topk) if m in future)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(future), K)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return (
        float(np.mean(hits))  if hits  else 0.0,
        float(np.mean(ndcgs)) if ndcgs else 0.0,
    )

def eval_all_ks(net):
    """Mirrors GPU_Enabled_Combine.py: eval_all_ks(net, ret_trajs, for_trajs, all_trajs)
    — build_state_fn and candidate_movies are global in the main script too."""
    out = {}
    for K in KS:
        h_r, n_r = evaluate_policy(net, retain_trajectories,  K=K)
        h_f, n_f = evaluate_policy(net, forget_trajectories,  K=K)
        h_c, n_c = evaluate_policy(net, trajectories_all, K=K)
        out[K] = (h_r, n_r, h_f, n_f, h_c, n_c)
    return out

# ---------------------------------------------------------------------------
# Filename helpers  — must mirror GPU_Enabled_Combine.py exactly
# NOTE: unlearned_model_path arg order changed:
#       OLD: (method, t_lr, γ, h, bs, u_lr, u_iters, lam)
#       NEW: (t_lr, γ, h, bs, method, u_lr, u_iters, lam)  ← method after bs
# ---------------------------------------------------------------------------
def _fmt(v):
    if v < 0.01:
        return f"{v:.0e}".replace("-", "n").replace("+", "p")
    return str(v).replace(".", "d")

def trained_model_path(t_lr, gamma, hidden_dim, train_bs):
    name = (
        f"trained__tlr{_fmt(t_lr)}__g{_fmt(gamma)}"
        f"__h{hidden_dim}__bs{train_bs}.pt"
    )
    return MODELS_DIR / name

def unlearned_model_path(t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam):
    name = (
        f"unlearn__{method}__tlr{_fmt(t_lr)}__g{_fmt(gamma)}"
        f"__h{hidden_dim}__bs{train_bs}__ulr{_fmt(u_lr)}"
        f"__ui{u_iters}__lam{_fmt(lam)}.pt"
    )
    return MODELS_DIR / name

# ===========================================================================
# BEGIN VERIFICATION
# ===========================================================================
print(f"\n{BOLD}{'='*62}")
print(f"  Seed & Metric Verification — {FORGET_PERCENTAGE}% Forget")
print(f"{'='*62}{RESET}")
print(f"  SEED            : {SEED}")
print(f"  max_retain_drop : {MAX_RETAIN_DROP}")
print(f"  Tolerance       : {TOLERANCE}")
print(f"  Device          : {DEVICE}")
print(f"  deterministic   : {torch.are_deterministic_algorithms_enabled()}")

# ---------------------------------------------------------------------------
sep("1. Data Loading")
# ---------------------------------------------------------------------------
info("Loading ML-1M dataset...")
t0 = time.time()
ratings_df, movies_df, users_df = load_data(DATA_DIR)
info(f"Loaded in {time.time()-t0:.2f}s")

# Mirror main script: pilot_ratings used for split size, pilot_ratings_all for trajectories
sample_users      = ratings_df["user_id"].unique().tolist()
pilot_ratings     = ratings_df[ratings_df["user_id"].isin(sample_users)].copy()
pilot_ratings.sort_values(["user_id", "timestamp"], inplace=True)
pilot_ratings_all = ratings_df[ratings_df["user_id"].isin(sample_users)].copy()
pilot_ratings_all.sort_values(["user_id", "timestamp"], inplace=True)

record(len(sample_users) > 0, f"Loaded {len(sample_users):,} unique users")

# ---------------------------------------------------------------------------
sep("2. Split Determinism  (run × 3)")
# ---------------------------------------------------------------------------
def compute_split():
    _users_meta_v = users_df[users_df["user_id"].isin(sample_users)].set_index("user_id")
    _sorted_users = np.array(sorted(sample_users))

    _uid_mult = []
    for uid in _sorted_users:
        if uid not in _users_meta_v.index:
            m = 1.0
        else:
            row = _users_meta_v.loc[uid]
            m = get_multiplier_demo(row["gender"], row["age"], row["occupation"])
        _uid_mult.append((uid, m))

    _mean_mult = float(np.mean([m for _, m in _uid_mult]))
    _base_prob = (FORGET_PERCENTAGE / 100) / _mean_mult

    _rng = np.random.default_rng(make_seed("forget_split"))
    forget_list, retain_list = [], []
    for uid, mult in _uid_mult:
        p = min(_base_prob * mult, 0.30)
        if _rng.random() < p:
            forget_list.append(uid)
        else:
            retain_list.append(uid)
    return forget_list, retain_list

def get_multiplier_demo(gender, age, occupation):
    m = 1.0
    if gender == "F":       m *= 1.36
    if age == 1:            m *= 1.0
    elif age == 18:         m *= 1.65
    elif age == 25:         m *= 1.60
    elif age == 35:         m *= 1.55
    elif age == 45:         m *= 1.22
    elif age == 50:         m *= 1.0
    elif age == 56:         m *= 0.92
    if occupation in (1, 4, 6, 10, 11, 15): m *= 1.22
    return m

forget_a, retain_a = compute_split()
forget_b, retain_b = compute_split()
forget_c, retain_c = compute_split()

record(forget_a == forget_b == forget_c,
       f"Forget split identical across 3 calls ({len(forget_a):,} users)")
record(retain_a == retain_b == retain_c,
       f"Retain split identical across 3 calls ({len(retain_a):,} users)")

n_total  = pilot_ratings["user_id"].nunique()
record(
    abs(len(forget_a) - int(FORGET_PERCENTAGE / 100 * n_total)) <= int(0.02 * n_total),
    f"Forget size approx correct: {len(forget_a)} ≈ {int(FORGET_PERCENTAGE/100*n_total)} "
    f"({FORGET_PERCENTAGE}% of {n_total:,})"
)
record(len(set(forget_a) & set(retain_a)) == 0,
       "Forget ∩ retain = ∅ (disjoint)")
record(len(set(forget_a) | set(retain_a)) == n_total,
       f"Forget ∪ retain = all {n_total:,} users (no leakage)")

split_hash = hashlib.sha256(
    (",".join(str(u) for u in sorted(forget_a))).encode()
).hexdigest()[:16]
info(f"Split fingerprint : {split_hash}")

# ---------------------------------------------------------------------------
sep("3. make_seed() Stability")
# ---------------------------------------------------------------------------
seeds_1 = [make_seed("train", t, g) for t, g in [(1e-3, 0.99), (1e-4, 0.98), (1e-5, 0.97)]]
seeds_2 = [make_seed("train", t, g) for t, g in [(1e-3, 0.99), (1e-4, 0.98), (1e-5, 0.97)]]
record(seeds_1 == seeds_2, f"make_seed() stable across calls: {seeds_1}")
record(len(set(make_seed(i) for i in range(20))) == 20,
       "make_seed(0..19) → 20 distinct values (no collisions)")

# ---------------------------------------------------------------------------
sep("4. Feature Engineering Determinism")
# ---------------------------------------------------------------------------
uf_a, mgm_a, ng_a = build_features(ratings_df, movies_df, users_df, sample_users)
uf_b, mgm_b, ng_b = build_features(ratings_df, movies_df, users_df, sample_users)

record(uf_a.equals(uf_b), "User feature matrix identical across 2 builds")
record(ng_a == ng_b,      f"num_genres identical: {ng_a}")
sample_mid = list(mgm_a.keys())[0]
record(np.array_equal(mgm_a[sample_mid], mgm_b[sample_mid]),
       f"Genre vector identical for movie_id={sample_mid}")

# Assign globals (needed by evaluate_policy / eval_all_ks)
user_feat_df   = uf_a
movie_genre_map = mgm_a
num_genres     = ng_a
state_dim      = user_feat_df.shape[1] + num_genres

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
sep("5. Trajectory + State Determinism")
# ---------------------------------------------------------------------------
def build_trajectories():
    return [
        {"user_id": uid, "movies": g["movie_id"].tolist(), "ratings": g["rating"].tolist()}
        for uid, g in pilot_ratings_all.groupby("user_id")
        if len(g) >= 5
    ]

trajs_x = build_trajectories()
trajs_y = build_trajectories()
h_x = hashlib.sha256(str([(t["user_id"], t["movies"][:5]) for t in trajs_x]).encode()).hexdigest()[:16]
h_y = hashlib.sha256(str([(t["user_id"], t["movies"][:5]) for t in trajs_y]).encode()).hexdigest()[:16]
record(h_x == h_y, f"Trajectory construction identical (hash={h_x})")
record(len(trajs_x) == len(trajs_y), f"Trajectory count identical: {len(trajs_x):,}")

trajectories_all    = trajs_x
forget_user_set     = set(forget_a)
retain_user_set     = set(retain_a)
forget_trajectories = [t for t in trajectories_all if t["user_id"] in forget_user_set]
retain_trajectories  = [t for t in trajectories_all if t["user_id"] in retain_user_set]
candidate_movies    = np.array(sorted(pilot_ratings_all["movie_id"].unique()))

test_users       = [retain_trajectories[i]["user_id"]    for i in range(min(5, len(retain_trajectories)))]
test_movie_lists = [retain_trajectories[i]["movies"][:3] for i in range(min(5, len(retain_trajectories)))]
states_ok = all(
    np.array_equal(build_state_fn(u, m), build_state_fn(u, m))
    for u, m in zip(test_users, test_movie_lists)
)
record(states_ok, f"build_state_fn identical across repeated calls ({len(test_users)} users tested)")
record(state_dim == user_feat_df.shape[1] + num_genres,
       f"state_dim = {state_dim} = user_feat({user_feat_df.shape[1]}) + genres({num_genres})")

info(f"Combined trajectories: {len(trajectories_all):,}")
info(f"Forget trajectories  : {len(forget_trajectories):,}")
info(f"Retain trajectories  : {len(retain_trajectories):,}")
info(f"Candidate movies     : {len(candidate_movies):,}")

# ---------------------------------------------------------------------------
sep("6. Eval Determinism  (best trained model × 3)")
# ---------------------------------------------------------------------------
train_res_df = pd.read_csv(TRAIN_RESULTS_MERGED)
res_df       = pd.read_csv(RESULTS_MERGED)

_TRAIN_KEY_COLS = ["train_lr", "gamma", "hidden_dim", "train_batch", "K"]
train_res_df = train_res_df.drop_duplicates(subset=_TRAIN_KEY_COLS, keep="last")

k10 = res_df[res_df["K"] == EVAL_K].copy()
k10["retain_drop"] = k10["base_retain_Hit"] - k10["retain_Hit"]
k10["forget_drop"] = k10["base_forget_Hit"] - k10["forget_Hit"]

det_model_path = None
for method in METHODS:
    eligible = k10[(k10["method"] == method) & (k10["retain_drop"] <= MAX_RETAIN_DROP)]
    if eligible.empty:
        continue
    best = eligible.sort_values("forget_drop", ascending=False).iloc[0]
    p    = trained_model_path(
        best["train_lr"], best["gamma"],
        int(best["hidden_dim"]), int(best["train_batch"])
    )
    if p.exists():
        det_model_path = p
        det_hidden     = int(best["hidden_dim"])
        break

if det_model_path is None:
    warn("No trained model found for eval determinism check — skipping section 6")
else:
    info(f"Using model: {det_model_path.name}")

    def load_and_eval():
        set_seed(SEED)
        net = PolicyNet(state_dim, len(candidate_movies), det_hidden).to(DEVICE)
        net.load_state_dict(torch.load(det_model_path, map_location=DEVICE))
        net.eval()
        return eval_all_ks(net)   # ← matches new signature (globals only)

    res_1 = load_and_eval()
    res_2 = load_and_eval()
    res_3 = load_and_eval()

    for K in KS:
        r1, r2, r3 = res_1[K], res_2[K], res_3[K]
        all_match  = all(
            abs(r1[i] - r2[i]) < TOLERANCE and abs(r1[i] - r3[i]) < TOLERANCE
            for i in range(6)
        )
        record(all_match,
               f"eval_all_ks K={K} bit-identical × 3  "
               f"retain_Hit={r1[0]:.6f}  forget_Hit={r1[2]:.6f}  combined_Hit={r1[4]:.6f}")
        if not all_match:
            for i, name in enumerate(["retain_Hit", "retain_NDCG", "forget_Hit", "forget_NDCG", "combined_Hit", "combined_NDCG"]):
                d12, d13 = abs(r1[i]-r2[i]), abs(r1[i]-r3[i])
                if max(d12, d13) > TOLERANCE:
                    warn(f"    K={K} {name}: run1={r1[i]:.8f} run2={r2[i]:.8f} "
                         f"run3={r3[i]:.8f}  max_diff={max(d12,d13):.2e}")

# ---------------------------------------------------------------------------
sep("7. Best-Model Metric Integrity  (1 per method)")
# ---------------------------------------------------------------------------
info(f"Selecting best run per method with retain_drop <= {MAX_RETAIN_DROP}")
info("Verifying trained model vs train_phase_results.csv")
info("Verifying unlearned model vs tuning_full_results.csv\n")

_KEY_COLS = [
    "train_lr", "gamma", "hidden_dim", "train_batch",
    "unlearn_lr", "unlearn_iters", "lambda_retain", "method", "K",
]
res_df_dedup = res_df.drop_duplicates(subset=_KEY_COLS, keep="last")

for method in METHODS:
    print(f"\n  {BOLD}── {method}{RESET}")

    method_k10 = k10[k10["method"] == method]
    eligible   = method_k10[method_k10["retain_drop"] <= MAX_RETAIN_DROP]

    if eligible.empty:
        warn(f"No eligible runs for {method} — skipping")
        continue

    best       = eligible.sort_values("forget_drop", ascending=False).iloc[0]
    t_lr       = best["train_lr"]
    gamma      = best["gamma"]
    hidden_dim = int(best["hidden_dim"])
    train_bs   = int(best["train_batch"])
    u_lr       = best["unlearn_lr"]
    u_iters    = int(best["unlearn_iters"])
    lam        = best["lambda_retain"]

    info(f"  Selected: tlr={t_lr} g={gamma} h={hidden_dim} bs={train_bs} "
         f"ulr={u_lr} ui={u_iters} lam={lam}")
    info(f"  Stored  : retain_drop={best['retain_drop']:.4f}  "
         f"forget_drop={best['forget_drop']:.4f}")

    t_path = trained_model_path(t_lr, gamma, hidden_dim, train_bs)
    # NOTE: new arg order — method is 5th arg (after train_bs)
    ul_path = unlearned_model_path(t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam)

    # ── A. Trained model vs train_phase_results.csv ───────────────────
    if not t_path.exists():
        warn(f"  Trained checkpoint missing: {t_path.name}")
    else:
        set_seed(SEED)
        trained_net = PolicyNet(state_dim, len(candidate_movies), hidden_dim).to(DEVICE)
        trained_net.load_state_dict(torch.load(t_path, map_location=DEVICE))
        trained_net.eval()
        recomp_trained = eval_all_ks(trained_net)
        del trained_net
        torch.cuda.empty_cache()

        trained_ok = True
        max_diff   = 0.0
        for K in KS:
            mask = (
                (train_res_df["train_lr"]      == t_lr)
                & (train_res_df["gamma"]        == gamma)
                & (train_res_df["hidden_dim"]   == hidden_dim)
                & (train_res_df["train_batch"]  == train_bs)
                & (train_res_df["K"]            == K)
            )
            rows = train_res_df[mask]
            if rows.empty:
                warn(f"  No stored train result for K={K}")
                continue
            stored = rows.iloc[0]
            for col, val in [
                ("base_retain_Hit",  recomp_trained[K][0]),
                ("base_retain_NDCG", recomp_trained[K][1]),
                ("base_forget_Hit",  recomp_trained[K][2]),
                ("base_forget_NDCG", recomp_trained[K][3]),
                ("base_combined_Hit",  recomp_trained[K][4]),
                ("base_combined_NDCG", recomp_trained[K][5]),
            ]:
                diff     = abs(float(stored[col]) - val)
                max_diff = max(max_diff, diff)
                if diff > TOLERANCE:
                    trained_ok = False
                    warn(f"    K={K} {col}: stored={stored[col]:.8f}  "
                         f"recomp={val:.8f}  diff={diff:.2e}")

        record(trained_ok,
               f"  [{method}] Trained model matches train_phase_results.csv  "
               f"(max_diff={max_diff:.2e})")

    # ── B. Unlearned model vs tuning_full_results.csv ─────────────────
    if not ul_path.exists():
        warn(f"  Unlearned checkpoint missing: {ul_path.name}")
    else:
        set_seed(SEED)
        unlearned_net = PolicyNet(state_dim, len(candidate_movies), hidden_dim).to(DEVICE)
        unlearned_net.load_state_dict(torch.load(ul_path, map_location=DEVICE))
        unlearned_net.eval()
        recomp_ul = eval_all_ks(unlearned_net)
        del unlearned_net
        torch.cuda.empty_cache()

        unlearned_ok = True
        max_diff     = 0.0
        for K in KS:
            mask = (
                (res_df_dedup["train_lr"]       == t_lr)
                & (res_df_dedup["gamma"]         == gamma)
                & (res_df_dedup["hidden_dim"]    == hidden_dim)
                & (res_df_dedup["train_batch"]   == train_bs)
                & (res_df_dedup["unlearn_lr"]    == u_lr)
                & (res_df_dedup["unlearn_iters"] == u_iters)
                & (res_df_dedup["lambda_retain"] == lam)
                & (res_df_dedup["method"]        == method)
                & (res_df_dedup["K"]             == K)
            )
            rows = res_df_dedup[mask]
            if rows.empty:
                warn(f"  No stored unlearn result for K={K}")
                continue
            stored = rows.iloc[0]
            for col, val in [
                ("retain_Hit",  recomp_ul[K][0]),
                ("retain_NDCG", recomp_ul[K][1]),
                ("forget_Hit",  recomp_ul[K][2]),
                ("forget_NDCG", recomp_ul[K][3]),
            ]:
                diff     = abs(float(stored[col]) - val)
                max_diff = max(max_diff, diff)
                if diff > TOLERANCE:
                    unlearned_ok = False
                    warn(f"    K={K} {col}: stored={stored[col]:.8f}  "
                         f"recomp={val:.8f}  diff={diff:.2e}")

        record(unlearned_ok,
               f"  [{method}] Unlearned model matches tuning_full_results.csv  "
               f"(max_diff={max_diff:.2e})")

# ---------------------------------------------------------------------------
sep("SUMMARY")
# ---------------------------------------------------------------------------
total = pass_count + fail_count
print(f"\n  {BOLD}Total checks : {total}{RESET}")
print(f"  {GREEN}{BOLD}Passed : {pass_count}{RESET}")
if fail_count > 0:
    print(f"  {RED}{BOLD}Failed : {fail_count}{RESET}")
    print(f"\n  {RED}⚠  Seed/metric integrity NOT confirmed.{RESET}")
    print(f"  {RED}   Do not mix CSVs from different workers or runs.{RESET}")
else:
    print(f"  {GREEN}{BOLD}Failed : 0{RESET}")
    print(f"\n  {GREEN}{'='*58}{RESET}")
    print(f"  {GREEN}{BOLD}✓ All checks passed — results are fully reproducible.{RESET}")
    print(f"  {GREEN}  Split fingerprint : {split_hash}{RESET}")
    print(f"  {GREEN}{'='*58}{RESET}")

sys.exit(0 if fail_count == 0 else 1)