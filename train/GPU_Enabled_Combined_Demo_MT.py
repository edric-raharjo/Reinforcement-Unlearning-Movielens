#!/usr/bin/env python
# coding: utf-8

import copy
import hashlib
import itertools
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import pickle
import random
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Global seed for reproducibility
# ---------------------------------------------------------------------------
SEED = 97620260313

def make_seed(*args):
    """Derive a deterministic 32-bit seed from SEED + args.
    Uses SHA-256 so it is stable across Python versions/runs
    (unlike the built-in hash() which is randomised since Python 3.3).
    """
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
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for cuBLAS determinism on CUDA >= 10.2
torch.use_deterministic_algorithms(True, warn_only=True)

# ---------------------------------------------------------------------------
# CLI argument
# ---------------------------------------------------------------------------

if len(sys.argv) < 2:
    raise ValueError(
        "Usage: python GPU_Enabled_Combine.py <forget_pct> [num_workers] [worker_id] [phase]"
    )

FORGET_PERCENTAGE = int(sys.argv[1])
NUM_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
WORKER_ID   = int(sys.argv[3]) if len(sys.argv) > 3 else 0
RUN_PHASE   = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # 0=both, 1=phase1 only, 2=phase2 only

assert 0 <= WORKER_ID < NUM_WORKERS, "worker_id must be in [0, num_workers)"

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_DIR = "C:/Bob/ml-1m"
RESULTS_BASE = f"D:/Bob_Skripsi_Do Not Delete/results_demography/{FORGET_PERCENTAGE}_percent"
MODELS_DIR = os.path.join(RESULTS_BASE, "models")
F_BUF_PATH = os.path.join(RESULTS_BASE, "forget_buffer.pkl")

_W = f"_w{WORKER_ID}" if NUM_WORKERS > 1 else ""

# Per-worker private files (no conflicts between parallel workers)
RESULTS_PATH          = os.path.join(RESULTS_BASE, f"tuning_full_results{_W}.csv")
PROGRESS_PATH         = os.path.join(RESULTS_BASE, f"progress{_W}.csv")
TRAIN_RESULTS_PATH    = os.path.join(RESULTS_BASE, f"train_phase_results{_W}.csv")
TRAIN_PROGRESS_PATH   = os.path.join(RESULTS_BASE, f"train_phase_progress{_W}.csv")
UNLEARN_LOSS_LOG_PATH = os.path.join(RESULTS_BASE, f"unlearning_loss_log{_W}.csv")
UNLEARN_LOG_INTERVAL = 25

# Merged files (written by merge script, read by Phase 2)
TRAIN_RESULTS_MERGED  = os.path.join(RESULTS_BASE, "train_phase_results.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

SAVE_UNLEARNED_MODELS = True
TOP_PERCENT = 0.1
TOP_SELECTION_METRIC = "base_combined_Hit"
TOP_SELECTION_K = 10

NUM_EPISODES = 10_000
LOG_INTERVAL = 100   # set to 25 too if you want Phase 1 training logs every 25
MAX_STEPS = 30
UNLEARN_BATCH = 64
KS = [1, 5, 10]
PATIENCE = max(10, int(0.1 * NUM_EPISODES / LOG_INTERVAL))

TRAIN_LRS = [1e-3, 1e-4, 1e-5]
GAMMAS = [0.99, 0.98, 0.97]
HIDDEN_DIMS = [128, 256, 512]
TRAIN_BATCH_SIZES = [1, 2, 4]

UNLEARN_LRS = [1e-3, 1e-4, 1e-5]
UNLEARN_ITERS = [500, 1000, 1500, 2000]

LAMBDA_VALS = sorted(set(
    [round(0.1 * i, 1) for i in range(1, 11)] +
    [0.5 * i for i in range(3, 21)]
))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"Using device: {DEVICE} "
    f"({'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
)


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


ratings_df, movies_df, users_df = load_data(DATA_DIR)

sample_users = ratings_df["user_id"].unique().tolist()
print(f"Total users: {len(sample_users)}")

pilot_ratings = ratings_df[ratings_df["user_id"].isin(sample_users)].copy()
pilot_ratings.sort_values(["user_id", "timestamp"], inplace=True)
print(f"Users in pilot data: {pilot_ratings['user_id'].nunique()}")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

pilot_users_df = users_df[users_df["user_id"].isin(sample_users)]
user_cats = pilot_users_df[["user_id", "gender", "age", "occupation"]].copy()
oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
user_cat_mat = oh.fit_transform(user_cats[["gender", "age", "occupation"]])
user_feat_df = pd.DataFrame(user_cat_mat, index=user_cats["user_id"])

all_genres = sorted({
    g
    for s in movies_df["genres"].astype(str)
    for g in s.split("|")
})
genre_to_idx = {g: i for i, g in enumerate(all_genres)}
num_genres = len(all_genres)


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
# Train / forget split — demography + genre-grounded probabilistic sampling
# ---------------------------------------------------------------------------

print ()

def get_multiplier_demo(gender, age, occupation):
    m = 1.0
    if gender == "F":           
        m *= 1.36  # 45% women vs 33% men → 45/33 ≈ 1.36x
    if age == 1:               
        m *= 1.0   # Under 18 → map to 75+ (24%)
    elif age == 18:            
        m *= 1.65  # 18-24 → 65%
    elif age == 25:            
        m *= 1.60  # 25-34 → 64%
    elif age == 35:            
        m *= 1.55  # 35-44 → 62%
    elif age == 45:            
        m *= 1.22  # 45-54 → 49%
    elif age == 50:            
        m *= 1.0   # 50-55 → map to 55-64 (39%)
    elif age == 56:            
        m *= 0.92  # 56+ → map to 65-74 (37%)
    if occupation in (1, 4, 6, 10, 11, 15):   
        m *= 1.22  # Bachelor's+ (73%) vs HS or less (60%) → 73/60 ≈ 1.22x
    return m

# Compute per-user multiplier (demographic only — sorted by uid for determinism)
_users_meta = users_df[users_df["user_id"].isin(sample_users)].set_index("user_id")
sample_users = np.array(sorted(sample_users))

_uid_mult = []
for uid in sample_users:
    if uid not in _users_meta.index:
        m = 1.0
    else:
        row = _users_meta.loc[uid]
        m = get_multiplier_demo(row["gender"], row["age"], row["occupation"])
    _uid_mult.append((uid, m))

# Auto-calibrate base_prob so E[forget users] = target % * N exactly
_mean_mult = float(np.mean([m for _, m in _uid_mult]))
base_prob = (FORGET_PERCENTAGE / 100) / _mean_mult
print(f"Mean combined multiplier : {_mean_mult:.4f}")
print(f"Auto-calibrated base_prob: {base_prob:.5f} (target {FORGET_PERCENTAGE}%)")

# Isolated RNG — decoupled from global RNG state
_rng = np.random.default_rng(make_seed("forget_split"))

forget_users_list = []
retain_users_list = []

for uid, mult in _uid_mult:
    p = min(base_prob * mult, 0.30)
    if _rng.random() < p:
        forget_users_list.append(uid)
    else:
        retain_users_list.append(uid)

forget_users = np.array(forget_users_list)
retain_users = np.array(retain_users_list)

_forget_meta = _users_meta[_users_meta.index.isin(forget_users_list)]
print(f"Forget users : {len(forget_users)} ({100 * len(forget_users) / len(sample_users):.1f}%)")
print(f"Retain users : {len(retain_users)}")
if len(_forget_meta) > 0:
    print(f"  Gender breakdown : {_forget_meta['gender'].value_counts().to_dict()}")
    print(f"  Age    breakdown : {_forget_meta['age'].value_counts().sort_index().to_dict()}")
    print(f"  Occ    breakdown : {_forget_meta['occupation'].value_counts().sort_index().to_dict()}")

# raise ValueError("Break Stop")

# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

pilot_ratings_all = ratings_df[ratings_df["user_id"].isin(sample_users)].copy()
pilot_ratings_all.sort_values(["user_id", "timestamp"], inplace=True)

trajectories_all = [
    {
        "user_id": uid,
        "movies": g["movie_id"].tolist(),
        "ratings": g["rating"].tolist(),
    }
    for uid, g in pilot_ratings_all.groupby("user_id")
    if len(g) >= 5
]

candidate_movies = np.array(sorted(pilot_ratings_all["movie_id"].unique()))
num_actions = len(candidate_movies)
print(f"Candidate movies (actions): {num_actions}")

forget_user_set = set(forget_users.tolist())
retain_user_set = set(retain_users.tolist())

forget_trajectories = [t for t in trajectories_all if t["user_id"] in forget_user_set]
retain_trajectories = [t for t in trajectories_all if t["user_id"] in retain_user_set]

print(f"Forget trajectories : {len(forget_trajectories)}")
print(f"Retain trajectories : {len(retain_trajectories)}")


# ---------------------------------------------------------------------------
# State function
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MovieLensEnv:
    def __init__(self, trajectories, build_state_fn, candidate_movies):
        self.trajs = trajectories
        self.build_state_fn = build_state_fn
        self.candidate_movies = np.array(candidate_movies)
        self.num_actions = len(candidate_movies)

    def reset(self):
        traj = self.trajs[np.random.randint(len(self.trajs))]
        self.user_id = traj["user_id"]
        self.movies = traj["movies"]
        self.ratings = traj["ratings"]
        self.t = 1
        self.future_dict = {
            m: r for m, r in zip(self.movies[self.t:], self.ratings[self.t:])
        }
        return self.build_state_fn(self.user_id, self.movies[: self.t])

    @staticmethod
    def _rating_to_reward(rating):
        if rating >= 5:
            return 1.0
        if rating >= 4:
            return rating / 5
        return 0.0

    def step(self, action_idx):
        rec_movie = self.candidate_movies[action_idx]
        reward = (
            self._rating_to_reward(self.future_dict[rec_movie])
            if rec_movie in self.future_dict
            else 0.0
        )
        self.t += 1
        done = self.t >= len(self.movies)
        if not done:
            self.future_dict = {
                m: r for m, r in zip(self.movies[self.t:], self.ratings[self.t:])
            }
            next_state = self.build_state_fn(self.user_id, self.movies[: self.t])
        else:
            next_state = None
        return next_state, reward, done


# ---------------------------------------------------------------------------
# Policy network
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


# ---------------------------------------------------------------------------
# Training (batched REINFORCE)
# ---------------------------------------------------------------------------

def train_policy_gradient_batched(
    env,
    policy_net,
    optimizer,
    num_episodes=10_000,
    gamma=0.99,
    max_steps_per_ep=30,
    batch_size=1,
    patience=PATIENCE,
    min_delta=0.01,
    log_interval=LOG_INTERVAL,
):
    returns_log = []
    best_avg_return = -float("inf")
    patience_counter = 0
    ep = 0

    while ep < num_episodes:
        batch_log_probs, batch_returns, batch_ep_rets = [], [], []

        for _ in range(batch_size):
            if ep >= num_episodes:
                break
            state = env.reset()
            lps, rews = [], []

            for _ in range(max_steps_per_ep):
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                probs = F.softmax(policy_net(st), dim=-1).squeeze(0)
                m = Categorical(probs)
                a = m.sample()
                next_state, rew, done = env.step(a.item())
                lps.append(m.log_prob(a))
                rews.append(rew)
                if done:
                    break
                state = next_state

            G, rets = 0.0, []
            for r in reversed(rews):
                G = r + gamma * G
                rets.insert(0, G)
            rets = torch.tensor(rets, dtype=torch.float32).to(DEVICE)
            if len(rets) > 1:
                rets = (rets - rets.mean()) / (rets.std() + 1e-10)

            batch_log_probs.append(torch.stack(lps))
            batch_returns.append(rets)
            batch_ep_rets.append(float(sum(rews)))
            ep += 1

        loss = sum(-(lp * r).sum() for lp, r in zip(batch_log_probs, batch_returns))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        returns_log.extend(batch_ep_rets)

        if ep % log_interval == 0 and ep > 0:
            avg_ret = np.mean(returns_log[-log_interval:])
            print(f"  ep {ep} | avg_ret={avg_ret:.3f} | best={best_avg_return:.3f}")
            if avg_ret > best_avg_return + min_delta:
                best_avg_return = avg_ret
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at ep {ep}")
                break

    return returns_log


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(policy_net, trajectories, build_state_fn, candidate_movies, K=10):
    hits, ndcgs = [], []
    candidate_movies = np.array(candidate_movies)

    for traj in trajectories:
        uid = traj["user_id"]
        movies = traj["movies"]
        if len(movies) < 5:
            continue

        split = len(movies) // 2
        future = set(movies[split:])
        state = build_state_fn(uid, movies[:split])
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            probs = F.softmax(policy_net(state_t), dim=-1).squeeze(0).cpu().numpy()

        topk_movies = candidate_movies[np.argsort(-probs, kind="stable")[:K]]

        hits.append(int(any(m in future for m in topk_movies)))

        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, m in enumerate(topk_movies)
            if m in future
        )
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(future), K)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return (
        float(np.mean(hits)) if hits else 0.0,
        float(np.mean(ndcgs)) if ndcgs else 0.0,
    )


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = map(lambda x: torch.tensor(np.array(x)), zip(*batch))
        return (
            s.float().to(device),
            a.long().to(device),
            r.float().to(device),
            s_next.float().to(device),
            done.float().to(device),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Buffer collection
# ---------------------------------------------------------------------------

def collect_random_experience_forget(env_forget, num_steps, buffer):
    """Collect forget-set experience using a random policy."""
    state = env_forget.reset()
    for _ in range(num_steps):
        action = np.random.randint(env_forget.num_actions)
        next_state, reward, done = env_forget.step(action)
        s_next = np.zeros_like(state) if next_state is None else next_state
        buffer.push(state, action, reward, s_next, float(done))
        state = env_forget.reset() if done else next_state


def collect_policy_experience(env, policy_net, num_steps, buffer):
    """Collect retain-set experience using the trained policy."""
    state = env.reset()
    for _ in range(num_steps):
        state_t = (
            torch.tensor(state, dtype=torch.float32)
            .unsqueeze(0)
            .to(next(policy_net.parameters()).device)
        )
        with torch.no_grad():
            probs = F.softmax(policy_net(state_t), dim=-1).squeeze(0)
        action = Categorical(probs).sample().item()
        next_state, reward, done = env.step(action)
        s_next = np.zeros_like(state) if next_state is None else next_state
        buffer.push(state, action, reward, s_next, float(done))
        state = env.reset() if done else next_state


# ---------------------------------------------------------------------------
# Unlearning 
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Unlearning loss log helpers
# ---------------------------------------------------------------------------

_LOSS_KEY_COLS = [
    "train_lr", "gamma", "hidden_dim", "train_batch",
    "unlearn_lr", "unlearn_iters", "lambda_retain",
    "method", "iter"
]

def load_unlearn_loss_log():
    if os.path.exists(UNLEARN_LOSS_LOG_PATH):
        df = pd.read_csv(UNLEARN_LOSS_LOG_PATH)
        before = len(df)
        df = df.drop_duplicates(subset=_LOSS_KEY_COLS, keep="last")
        print(f"✓ Unlearn loss log: {len(df):,} rows ({before - len(df)} duplicates removed)")
        return df.to_dict("records")
    print(" No unlearn loss log — starting fresh")
    return []

def save_unlearn_loss_log(loss_log_rows):
    if loss_log_rows:
        pd.DataFrame(loss_log_rows).drop_duplicates(
            subset=_LOSS_KEY_COLS, keep="last"
        ).to_csv(UNLEARN_LOSS_LOG_PATH, index=False)

def append_unlearn_loss_row(
    loss_log_rows,
    *,
    train_lr,
    gamma,
    hidden_dim,
    train_batch,
    unlearn_lr,
    unlearn_iters,
    lambda_retain,
    method,
    iter_idx,
    loss_forget,
    loss_retain,
    loss_total,
):
    loss_log_rows.append({
        "train_lr": train_lr,
        "gamma": gamma,
        "hidden_dim": hidden_dim,
        "train_batch": train_batch,
        "unlearn_lr": unlearn_lr,
        "unlearn_iters": unlearn_iters,
        "lambda_retain": lambda_retain,
        "method": method,
        "iter": iter_idx,
        "loss_forget": loss_forget,
        "loss_retain": loss_retain,
        "loss_total": loss_total,
    })
    save_unlearn_loss_log(loss_log_rows)

def append_eval_rows(
    all_results,
    *,
    net_after,
    baseline,
    train_lr,
    gamma,
    hidden_dim,
    train_batch,
    train_time_s,
    trained_model_path,
    unlearn_lr,
    unlearn_iters,
    lambda_retain,
    method,
    unlearn_time_s,
    unlearned_model_path,
    loss_forget_final,
    loss_retain_final,
    loss_total_final,
):
    # FIX 1: Pass trajectories_all to match the new eval_all_ks signature
    after = eval_all_ks(net_after, retain_trajectories, forget_trajectories, trajectories_all)
    
    for K in KS:
        # FIX 2: Unpack all 6 returned values (including the new combined metrics)
        h_r, n_r, h_f, n_f, h_c, n_c = after[K]
        
        # FIX 3: Safely slice the first 4 elements of baseline so it never crashes, 
        # regardless of how many items the baseline dictionary holds.
        base_metrics = baseline.get(K, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        bh_r, bn_r, bh_f, bn_f = base_metrics[:4] 
        
        all_results.append({
            "train_lr": train_lr,
            "gamma": gamma,
            "hidden_dim": hidden_dim,
            "train_batch": train_batch,
            "train_time_s": train_time_s,
            "trained_model_path": trained_model_path,
            "unlearn_lr": unlearn_lr,
            "unlearn_iters": unlearn_iters,
            "lambda_retain": lambda_retain,
            "method": method,
            "unlearn_time_s": unlearn_time_s,
            "unlearned_model_path": unlearned_model_path,
            "loss_forget_final": loss_forget_final,
            "loss_retain_final": loss_retain_final,
            "loss_total_final": loss_total_final,
            "K": K,
            "retain_Hit": h_r,
            "retain_NDCG": n_r,
            "forget_Hit": h_f,
            "forget_NDCG": n_f,
            "combined_Hit": h_c,    # FIX 4: Add new combined Hit
            "combined_NDCG": n_c,   # FIX 4: Add new combined NDCG
            "fq_hit": (bh_f - h_f) - 2.0 * (bh_r - h_r),
            "fq_ndcg": (bn_f - n_f) - 2.0 * (bn_r - n_r),
            "base_retain_Hit": bh_r,
            "base_retain_NDCG": bn_r,
            "base_forget_Hit": bh_f,
            "base_forget_NDCG": bn_f,
        })

    pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)

# ---------------------------------------------------------------------------
# Method 1 — Ye Appendix I (lambda fixed at 1.0)
# ---------------------------------------------------------------------------

def unlearning_finetune_ye_apxi(
    policy_net,
    forget_buffer,
    retain_buffer,
    candidate_movies,
    num_iters=2000,
    batch_size=64,
    lambda_retain=1.0,
    lr=1e-4,
    log_every=UNLEARN_LOG_INTERVAL,
    loss_log_rows=None,
    loss_log_meta=None,
):
    device = next(policy_net.parameters()).device
    old_net = copy.deepcopy(policy_net).eval().to(device)
    for p in old_net.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    last_lf = last_lr = last_lt = 0.0

    for it in range(num_iters):
        if len(forget_buffer) < batch_size or len(retain_buffer) < batch_size:
            break

        s_f, a_f, _, _, _ = forget_buffer.sample(batch_size, device)
        s_r, a_r, _, _, _ = retain_buffer.sample(batch_size, device)

        q_f_new = policy_net(s_f).gather(1, a_f.unsqueeze(1)).squeeze(1)
        loss_forget = q_f_new.mean()

        q_r_new = policy_net(s_r).gather(1, a_r.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_r_old = old_net(s_r).gather(1, a_r.unsqueeze(1)).squeeze(1)
        loss_retain = (q_r_new - q_r_old).abs().mean()

        loss = loss_forget + lambda_retain * loss_retain

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_lf, last_lr, last_lt = (
            loss_forget.item(),
            loss_retain.item(),
            loss.item(),
        )

        if (it + 1) % log_every == 0:
            print(
                f" [Ye_ApxI] iter {it+1:4d}/{num_iters} "
                f"| L_forget={last_lf:.4f} | L_retain={last_lr:.4f} | L_total={last_lt:.4f}"
            )
            if loss_log_rows is not None and loss_log_meta is not None:
                append_unlearn_loss_row(
                    loss_log_rows,
                    **loss_log_meta,
                    iter_idx=it + 1,
                    loss_forget=last_lf,
                    loss_retain=last_lr,
                    loss_total=last_lt,
                )

    return last_lf, last_lr, last_lt


# ---------------------------------------------------------------------------
# Method 2 — Ye Multi-Environment (true L_inf, lambda fixed at 1.0)
# ---------------------------------------------------------------------------

def unlearning_finetune_ye_multi(
    policy_net,
    forget_buffer,
    retain_buffer,
    candidate_movies,
    num_iters=2000,
    batch_size=64,
    lambda_retain=1.0,
    lr=1e-4,
    log_every=UNLEARN_LOG_INTERVAL,
    loss_log_rows=None,
    loss_log_meta=None,
):
    device = next(policy_net.parameters()).device
    old_net = copy.deepcopy(policy_net).eval().to(device)
    for p in old_net.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    last_lf = last_lr = last_lt = 0.0

    for it in range(num_iters):
        if len(forget_buffer) < batch_size or len(retain_buffer) < batch_size:
            break

        s_f, _, _, _, _ = forget_buffer.sample(batch_size, device)
        s_r, _, _, _, _ = retain_buffer.sample(batch_size, device)

        loss_forget = policy_net(s_f).abs().max(dim=1).values.mean()

        with torch.no_grad():
            q_r_old = old_net(s_r)
        q_r_diff = (policy_net(s_r) - q_r_old).abs().max(dim=1).values
        loss_retain = q_r_diff.mean()

        loss = loss_forget + lambda_retain * loss_retain

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_lf, last_lr, last_lt = (
            loss_forget.item(),
            loss_retain.item(),
            loss.item(),
        )

        if (it + 1) % log_every == 0:
            print(
                f" [Ye_multi] iter {it+1:4d}/{num_iters} "
                f"| L_forget={last_lf:.4f} | L_retain={last_lr:.4f} | L_total={last_lt:.4f}"
            )
            if loss_log_rows is not None and loss_log_meta is not None:
                append_unlearn_loss_row(
                    loss_log_rows,
                    **loss_log_meta,
                    iter_idx=it + 1,
                    loss_forget=last_lf,
                    loss_retain=last_lr,
                    loss_total=last_lt,
                )

    return last_lf, last_lr, last_lt


# ---------------------------------------------------------------------------
# Method 3 — New_True_inf (true L_inf, both terms squared, lambda swept)
# ---------------------------------------------------------------------------

def unlearning_finetune_new_true_inf(
    policy_net,
    forget_buffer,
    retain_buffer,
    candidate_movies,
    num_iters=2000,
    batch_size=64,
    lambda_retain=1.0,
    lr=1e-4,
    log_every=UNLEARN_LOG_INTERVAL,
    loss_log_rows=None,
    loss_log_meta=None,
):
    device = next(policy_net.parameters()).device
    old_net = copy.deepcopy(policy_net).eval().to(device)
    for p in old_net.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    last_lf = last_lr = last_lt = 0.0

    for it in range(num_iters):
        if len(forget_buffer) < batch_size or len(retain_buffer) < batch_size:
            break

        s_f, _, _, _, _ = forget_buffer.sample(batch_size, device)
        s_r, _, _, _, _ = retain_buffer.sample(batch_size, device)

        loss_forget = policy_net(s_f).abs().max(dim=1).values.pow(2).mean()

        with torch.no_grad():
            q_r_old = old_net(s_r)
        q_r_diff = (policy_net(s_r) - q_r_old).abs().max(dim=1).values
        loss_retain = q_r_diff.pow(2).mean()

        loss = loss_forget + lambda_retain * loss_retain

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_lf, last_lr, last_lt = (
            loss_forget.item(),
            loss_retain.item(),
            loss.item(),
        )

        if (it + 1) % log_every == 0:
            print(
                f" [New_True_inf] iter {it+1:4d}/{num_iters} "
                f"| L_forget={last_lf:.4f} | L_retain={last_lr:.4f} | L_total={last_lt:.4f}"
            )
            if loss_log_rows is not None and loss_log_meta is not None:
                append_unlearn_loss_row(
                    loss_log_rows,
                    **loss_log_meta,
                    iter_idx=it + 1,
                    loss_forget=last_lf,
                    loss_retain=last_lr,
                    loss_total=last_lt,
                )

    return last_lf, last_lr, last_lt


# ---------------------------------------------------------------------------
# Method 4 — New_Max (your current finetune_new, renamed)
# ---------------------------------------------------------------------------

def unlearning_finetune_new_max(
    policy_net,
    forget_buffer,
    retain_buffer,
    candidate_movies,
    num_iters=2000,
    batch_size=64,
    lambda_retain=1.0,
    lr=1e-4,
    log_every=UNLEARN_LOG_INTERVAL,
    loss_log_rows=None,
    loss_log_meta=None,
):
    device = next(policy_net.parameters()).device
    old_net = copy.deepcopy(policy_net).eval().to(device)
    for p in old_net.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    last_lf = last_lr = last_lt = 0.0

    for it in range(num_iters):
        if len(forget_buffer) < batch_size or len(retain_buffer) < batch_size:
            break

        s_f, _, _, _, _ = forget_buffer.sample(batch_size, device)
        s_r, _, _, _, _ = retain_buffer.sample(batch_size, device)

        loss_forget = policy_net(s_f).max(dim=1).values.pow(2).mean()

        with torch.no_grad():
            q_r_old = old_net(s_r)
        q_r_diff = (policy_net(s_r) - q_r_old).abs().max(dim=1).values
        loss_retain = q_r_diff.pow(2).mean()

        loss = loss_forget + lambda_retain * loss_retain

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_lf, last_lr, last_lt = (
            loss_forget.item(),
            loss_retain.item(),
            loss.item(),
        )

        if (it + 1) % log_every == 0:
            print(
                f" [New_Max] iter {it+1:4d}/{num_iters} "
                f"| L_forget={last_lf:.4f} | L_retain={last_lr:.4f} | L_total={last_lt:.4f}"
            )
            if loss_log_rows is not None and loss_log_meta is not None:
                append_unlearn_loss_row(
                    loss_log_rows,
                    **loss_log_meta,
                    iter_idx=it + 1,
                    loss_forget=last_lf,
                    loss_retain=last_lr,
                    loss_total=last_lt,
                )

    return last_lf, last_lr, last_lt


# ---------------------------------------------------------------------------
# Method 5 - Gradient Ascent
# ---------------------------------------------------------------------------

def unlearning_gradient_ascent(
    env,
    policy_net,
    num_iters=2000,
    batch_size=64,
    lr=1e-4,
    gamma=0.99,
    max_steps_per_ep=30,
    log_every=UNLEARN_LOG_INTERVAL,
    loss_log_rows=None,
    loss_log_meta=None,
):
    device = next(policy_net.parameters()).device
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    last_lf = last_lr = last_lt = 0.0

    ep = 0
    while ep < num_iters:
        batch_log_probs, batch_returns = [], []

        for _ in range(batch_size):
            if ep >= num_iters:
                break
            state = env.reset()
            lps, rews = [], []

            for _ in range(max_steps_per_ep):
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                probs = F.softmax(policy_net(st), dim=-1).squeeze(0)
                m = Categorical(probs)
                a = m.sample()
                next_state, rew, done = env.step(a.item())
                lps.append(m.log_prob(a))
                rews.append(rew)
                if done:
                    break
                state = next_state

            G, rets = 0.0, []
            for r in reversed(rews):
                G = r + gamma * G
                rets.insert(0, G)
            rets = torch.tensor(rets, dtype=torch.float32).to(device)
            if len(rets) > 1:
                rets = (rets - rets.mean()) / (rets.std() + 1e-10)

            batch_log_probs.append(torch.stack(lps))
            batch_returns.append(rets)
            ep += 1

        if not batch_log_probs:
            break

        loss_forget = sum((lp * r).sum() for lp, r in zip(batch_log_probs, batch_returns))
        
        optimizer.zero_grad()
        loss_forget.backward()
        optimizer.step()

        last_lf = loss_forget.item()
        last_lr = 0.0
        last_lt = last_lf

        if ep % log_every == 0 or ep >= num_iters:
            print(
                f" [Grad_Ascent] iter {ep:4d}/{num_iters} "
                f"| L_forget={last_lf:.4f} | L_retain={last_lr:.4f} | L_total={last_lt:.4f}"
            )
            if loss_log_rows is not None and loss_log_meta is not None:
                append_unlearn_loss_row(
                    loss_log_rows,
                    **loss_log_meta,
                    iter_idx=ep,
                    loss_forget=last_lf,
                    loss_retain=last_lr,
                    loss_total=last_lt,
                )

    return last_lf, last_lr, last_lt

# ---------------------------------------------------------------------------
# Filename helpers
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
    return os.path.join(MODELS_DIR, name)


def unlearned_model_path(t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam):
    name = (
        f"unlearn__{method}__tlr{_fmt(t_lr)}__g{_fmt(gamma)}"
        f"__h{hidden_dim}__bs{train_bs}__ulr{_fmt(u_lr)}"
        f"__ui{u_iters}__lam{_fmt(lam)}.pt"
    )
    return os.path.join(MODELS_DIR, name)


def retain_buf_path(t_lr, gamma, hidden_dim, train_bs):
    return os.path.join(
        RESULTS_BASE,
        f"retain_buf__tlr{_fmt(t_lr)}__g{_fmt(gamma)}"
        f"__h{hidden_dim}__bs{train_bs}.pkl",
    )


# ---------------------------------------------------------------------------
# Progress and results trackers
# ---------------------------------------------------------------------------

_TRAIN_PROG_COLS = ["t_lr", "gamma", "hidden_dim", "train_bs"]
_TRAIN_KEY_COLS = ["train_lr", "gamma", "hidden_dim", "train_batch", "K"]
_PROG_COLS = ["t_lr", "gamma", "hidden_dim", "train_bs", "u_lr", "u_iters", "lam", "method"]
_KEY_COLS = [
    "train_lr", "gamma", "hidden_dim", "train_batch",
    "unlearn_lr", "unlearn_iters", "lambda_retain", "method", "K",
]


def load_train_progress(train_configs):
    if os.path.exists(TRAIN_PROGRESS_PATH):
        df = pd.read_csv(TRAIN_PROGRESS_PATH)
        done = set(tuple(r) for r in df[_TRAIN_PROG_COLS].itertuples(index=False))
        print(f"✓ Train progress: {len(done)}/{len(train_configs)} done")
        return df, done
    print("  No train progress — starting fresh")
    return pd.DataFrame(columns=_TRAIN_PROG_COLS), set()


def mark_train_done(prog_df, done_set, t_lr, gamma, hidden_dim, train_bs):
    key = (t_lr, gamma, hidden_dim, train_bs)
    if key not in done_set:
        done_set.add(key)
        row = pd.DataFrame([dict(zip(_TRAIN_PROG_COLS, key))])
        prog_df = row if prog_df.empty else pd.concat([prog_df, row], ignore_index=True)
        prog_df.to_csv(TRAIN_PROGRESS_PATH, index=False)
    return prog_df


def load_legacy_unlearn_progress():
    import pandas as pd
    import glob
    done = set()
    
    # Load from merged file AND all existing worker files dynamically just in case user forgot to merge
    all_files = glob.glob(os.path.join(RESULTS_BASE, "tuning_full_results*.csv"))
    
    for f in all_files:
        df = pd.read_csv(f)
        if not df.empty:
            legacy_cols = ["train_lr", "gamma", "hidden_dim", "train_batch", "unlearn_lr", "unlearn_iters", "lambda_retain", "method"]
            if all(col in df.columns for col in legacy_cols):
                df = df.drop_duplicates(subset=legacy_cols)
                for r in df[legacy_cols].itertuples(index=False):
                    done.add((
                        float(r.train_lr), float(r.gamma), int(r.hidden_dim), int(r.train_batch),
                        float(r.unlearn_lr), int(float(r.unlearn_iters)), float(r.lambda_retain), str(r.method)
                    ))
    print(f"✓ Legacy unlearn progress universally loaded: {len(done):,} combos done across {len(all_files)} files")
    return done

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        df = pd.read_csv(PROGRESS_PATH)
        done = set(tuple(r) for r in df[_PROG_COLS].itertuples(index=False))
        print(f"✓ Unlearn progress: {len(done):,} combos done")
        return df, done
    print("  No unlearn progress — starting fresh")
    return pd.DataFrame(columns=_PROG_COLS), set()


def mark_done(
    prog_df, done_set,
    t_lr, gamma, hidden_dim, train_bs,
    u_lr, u_iters, lam, method,
):
    key = (t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam, method)
    if key not in done_set:
        done_set.add(key)
        row = pd.DataFrame([dict(zip(_PROG_COLS, key))])
        prog_df = row if prog_df.empty else pd.concat([prog_df, row], ignore_index=True)
        prog_df.to_csv(PROGRESS_PATH, index=False)
    return prog_df


def load_train_results():
    if os.path.exists(TRAIN_RESULTS_PATH):
        df = (
            pd.read_csv(TRAIN_RESULTS_PATH)
            .drop_duplicates(subset=_TRAIN_KEY_COLS, keep="last")
        )
        print(f"✓ Train results loaded from {os.path.basename(TRAIN_RESULTS_PATH)}: {len(df):,} rows")
        return df.to_dict("records")
    print("  No train results — starting fresh")
    return []


def load_results():
    if os.path.exists(RESULTS_PATH):
        df = pd.read_csv(RESULTS_PATH)
        before = len(df)
        df = df.drop_duplicates(subset=_KEY_COLS, keep="last")
        print(f"✓ Unlearn results loaded from {os.path.basename(RESULTS_PATH)}: {len(df):,} rows ({before - len(df)} duplicates removed)")
        return df.to_dict("records")
    print("  No unlearn results — starting fresh")
    return []

def eval_all_ks(net, ret_trajs, for_trajs, all_trajs):
    out = {}
    for K in KS:
        h_r, n_r = evaluate_policy(net, ret_trajs, build_state_fn, candidate_movies, K=K)
        h_f, n_f = evaluate_policy(net, for_trajs, build_state_fn, candidate_movies, K=K)
        h_c, n_c = evaluate_policy(net, all_trajs, build_state_fn, candidate_movies, K=K)
        out[K] = (h_r, n_r, h_f, n_f, h_c, n_c)
    return out


# ---------------------------------------------------------------------------
# Sweep configuration summary
# ---------------------------------------------------------------------------

train_configs = list(itertools.product(TRAIN_LRS, GAMMAS, HIDDEN_DIMS, TRAIN_BATCH_SIZES))
unlearn_configs = list(itertools.product(UNLEARN_LRS, UNLEARN_ITERS))

print(f"Train configs    : {len(train_configs)}")
print(f"Unlearn configs  : {len(unlearn_configs)}")
print(f"Lambda values    : {len(LAMBDA_VALS)}")
print(
    f"Patience         : {PATIENCE} intervals x {LOG_INTERVAL} eps "
    f"= {PATIENCE * LOG_INTERVAL} episodes without improvement"
)


# ---------------------------------------------------------------------------
# Global forget buffer (random policy, built once per forget_percentage)
# ---------------------------------------------------------------------------

set_seed(make_seed("forget_buffer"))  # ← deterministic regardless of what ran before
if os.path.exists(F_BUF_PATH):
    with open(F_BUF_PATH, "rb") as fh:
        f_buf_global = pickle.load(fh)
    print(f"✓ Forget buffer loaded — {len(f_buf_global):,} steps")
else:
    f_buf_global = ReplayBuffer(capacity=100_000)
    collect_random_experience_forget(
        MovieLensEnv(forget_trajectories, build_state_fn, candidate_movies),
        num_steps=100_000,
        buffer=f_buf_global,
    )
    with open(F_BUF_PATH, "wb") as fh:
        pickle.dump(f_buf_global, fh)
    print(f"✓ Forget buffer built and saved — {len(f_buf_global):,} steps")



# ===========================================================================
# PHASE 1 — Train all configs, record baseline metrics
# ===========================================================================

if RUN_PHASE == 2:
    print("Skipping Phase 1 (--phase 2 specified)")
else:
    print(f"\n{'#' * 72}")
    print(f" PHASE 1 — Worker {WORKER_ID}/{NUM_WORKERS} | Training assigned configs")
    print(f"{'#' * 72}")

train_prog_df, train_done_set = load_train_progress(train_configs)
train_results = load_train_results()

# --- BACKFILL CHECK: Test combined dataset if column is missing ---
needs_save = False
for r in train_results:
    if "base_combined_Hit" not in r or pd.isna(r.get("base_combined_Hit")):
        print(f"  [Backfill] Testing missing combined metric for tlr={r['train_lr']} g={r['gamma']} h={r['hidden_dim']} bs={r['train_batch']} K={r['K']}")
        t_model_path = trained_model_path(r["train_lr"], r["gamma"], r["hidden_dim"], r["train_batch"])
        if os.path.exists(t_model_path):
            set_seed(make_seed(r["train_lr"], r["gamma"], r["hidden_dim"], r["train_batch"], "eval_backfill"))
            net_bf = PolicyNet(state_dim, num_actions, hidden_dim=int(r["hidden_dim"])).to(DEVICE)
            net_bf.load_state_dict(torch.load(t_model_path, map_location=DEVICE))
            net_bf.eval()
            h_c, n_c = evaluate_policy(net_bf, trajectories_all, build_state_fn, candidate_movies, K=r["K"])
            r["base_combined_Hit"] = h_c
            r["base_combined_NDCG"] = n_c
            needs_save = True
        else:
            print(f"  [Backfill] Warning: Model file missing for backfill: {t_model_path}")

if needs_save:
    pd.DataFrame(train_results).to_csv(TRAIN_RESULTS_PATH, index=False)
    print("  ✓ Saved backfilled combined metrics.")
# ------------------------------------------------------------------

for cfg_idx, (t_lr, gamma, hidden_dim, train_bs) in enumerate(train_configs):
    if cfg_idx % NUM_WORKERS != WORKER_ID:
        continue

    set_seed(make_seed(t_lr, gamma, hidden_dim, train_bs, "phase1"))
    cfg_key = (t_lr, gamma, hidden_dim, train_bs)

    if cfg_key in train_done_set:
        print(
            f" [SKIP {cfg_idx + 1:>3}/{len(train_configs)}] "
            f"tlr={t_lr} g={gamma} h={hidden_dim} bs={train_bs} — already done"
        )
        continue

    print(
        f"\n [{cfg_idx + 1:>3}/{len(train_configs)}] "
        f"train_lr={t_lr} gamma={gamma} hidden={hidden_dim} batch={train_bs}"
    )

    t_model_path = trained_model_path(t_lr, gamma, hidden_dim, train_bs)

    if os.path.exists(t_model_path):
        print("  ↩ Loading existing model...")
        net = PolicyNet(state_dim, num_actions, hidden_dim=hidden_dim).to(DEVICE)
        net.load_state_dict(torch.load(t_model_path, map_location=DEVICE))
        train_time_s = float("nan")
    else:
        print("  Training from scratch...")
        t0 = time.time()
        env_tr = MovieLensEnv(trajectories_all, build_state_fn, candidate_movies)
        net = PolicyNet(state_dim, num_actions, hidden_dim=hidden_dim).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=t_lr)
        train_policy_gradient_batched(
            env_tr, net, opt,
            num_episodes=NUM_EPISODES,
            gamma=gamma,
            max_steps_per_ep=MAX_STEPS,
            batch_size=train_bs,
        )
        train_time_s = round(time.time() - t0, 2)
        torch.save(net.state_dict(), t_model_path)
        print(f"  ✓ Trained in {train_time_s:.1f}s")

    # Force seed before evaluation for maximum reproducibility
    set_seed(make_seed(t_lr, gamma, hidden_dim, train_bs, "eval"))
    net.eval()
    baseline = eval_all_ks(net, retain_trajectories, forget_trajectories, trajectories_all)

    for K in KS:
        h_r, n_r, h_f, n_f, h_c, n_c = baseline[K]
        train_results.append({
            "train_lr": t_lr,
            "gamma": gamma,
            "hidden_dim": hidden_dim,
            "train_batch": train_bs,
            "train_time_s": train_time_s,
            "trained_model_path": t_model_path,
            "K": K,
            "base_retain_Hit": h_r,
            "base_retain_NDCG": n_r,
            "base_forget_Hit": h_f,
            "base_forget_NDCG": n_f,
            "base_combined_Hit": h_c,
            "base_combined_NDCG": n_c,
        })
    
    pd.DataFrame(train_results).drop_duplicates(
        subset=_TRAIN_KEY_COLS, keep="last"
    ).to_csv(TRAIN_RESULTS_PATH, index=False)

    train_prog_df = mark_train_done(
        train_prog_df, train_done_set, t_lr, gamma, hidden_dim, train_bs
    )

print(f"\n✓ Phase 1 complete — {len(train_done_set)} models trained / loaded")

# ===========================================================================
# PHASE 1 → PHASE 2 — Select top configs by retain metric
# ===========================================================================

# Phase 2 must rank ALL configs, not just this worker's — read the merged file
if NUM_WORKERS > 1:
    if not os.path.exists(TRAIN_RESULTS_MERGED):
        raise FileNotFoundError(
            f"Merged Phase 1 results not found at {TRAIN_RESULTS_MERGED}.\n"
            f"Run the merge script before launching Phase 2 workers."
        )
    print(f"✓ Merged Phase 1 results loaded from {os.path.basename(TRAIN_RESULTS_MERGED)}")
    all_train_df = pd.read_csv(TRAIN_RESULTS_MERGED).drop_duplicates(subset=_TRAIN_KEY_COLS, keep="last")
else:
    print(f"✓ Using un-merged local Phase 1 results")
    all_train_df = pd.DataFrame(train_results).drop_duplicates(subset=_TRAIN_KEY_COLS, keep="last")

rank_df = (
    all_train_df[all_train_df["K"] == TOP_SELECTION_K]
    .sort_values(
        [TOP_SELECTION_METRIC, "train_lr", "gamma", "hidden_dim", "train_batch"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    .reset_index(drop=True)
)
n_top = max(1, int(len(rank_df) * TOP_PERCENT))
top_configs_df = rank_df.head(n_top)
all_top_configs = [
    (row["train_lr"], row["gamma"], int(row["hidden_dim"]), int(row["train_batch"]))
    for _, row in top_configs_df.iterrows()
]

# Partition top configs across workers (round-robin keeps load balanced)
top_configs = [cfg for i, cfg in enumerate(all_top_configs) if i % NUM_WORKERS == WORKER_ID]
print(f"Worker {WORKER_ID} handling {len(top_configs)}/{len(all_top_configs)} top configs")


print(f"\n{'#' * 72}")
print(
    f" PHASE 2 — Unlearning on top {n_top}/{len(rank_df)} configs "
    f"({TOP_PERCENT * 100:.0f}% by {TOP_SELECTION_METRIC}@{TOP_SELECTION_K})"
)
print(f"{'#' * 72}")
for i, (t_lr, gamma, hidden_dim, train_bs) in enumerate(top_configs):
    score = top_configs_df.iloc[i][TOP_SELECTION_METRIC]
    print(
        f"  {i + 1}. tlr={t_lr} g={gamma} h={hidden_dim} "
        f"bs={train_bs} | score={score:.4f}"
    )


# ===========================================================================
# PHASE 2 — Unlearning sweep
# ===========================================================================

progress_df, done_set = load_progress()
legacy_done_set = load_legacy_unlearn_progress()
done_set.update(legacy_done_set)
all_results = load_results()
unlearn_loss_log_rows = load_unlearn_loss_log()

train_results = all_train_df.to_dict("records")

# Create a locks directory to prevent workers from picking the same config
LOCKS_DIR = os.path.join(RESULTS_BASE, "locks")
os.makedirs(LOCKS_DIR, exist_ok=True)

for cfg_idx, (t_lr, gamma, hidden_dim, train_bs) in enumerate(top_configs):

    if cfg_idx % NUM_WORKERS != WORKER_ID:
        continue

    # --- ATOMIC LOCK (Dynamic Queue mechanism) ---
    lock_file = os.path.join(LOCKS_DIR, f"lock_{t_lr}_{gamma}_{hidden_dim}_{train_bs}.txt")
    try:
        # os.O_EXCL ensures this fails if the file already exists (claimed by other worker)
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print(f"\n[{cfg_idx + 1}/{len(top_configs)}] tlr={t_lr} g={gamma} h={hidden_dim} bs={train_bs} — claimed by another worker. Skipping.")
        continue
    # ---------------------------------------------

    fixed_methods = ["Ye_ApxI", "Ye_multi"]
    sweep_methods = ["New_True_inf", "New_Max"]

    all_done_fixed = all(
        (t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, 1.0, method) in done_set
        for u_lr, u_iters in unlearn_configs
        for method in fixed_methods
    )
    all_done_swept = all(
        (t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam, method) in done_set
        for u_lr, u_iters in unlearn_configs
        for lam in LAMBDA_VALS
        for method in sweep_methods
    )
    all_done_ga = all(
        (t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, 0.0, "Gradient_Ascent") in done_set
        for u_lr, u_iters in unlearn_configs
    )

    if all_done_fixed and all_done_swept and all_done_ga:
        print(
            f"\n[{cfg_idx + 1}/{len(top_configs)}] "
            f"tlr={t_lr} g={gamma} h={hidden_dim} bs={train_bs} — all combos done"
        )
        continue

    print(f"\n{'=' * 72}")
    print(
        f"[{cfg_idx + 1}/{len(top_configs)}] "
        f"train_lr={t_lr} gamma={gamma} hidden={hidden_dim} train_batch={train_bs}"
    )
    print(f"{'=' * 72}")

    # FIX: Use safe float/int matching to avoid pandas precision issues
    prev = [
        r for r in train_results
        if (
            abs(float(r["train_lr"]) - float(t_lr)) < 1e-6
            and abs(float(r["gamma"]) - float(gamma)) < 1e-6
            and int(r["hidden_dim"]) == int(hidden_dim)
            and int(r["train_batch"]) == int(train_bs)
        )
    ]
    train_time_s = prev[0]["train_time_s"] if prev else float("nan")
    t_model_path = trained_model_path(t_lr, gamma, hidden_dim, train_bs)

    net = PolicyNet(state_dim, num_actions, hidden_dim=hidden_dim).to(DEVICE)
    net.load_state_dict(torch.load(t_model_path, map_location=DEVICE))
    print(f" ↩ Loaded {t_model_path}")

    baseline = {
        int(float(r["K"])): (
            r["base_retain_Hit"],
            r["base_retain_NDCG"],
            r["base_forget_Hit"],
            r["base_forget_NDCG"],
        )
        for r in train_results
        if (
            abs(float(r["train_lr"]) - float(t_lr)) < 1e-6
            and abs(float(r["gamma"]) - float(gamma)) < 1e-6
            and int(r["hidden_dim"]) == int(hidden_dim)
            and int(r["train_batch"]) == int(train_bs)
        )
    }

    set_seed(make_seed(t_lr, gamma, hidden_dim, train_bs, "retain_buf"))
    r_buf_path = retain_buf_path(t_lr, gamma, hidden_dim, train_bs)
    if os.path.exists(r_buf_path):
        with open(r_buf_path, "rb") as fh:
            r_buf = pickle.load(fh)
        print(f" ↩ Retain buffer loaded — {len(r_buf):,} steps")
    else:
        r_buf = ReplayBuffer(capacity=150_000)
        collect_policy_experience(
            MovieLensEnv(retain_trajectories, build_state_fn, candidate_movies),
            net,
            num_steps=100_000,
            buffer=r_buf,
        )
        with open(r_buf_path, "wb") as fh:
            pickle.dump(r_buf, fh)
        print(f" ↩ Retain buffer built — {len(r_buf):,} steps")

    f_buf = f_buf_global
    total_ul = len(unlearn_configs) * (2 + 2 * len(LAMBDA_VALS) + 1)
    ul_done = ul_skipped = 0

    for u_lr, u_iters in unlearn_configs:

        # ==============================================================
        # Method 1 — Ye_ApxI (lambda fixed at 1.0)
        # ==============================================================
        lam = 1.0
        method = "Ye_ApxI"
        combo_key = (t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam, method)
        set_seed(make_seed(t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam, method)) 

        if combo_key in done_set:
            ul_skipped += 1
            ul_done += 1
            print(
                f" [SKIP {ul_done:>4}/{total_ul}] "
                f"{method} u_lr={u_lr} u_iters={u_iters} lam={lam}"
            )
        else:
            print(
                f" [RUN {ul_done + 1:>4}/{total_ul}] "
                f"{method} u_lr={u_lr} u_iters={u_iters} lam={lam} ...",
                end="",
                flush=True,
            )
            net_copy = copy.deepcopy(net)
            t_ul0 = time.time()
            lf, lr_, lt = unlearning_finetune_ye_apxi(
                net_copy,
                forget_buffer=f_buf,
                retain_buffer=r_buf,
                candidate_movies=candidate_movies,
                num_iters=u_iters,
                batch_size=UNLEARN_BATCH,
                lambda_retain=lam,
                lr=u_lr,
                log_every=UNLEARN_LOG_INTERVAL,
                loss_log_rows=unlearn_loss_log_rows,
                loss_log_meta={
                    "train_lr": t_lr,
                    "gamma": gamma,
                    "hidden_dim": hidden_dim,
                    "train_batch": train_bs,
                    "unlearn_lr": u_lr,
                    "unlearn_iters": u_iters,
                    "lambda_retain": lam,
                    "method": method,
                },
            )
            unlearn_time_s = round(time.time() - t_ul0, 2)
            print(f" done in {unlearn_time_s:.1f}s (Lf={lf:.4f}, Lr={lr_:.4f})")

            ul_path = unlearned_model_path(
                t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam
            )
            if SAVE_UNLEARNED_MODELS:
                torch.save(net_copy.state_dict(), ul_path)

            append_eval_rows(
                all_results,
                net_after=net_copy,
                baseline=baseline,
                train_lr=t_lr,
                gamma=gamma,
                hidden_dim=hidden_dim,
                train_batch=train_bs,
                train_time_s=train_time_s,
                trained_model_path=t_model_path,
                unlearn_lr=u_lr,
                unlearn_iters=u_iters,
                lambda_retain=lam,
                method=method,
                unlearn_time_s=unlearn_time_s,
                unlearned_model_path=ul_path if SAVE_UNLEARNED_MODELS else "",
                loss_forget_final=lf,
                loss_retain_final=lr_,
                loss_total_final=lt,
            )

            progress_df = mark_done(
                progress_df, done_set,
                t_lr, gamma, hidden_dim, train_bs,
                u_lr, u_iters, lam, method,
            )
            ul_done += 1

        # ==============================================================
        # Method 2 — Ye_multi (lambda fixed at 1.0)
        # ==============================================================
        lam = 1.0
        method = "Ye_multi"
        combo_key = (t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam, method)
        set_seed(make_seed(t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam, method)) 

        if combo_key in done_set:
            ul_skipped += 1
            ul_done += 1
            print(
                f" [SKIP {ul_done:>4}/{total_ul}] "
                f"{method} u_lr={u_lr} u_iters={u_iters} lam={lam}"
            )
        else:
            print(
                f" [RUN {ul_done + 1:>4}/{total_ul}] "
                f"{method} u_lr={u_lr} u_iters={u_iters} lam={lam} ...",
                end="",
                flush=True,
            )
            net_copy = copy.deepcopy(net)
            t_ul0 = time.time()
            lf, lr_, lt = unlearning_finetune_ye_multi(
                net_copy,
                forget_buffer=f_buf,
                retain_buffer=r_buf,
                candidate_movies=candidate_movies,
                num_iters=u_iters,
                batch_size=UNLEARN_BATCH,
                lambda_retain=lam,
                lr=u_lr,
                log_every=UNLEARN_LOG_INTERVAL,
                loss_log_rows=unlearn_loss_log_rows,
                loss_log_meta={
                    "train_lr": t_lr,
                    "gamma": gamma,
                    "hidden_dim": hidden_dim,
                    "train_batch": train_bs,
                    "unlearn_lr": u_lr,
                    "unlearn_iters": u_iters,
                    "lambda_retain": lam,
                    "method": method,
                },
            )
            unlearn_time_s = round(time.time() - t_ul0, 2)
            print(f" done in {unlearn_time_s:.1f}s (Lf={lf:.4f}, Lr={lr_:.4f})")

            ul_path = unlearned_model_path(
                t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam
            )
            if SAVE_UNLEARNED_MODELS:
                torch.save(net_copy.state_dict(), ul_path)

            append_eval_rows(
                all_results,
                net_after=net_copy,
                baseline=baseline,
                train_lr=t_lr,
                gamma=gamma,
                hidden_dim=hidden_dim,
                train_batch=train_bs,
                train_time_s=train_time_s,
                trained_model_path=t_model_path,
                unlearn_lr=u_lr,
                unlearn_iters=u_iters,
                lambda_retain=lam,
                method=method,
                unlearn_time_s=unlearn_time_s,
                unlearned_model_path=ul_path if SAVE_UNLEARNED_MODELS else "",
                loss_forget_final=lf,
                loss_retain_final=lr_,
                loss_total_final=lt,
            )

            progress_df = mark_done(
                progress_df, done_set,
                t_lr, gamma, hidden_dim, train_bs,
                u_lr, u_iters, lam, method,
            )
            ul_done += 1

        # ==============================================================
        # Method 3 — New_True_inf (lambda swept)
        # ==============================================================
        method = "New_True_inf"
        for lam in LAMBDA_VALS:
            combo_key = (t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam, method)
            set_seed(make_seed(t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam, method)) 

            if combo_key in done_set:
                ul_skipped += 1
                ul_done += 1
                print(
                    f" [SKIP {ul_done:>4}/{total_ul}] "
                    f"{method} u_lr={u_lr} u_iters={u_iters} lam={lam}"
                )
                continue

            print(
                f" [RUN {ul_done + 1:>4}/{total_ul}] "
                f"{method} u_lr={u_lr} u_iters={u_iters} lam={lam} ...",
                end="",
                flush=True,
            )
            net_copy = copy.deepcopy(net)
            t_ul0 = time.time()
            lf, lr_, lt = unlearning_finetune_new_true_inf(
                net_copy,
                forget_buffer=f_buf,
                retain_buffer=r_buf,
                candidate_movies=candidate_movies,
                num_iters=u_iters,
                batch_size=UNLEARN_BATCH,
                lambda_retain=lam,
                lr=u_lr,
                log_every=UNLEARN_LOG_INTERVAL,
                loss_log_rows=unlearn_loss_log_rows,
                loss_log_meta={
                    "train_lr": t_lr,
                    "gamma": gamma,
                    "hidden_dim": hidden_dim,
                    "train_batch": train_bs,
                    "unlearn_lr": u_lr,
                    "unlearn_iters": u_iters,
                    "lambda_retain": lam,
                    "method": method,
                },
            )
            unlearn_time_s = round(time.time() - t_ul0, 2)
            print(f" done in {unlearn_time_s:.1f}s (Lf={lf:.4f}, Lr={lr_:.4f})")

            ul_path = unlearned_model_path(
                t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam
            )
            if SAVE_UNLEARNED_MODELS:
                torch.save(net_copy.state_dict(), ul_path)

            append_eval_rows(
                all_results,
                net_after=net_copy,
                baseline=baseline,
                train_lr=t_lr,
                gamma=gamma,
                hidden_dim=hidden_dim,
                train_batch=train_bs,
                train_time_s=train_time_s,
                trained_model_path=t_model_path,
                unlearn_lr=u_lr,
                unlearn_iters=u_iters,
                lambda_retain=lam,
                method=method,
                unlearn_time_s=unlearn_time_s,
                unlearned_model_path=ul_path if SAVE_UNLEARNED_MODELS else "",
                loss_forget_final=lf,
                loss_retain_final=lr_,
                loss_total_final=lt,
            )

            progress_df = mark_done(
                progress_df, done_set,
                t_lr, gamma, hidden_dim, train_bs,
                u_lr, u_iters, lam, method,
            )
            ul_done += 1

        # ==============================================================
        # Method 4 — New_Max (lambda swept)
        # ==============================================================
        method = "New_Max"
        for lam in LAMBDA_VALS:
            combo_key = (t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam, method)
            set_seed(make_seed(t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam, method)) 

            if combo_key in done_set:
                ul_skipped += 1
                ul_done += 1
                print(
                    f" [SKIP {ul_done:>4}/{total_ul}] "
                    f"{method} u_lr={u_lr} u_iters={u_iters} lam={lam}"
                )
                continue

            print(
                f" [RUN {ul_done + 1:>4}/{total_ul}] "
                f"{method} u_lr={u_lr} u_iters={u_iters} lam={lam} ...",
                end="",
                flush=True,
            )
            net_copy = copy.deepcopy(net)
            t_ul0 = time.time()
            lf, lr_, lt = unlearning_finetune_new_max(
                net_copy,
                forget_buffer=f_buf,
                retain_buffer=r_buf,
                candidate_movies=candidate_movies,
                num_iters=u_iters,
                batch_size=UNLEARN_BATCH,
                lambda_retain=lam,
                lr=u_lr,
                log_every=UNLEARN_LOG_INTERVAL,
                loss_log_rows=unlearn_loss_log_rows,
                loss_log_meta={
                    "train_lr": t_lr,
                    "gamma": gamma,
                    "hidden_dim": hidden_dim,
                    "train_batch": train_bs,
                    "unlearn_lr": u_lr,
                    "unlearn_iters": u_iters,
                    "lambda_retain": lam,
                    "method": method,
                },
            )
            unlearn_time_s = round(time.time() - t_ul0, 2)
            print(f" done in {unlearn_time_s:.1f}s (Lf={lf:.4f}, Lr={lr_:.4f})")

            ul_path = unlearned_model_path(
                t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam
            )
            if SAVE_UNLEARNED_MODELS:
                torch.save(net_copy.state_dict(), ul_path)

            append_eval_rows(
                all_results,
                net_after=net_copy,
                baseline=baseline,
                train_lr=t_lr,
                gamma=gamma,
                hidden_dim=hidden_dim,
                train_batch=train_bs,
                train_time_s=train_time_s,
                trained_model_path=t_model_path,
                unlearn_lr=u_lr,
                unlearn_iters=u_iters,
                lambda_retain=lam,
                method=method,
                unlearn_time_s=unlearn_time_s,
                unlearned_model_path=ul_path if SAVE_UNLEARNED_MODELS else "",
                loss_forget_final=lf,
                loss_retain_final=lr_,
                loss_total_final=lt,
            )

            progress_df = mark_done(
                progress_df, done_set,
                t_lr, gamma, hidden_dim, train_bs,
                u_lr, u_iters, lam, method,
            )
            ul_done += 1

        # ==============================================================
        # Method 5 - Gradient Ascent
        # ==============================================================
        lam = 0.0
        method = "Gradient_Ascent"
        combo_key = (t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam, method)
        set_seed(make_seed(t_lr, gamma, hidden_dim, train_bs, u_lr, u_iters, lam, method)) 

        if combo_key in done_set:
            ul_skipped += 1
            ul_done += 1
            print(
                f" [SKIP {ul_done:>4}/{total_ul}] "
                f"{method} u_lr={u_lr} u_iters={u_iters} lam={lam}"
            )
        else:
            print(
                f" [RUN {ul_done + 1:>4}/{total_ul}] "
                f"{method} u_lr={u_lr} u_iters={u_iters} lam={lam} ...",
                end="",
                flush=True,
            )
            net_copy = copy.deepcopy(net)
            t_ul0 = time.time()
            
            env_for = MovieLensEnv(forget_trajectories, build_state_fn, candidate_movies)
            lf, lr_, lt = unlearning_gradient_ascent(
                env=env_for,
                policy_net=net_copy,
                num_iters=u_iters,
                batch_size=train_bs,
                lr=u_lr,
                gamma=gamma,
                max_steps_per_ep=MAX_STEPS,
                log_every=UNLEARN_LOG_INTERVAL,
                loss_log_rows=unlearn_loss_log_rows,
                loss_log_meta={
                    "train_lr": t_lr,
                    "gamma": gamma,
                    "hidden_dim": hidden_dim,
                    "train_batch": train_bs,
                    "unlearn_lr": u_lr,
                    "unlearn_iters": u_iters,
                    "lambda_retain": lam,
                    "method": method,
                },
            )
            unlearn_time_s = round(time.time() - t_ul0, 2)
            print(f" done in {unlearn_time_s:.1f}s (Lf={lf:.4f}, Lr={lr_:.4f})")

            ul_path = unlearned_model_path(
                t_lr, gamma, hidden_dim, train_bs, method, u_lr, u_iters, lam
            )
            if SAVE_UNLEARNED_MODELS:
                torch.save(net_copy.state_dict(), ul_path)

            append_eval_rows(
                all_results,
                net_after=net_copy,
                baseline=baseline,
                train_lr=t_lr,
                gamma=gamma,
                hidden_dim=hidden_dim,
                train_batch=train_bs,
                train_time_s=train_time_s,
                trained_model_path=t_model_path,
                unlearn_lr=u_lr,
                unlearn_iters=u_iters,
                lambda_retain=lam,
                method=method,
                unlearn_time_s=unlearn_time_s,
                unlearned_model_path=ul_path if SAVE_UNLEARNED_MODELS else "",
                loss_forget_final=lf,
                loss_retain_final=lr_,
                loss_total_final=lt,
            )

            progress_df = mark_done(
                progress_df, done_set,
                t_lr, gamma, hidden_dim, train_bs,
                u_lr, u_iters, lam, method,
            )
            ul_done += 1

    print(f" ✓ Config done — {ul_done - ul_skipped} ran, {ul_skipped} skipped")
    del net, r_buf
    torch.cuda.empty_cache()
