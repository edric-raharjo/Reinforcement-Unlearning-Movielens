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
    raise ValueError("Usage: python GPU_Enabled_Combine.py <forget_percentage>")

FORGET_PERCENTAGE = int(sys.argv[1])

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_DIR = "C:/Bob/ml-1m"
RESULTS_BASE = f"C:/Bob/results_deterministic/{FORGET_PERCENTAGE}_percent"
MODELS_DIR = os.path.join(RESULTS_BASE, "models")
RESULTS_PATH = os.path.join(RESULTS_BASE, "tuning_full_results.csv")
PROGRESS_PATH = os.path.join(RESULTS_BASE, "progress.csv")
TRAIN_RESULTS_PATH = os.path.join(RESULTS_BASE, "train_phase_results.csv")
TRAIN_PROGRESS_PATH = os.path.join(RESULTS_BASE, "train_phase_progress.csv")
F_BUF_PATH = os.path.join(RESULTS_BASE, "forget_buffer.pkl")

UNLEARN_LOSS_LOG_PATH = os.path.join(RESULTS_BASE, "unlearning_loss_log.csv")
UNLEARN_LOG_INTERVAL = 25

os.makedirs(MODELS_DIR, exist_ok=True)

SAVE_UNLEARNED_MODELS = True
TOP_PERCENT = 0.125
TOP_SELECTION_METRIC = "base_retain_Hit"
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
# Train / forget split
# ---------------------------------------------------------------------------

sample_users = np.array(sample_users)
np.random.shuffle(sample_users)

split_amt = int(np.round(FORGET_PERCENTAGE / 100 * pilot_ratings["user_id"].nunique()))
forget_users = sample_users[:split_amt]
retain_users = sample_users[split_amt:]

print(f"Forget users : {len(forget_users)}")
print(f"Retain users : {len(retain_users)}")


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
    after = eval_all_ks(net_after, retain_trajectories, forget_trajectories)
    for K in KS:
        h_r, n_r, h_f, n_f = after[K]
        bh_r, bn_r, bh_f, bn_f = baseline[K]
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
        print(f"✓ Train results: {len(df):,} rows")
        return df.to_dict("records")
    print("  No train results — starting fresh")
    return []


def load_results():
    if os.path.exists(RESULTS_PATH):
        df = pd.read_csv(RESULTS_PATH)
        before = len(df)
        df = df.drop_duplicates(subset=_KEY_COLS, keep="last")
        print(f"✓ Unlearn results: {len(df):,} rows ({before - len(df)} duplicates removed)")
        return df.to_dict("records")
    print("  No unlearn results — starting fresh")
    return []


def eval_all_ks(net, ret_trajs, for_trajs):
    out = {}
    for K in KS:
        h_r, n_r = evaluate_policy(net, ret_trajs, build_state_fn, candidate_movies, K=K)
        h_f, n_f = evaluate_policy(net, for_trajs, build_state_fn, candidate_movies, K=K)
        out[K] = (h_r, n_r, h_f, n_f)
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

print(f"\n{'#' * 72}")
print(f" PHASE 1 — Training {len(train_configs)} configs")
print(f"{'#' * 72}")

train_prog_df, train_done_set = load_train_progress(train_configs)
train_results = load_train_results()

for cfg_idx, (t_lr, gamma, hidden_dim, train_bs) in enumerate(train_configs):
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
        print("  Trainingfrom scratch...")
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

    baseline = eval_all_ks(net, retain_trajectories, forget_trajectories)

    for K in KS:
        h_r, n_r, h_f, n_f = baseline[K]
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
        })

    pd.DataFrame(train_results).to_csv(TRAIN_RESULTS_PATH, index=False)
    train_prog_df = mark_train_done(
        train_prog_df, train_done_set, t_lr, gamma, hidden_dim, train_bs
    )
    h_sel, n_sel = baseline[TOP_SELECTION_K][0], baseline[TOP_SELECTION_K][1]
    print(
        f"  ✓ Saved | "
        f"Retain Hit@{TOP_SELECTION_K}={h_sel:.4f} "
        f"NDCG@{TOP_SELECTION_K}={n_sel:.4f}"
    )

    del net
    torch.cuda.empty_cache()

print(f"\n✓ Phase 1 complete — {len(train_done_set)} models trained / loaded")


# ===========================================================================
# PHASE 1 → PHASE 2 — Select top configs by retain metric
# ===========================================================================

train_df = (
    pd.DataFrame(train_results)
    .drop_duplicates(subset=_TRAIN_KEY_COLS, keep="last")
)
rank_df = (
    train_df[train_df["K"] == TOP_SELECTION_K]
    .sort_values(
        [TOP_SELECTION_METRIC, "train_lr", "gamma", "hidden_dim", "train_batch"],
        ascending=[False, True, True, True, True],
        kind="mergesort",   # stable sort — tied rows always resolve the same way
    )
    .reset_index(drop=True)
)


n_top = max(1, int(len(rank_df) * TOP_PERCENT))
top_configs_df = rank_df.head(n_top)
top_configs = [
    (row["train_lr"], row["gamma"], int(row["hidden_dim"]), int(row["train_batch"]))
    for _, row in top_configs_df.iterrows()
]

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
all_results = load_results()
unlearn_loss_log_rows = load_unlearn_loss_log()

for cfg_idx, (t_lr, gamma, hidden_dim, train_bs) in enumerate(top_configs):

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

    if all_done_fixed and all_done_swept:
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

    prev = [
        r for r in train_results
        if (
            r["train_lr"] == t_lr
            and r["gamma"] == gamma
            and r["hidden_dim"] == hidden_dim
            and r["train_batch"] == train_bs
        )
    ]
    train_time_s = prev[0]["train_time_s"] if prev else float("nan")
    t_model_path = trained_model_path(t_lr, gamma, hidden_dim, train_bs)

    net = PolicyNet(state_dim, num_actions, hidden_dim=hidden_dim).to(DEVICE)
    net.load_state_dict(torch.load(t_model_path, map_location=DEVICE))
    print(f" ↩ Loaded {t_model_path}")

    baseline = {
        r["K"]: (
            r["base_retain_Hit"],
            r["base_retain_NDCG"],
            r["base_forget_Hit"],
            r["base_forget_NDCG"],
        )
        for r in train_results
        if (
            r["train_lr"] == t_lr
            and r["gamma"] == gamma
            and r["hidden_dim"] == hidden_dim
            and r["train_batch"] == train_bs
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
    total_ul = len(unlearn_configs) * (2 + 2 * len(LAMBDA_VALS))
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

    print(f" ✓ Config done — {ul_done - ul_skipped} ran, {ul_skipped} skipped")
    del net, r_buf
    torch.cuda.empty_cache()
