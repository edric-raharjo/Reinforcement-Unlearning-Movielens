#!/usr/bin/env python
# coding: utf-8

import copy
import hashlib
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import re
import sys
import time
from collections import deque
from pathlib import Path

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
torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
NUM_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
WORKER_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 0
assert 0 <= WORKER_ID < NUM_WORKERS, "worker_id must be in [0, num_workers)"

BASE_MODE = "Normal"
BASE_FORGET_PCT = 1
BASE_THRESHOLDS_PP = [0.0, 1.0, 2.0, 5.0, 10.0]
TOP_SELECTION_K = 10
TARGET_FORGET_COUNT = 60  # 1% of 6040 users

TARGET_METHODS = ["Ye_multi", "New_True_inf", "Gradient_Ascent"]
KS = [1, 5, 10]
MAX_STEPS = 30
UNLEARN_BATCH = 64
FORGET_BUFFER_STEPS = 100_000
RETAIN_BUFFER_STEPS = 100_000
UNLEARN_LOG_INTERVAL = 25
SAVE_UNLEARNED_MODELS = True

DATA_DIR_CANDIDATES = [
    "C:/Bob/ml-1m",
    "D:/Bob/ml-1m",
    str((Path(__file__).resolve().parents[2] / "data_movie").resolve()),
]

SOURCE_RESULTS_CSV_CANDIDATES = [
    str((Path(__file__).resolve().parents[2] / "Raw" / "Normal" / "1_percent" / "tuning_full_results.csv").resolve()),
    "C:/Bob/results/1_percent/tuning_full_results.csv",
    "D:/Bob_Skripsi_Do Not Delete/results/1_percent/tuning_full_results.csv",
]

RESULTS_ROOT = "D:/Bob_Skripsi_Do Not Delete/results_ugp_analysis"
METRICS_DIR = os.path.join(RESULTS_ROOT, "metrics")
MODELS_DIR = os.path.join(RESULTS_ROOT, "models")
LOCKS_DIR = os.path.join(RESULTS_ROOT, "locks")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOCKS_DIR, exist_ok=True)

METRICS_PATH = os.path.join(METRICS_DIR, "relearn_metrics.csv")
PROGRESS_PATH = os.path.join(METRICS_DIR, "relearn_progress.csv")
SELECTION_SUMMARY_PATH = os.path.join(METRICS_DIR, "relearn_selection_summary.csv")
LOSS_LOG_PATH = os.path.join(METRICS_DIR, "relearn_loss_log.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"Using device: {DEVICE} "
    f"({'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
)


# ---------------------------------------------------------------------------
# Constants and labels
# ---------------------------------------------------------------------------
AGE_LABELS = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}

OCCUPATION_LABELS = {
    0: "Other / not specified",
    1: "Academic/Educator",
    2: "Artist",
    3: "Clerical/Admin",
    4: "College/Grad Student",
    5: "Customer Service",
    6: "Doctor/Health Care",
    7: "Executive/Managerial",
    8: "Farmer",
    9: "Homemaker",
    10: "K-12 Student",
    11: "Lawyer",
    12: "Programmer",
    13: "Retired",
    14: "Sales/Marketing",
    15: "Scientist",
    16: "Self-Employed",
    17: "Technician/Engineer",
    18: "Tradesman/Craftsman",
    19: "Unemployed",
    20: "Writer",
}


def slugify(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if float(v).is_integer():
        return str(int(v))
    if abs(v) < 0.01:
        return f"{v:.0e}".replace("-", "n").replace("+", "p")
    return str(v).replace(".", "d")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_existing_path(candidates, description):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{description} not found. Candidates:\n" + "\n".join(candidates))


DATA_DIR = resolve_existing_path(DATA_DIR_CANDIDATES, "MovieLens data directory")
SOURCE_RESULTS_CSV = resolve_existing_path(SOURCE_RESULTS_CSV_CANDIDATES, "Source tuning_full_results.csv")

print(f"DATA_DIR           : {DATA_DIR}")
print(f"SOURCE_RESULTS_CSV : {SOURCE_RESULTS_CSV}")
print(f"RESULTS_ROOT       : {RESULTS_ROOT}")
print(f"NUM_WORKERS        : {NUM_WORKERS}")
print(f"WORKER_ID          : {WORKER_ID}")


# ---------------------------------------------------------------------------
# Data loading and feature engineering
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
pilot_ratings_all = ratings_df[ratings_df["user_id"].isin(sample_users)].copy()
pilot_ratings_all.sort_values(["user_id", "timestamp"], inplace=True)

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
# Environment and model
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


def eval_all_ks(net, ret_trajs, for_trajs, all_trajs):
    out = {}
    for K in KS:
        h_r, n_r = evaluate_policy(net, ret_trajs, build_state_fn, candidate_movies, K=K)
        h_f, n_f = evaluate_policy(net, for_trajs, build_state_fn, candidate_movies, K=K)
        h_c, n_c = evaluate_policy(net, all_trajs, build_state_fn, candidate_movies, K=K)
        out[K] = (h_r, n_r, h_f, n_f, h_c, n_c)
    return out


# ---------------------------------------------------------------------------
# Replay buffer and collection
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


def collect_random_experience_forget(env_forget, num_steps, buffer):
    state = env_forget.reset()
    for _ in range(num_steps):
        action = np.random.randint(env_forget.num_actions)
        next_state, reward, done = env_forget.step(action)
        s_next = np.zeros_like(state) if next_state is None else next_state
        buffer.push(state, action, reward, s_next, float(done))
        state = env_forget.reset() if done else next_state


def collect_policy_experience(env, policy_net, num_steps, buffer):
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
# Loss log helpers
# ---------------------------------------------------------------------------
LOSS_KEY_COLS = [
    "base_threshold_pp", "setting_id", "method", "iter"
]


def load_loss_log():
    if os.path.exists(LOSS_LOG_PATH):
        df = pd.read_csv(LOSS_LOG_PATH)
        if "base_threshold_pp" not in df.columns:
            df["base_threshold_pp"] = 5.0
        df = df.drop_duplicates(subset=LOSS_KEY_COLS, keep="last")
        return df.to_dict("records")
    return []


def save_loss_log(rows):
    if rows:
        pd.DataFrame(rows).drop_duplicates(subset=LOSS_KEY_COLS, keep="last").to_csv(LOSS_LOG_PATH, index=False)


def append_loss_row(loss_rows, meta, iter_idx, loss_forget, loss_retain, loss_total):
    row = {
        "base_threshold_pp": meta["base_threshold_pp"],
        "setting_id": meta["setting_id"],
        "setting_label": meta["setting_label"],
        "method": meta["method"],
        "train_lr": meta["train_lr"],
        "gamma": meta["gamma"],
        "hidden_dim": meta["hidden_dim"],
        "train_batch": meta["train_batch"],
        "unlearn_lr": meta["unlearn_lr"],
        "unlearn_iters": meta["unlearn_iters"],
        "lambda_retain": meta["lambda_retain"],
        "iter": iter_idx,
        "loss_forget": loss_forget,
        "loss_retain": loss_retain,
        "loss_total": loss_total,
    }
    loss_rows.append(row)
    save_loss_log(loss_rows)


# ---------------------------------------------------------------------------
# Unlearning methods
# ---------------------------------------------------------------------------
def unlearning_finetune_ye_multi(
    policy_net,
    forget_buffer,
    retain_buffer,
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

        if (it + 1) % log_every == 0 and loss_log_rows is not None and loss_log_meta is not None:
            append_loss_row(loss_log_rows, loss_log_meta, it + 1, last_lf, last_lr, last_lt)

    return last_lf, last_lr, last_lt


def unlearning_finetune_new_true_inf(
    policy_net,
    forget_buffer,
    retain_buffer,
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

        if (it + 1) % log_every == 0 and loss_log_rows is not None and loss_log_meta is not None:
            append_loss_row(loss_log_rows, loss_log_meta, it + 1, last_lf, last_lr, last_lt)

    return last_lf, last_lr, last_lt


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

        if (ep % log_every == 0 or ep >= num_iters) and loss_log_rows is not None and loss_log_meta is not None:
            append_loss_row(loss_log_rows, loss_log_meta, ep, last_lf, last_lr, last_lt)

    return last_lf, last_lr, last_lt


# ---------------------------------------------------------------------------
# Selection summary and source row choice
# ---------------------------------------------------------------------------
def select_source_rows(source_csv):
    df = pd.read_csv(source_csv).copy()
    df["source_row_id"] = np.arange(len(df))

    numeric_cols = [
        "K", "train_lr", "gamma", "hidden_dim", "train_batch",
        "unlearn_lr", "unlearn_iters", "lambda_retain",
        "retain_Hit", "forget_Hit", "base_retain_Hit", "base_forget_Hit",
        "train_time_s", "unlearn_time_s",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["retain_drop_hit_pp"] = (df["base_retain_Hit"] - df["retain_Hit"]) * 100.0
    df["forget_drop_hit_pp"] = (df["base_forget_Hit"] - df["forget_Hit"]) * 100.0

    selected = []
    for threshold_pp in BASE_THRESHOLDS_PP:
        for method in TARGET_METHODS:
            cand = df[
                (df["method"] == method)
                & (df["K"] == TOP_SELECTION_K)
                & (df["retain_drop_hit_pp"] <= threshold_pp)
            ].copy()
            if cand.empty:
                raise ValueError(
                    f"No valid source row found for method={method} threshold={threshold_pp}"
                )
            cand = cand.sort_values(
                ["forget_drop_hit_pp", "lambda_retain"],
                ascending=[False, True],
                kind="mergesort",
            ).reset_index(drop=True)
            chosen = cand.iloc[0].to_dict()
            chosen["base_threshold_pp"] = float(threshold_pp)
            selected.append(chosen)

    summary_df = pd.DataFrame(selected)
    summary_df["base_mode"] = BASE_MODE
    summary_df["base_forget_pct"] = BASE_FORGET_PCT
    return summary_df


selection_summary_df = select_source_rows(SOURCE_RESULTS_CSV)
selection_summary_df.to_csv(SELECTION_SUMMARY_PATH, index=False)
print("Selected source rows:")
print(selection_summary_df[[
    "base_threshold_pp", "method", "source_row_id", "train_lr", "gamma", "hidden_dim", "train_batch",
    "trained_model_path", "unlearn_lr", "unlearn_iters", "lambda_retain",
    "retain_drop_hit_pp", "forget_drop_hit_pp"
]].to_string(index=False))


# ---------------------------------------------------------------------------
# UGP settings
# ---------------------------------------------------------------------------
def build_settings():
    settings = [
        {"setting_id": 1, "setting_type": "gender", "setting_value_raw": "F", "setting_label": "gender_female"},
        {"setting_id": 2, "setting_type": "gender", "setting_value_raw": "M", "setting_label": "gender_male"},
    ]

    next_id = 3
    for age in [1, 18, 25, 35, 45, 50, 56]:
        settings.append({
            "setting_id": next_id,
            "setting_type": "age",
            "setting_value_raw": age,
            "setting_label": f"age_{slugify(AGE_LABELS[age])}",
        })
        next_id += 1

    for occ in range(21):
        settings.append({
            "setting_id": next_id,
            "setting_type": "occupation",
            "setting_value_raw": occ,
            "setting_label": f"occupation_{occ}_{slugify(OCCUPATION_LABELS[occ])}",
        })
        next_id += 1

    if len(settings) != 30:
        raise ValueError(f"Expected 30 settings, got {len(settings)}")
    return settings


SETTINGS = build_settings()


def select_forget_users_for_setting(setting, users_meta):
    feature = setting["setting_type"]
    raw_value = setting["setting_value_raw"]
    subset = users_meta[users_meta[feature] == raw_value].copy().sort_index()
    category_user_count = len(subset)
    if category_user_count == 0:
        return np.array([], dtype=int), category_user_count
    forget_user_count = min(TARGET_FORGET_COUNT, category_user_count)
    forget_users = subset.index.to_numpy(dtype=int)[:forget_user_count]
    return forget_users, category_user_count


# ---------------------------------------------------------------------------
# Progress and metrics persistence
# ---------------------------------------------------------------------------
PROGRESS_KEY_COLS = ["base_threshold_pp", "setting_id", "method"]


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        df = pd.read_csv(PROGRESS_PATH)
        if "base_threshold_pp" not in df.columns:
            df["base_threshold_pp"] = 5.0
        df = df.drop_duplicates(subset=PROGRESS_KEY_COLS, keep="last")
        return df, set(tuple(r) for r in df[PROGRESS_KEY_COLS].itertuples(index=False, name=None))
    cols = PROGRESS_KEY_COLS + ["status", "unlearned_model_path"]
    return pd.DataFrame(columns=cols), set()


def mark_progress(progress_df, done_set, base_threshold_pp, setting_id, method, status, unlearned_model_path):
    key = (base_threshold_pp, setting_id, method)
    if key not in done_set:
        done_set.add(key)
    row = pd.DataFrame([{
        "base_threshold_pp": base_threshold_pp,
        "setting_id": setting_id,
        "method": method,
        "status": status,
        "unlearned_model_path": unlearned_model_path,
    }])
    progress_df = pd.concat([progress_df, row], ignore_index=True)
    progress_df = progress_df.drop_duplicates(subset=PROGRESS_KEY_COLS, keep="last")
    progress_df.to_csv(PROGRESS_PATH, index=False)
    return progress_df


def load_metrics():
    if os.path.exists(METRICS_PATH):
        df = pd.read_csv(METRICS_PATH)
        if "base_threshold_pp" not in df.columns:
            df["base_threshold_pp"] = 5.0
        return df.to_dict("records")
    return []


def save_metrics(metric_rows):
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(METRICS_PATH, index=False)


def build_model_output_path(setting, source_row):
    return os.path.join(
        MODELS_DIR,
        "ugp__thr{thr}__s{sid:02d}__{label}__{method}__tlr{tlr}__g{g}__h{h}__bs{bs}__ulr{ulr}__ui{ui}__lam{lam}.pt".format(
            thr=_fmt(float(source_row["base_threshold_pp"])),
            sid=setting["setting_id"],
            label=slugify(setting["setting_label"]),
            method=source_row["method"],
            tlr=_fmt(source_row["train_lr"]),
            g=_fmt(source_row["gamma"]),
            h=int(source_row["hidden_dim"]),
            bs=int(source_row["train_batch"]),
            ulr=_fmt(source_row["unlearn_lr"]),
            ui=int(source_row["unlearn_iters"]),
            lam=_fmt(source_row["lambda_retain"]),
        ),
    )


def append_metric_rows(
    metric_rows,
    *,
    setting,
    category_user_count,
    forget_user_count,
    retain_user_count,
    source_row,
    seed_run,
    unlearn_time_s,
    loss_forget_final,
    loss_retain_final,
    loss_total_final,
    unlearned_model_path,
    baseline,
    after,
):
    for K in KS:
        bh_r, bn_r, bh_f, bn_f, bh_c, bn_c = baseline[K]
        h_r, n_r, h_f, n_f, h_c, n_c = after[K]
        metric_rows.append({
            "setting_id": setting["setting_id"],
            "setting_type": setting["setting_type"],
            "setting_value_raw": setting["setting_value_raw"],
            "setting_label": setting["setting_label"],
            "category_user_count": category_user_count,
            "forget_user_count": forget_user_count,
            "retain_user_count": retain_user_count,
            "method": source_row["method"],
            "base_mode": BASE_MODE,
            "base_forget_pct": BASE_FORGET_PCT,
            "base_threshold_pp": float(source_row["base_threshold_pp"]),
            "source_row_id": int(source_row["source_row_id"]),
            "train_lr": float(source_row["train_lr"]),
            "gamma": float(source_row["gamma"]),
            "hidden_dim": int(source_row["hidden_dim"]),
            "train_batch": int(source_row["train_batch"]),
            "trained_model_path": source_row["trained_model_path"],
            "unlearn_lr": float(source_row["unlearn_lr"]),
            "unlearn_iters": int(source_row["unlearn_iters"]),
            "lambda_retain": float(source_row["lambda_retain"]),
            "unlearned_model_path": unlearned_model_path,
            "seed_run": seed_run,
            "unlearn_time_s": unlearn_time_s,
            "loss_forget_final": loss_forget_final,
            "loss_retain_final": loss_retain_final,
            "loss_total_final": loss_total_final,
            "K": K,
            "retain_Hit": h_r,
            "retain_NDCG": n_r,
            "forget_Hit": h_f,
            "forget_NDCG": n_f,
            "combined_Hit": h_c,
            "combined_NDCG": n_c,
            "base_retain_Hit": bh_r,
            "base_retain_NDCG": bn_r,
            "base_forget_Hit": bh_f,
            "base_forget_NDCG": bn_f,
            "base_combined_Hit": bh_c,
            "base_combined_NDCG": bn_c,
        })
    save_metrics(metric_rows)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
progress_df, done_set = load_progress()
metric_rows = load_metrics()
loss_log_rows = load_loss_log()

selection_by_method = {
    (float(row["base_threshold_pp"]), row["method"]): row
    for _, row in selection_summary_df.iterrows()
}

jobs = [
    (threshold_pp, setting, method)
    for threshold_pp in BASE_THRESHOLDS_PP
    for setting in SETTINGS
    for method in TARGET_METHODS
]
jobs = [job for idx, job in enumerate(jobs) if idx % NUM_WORKERS == WORKER_ID]

print(f"Total thresholds         : {len(BASE_THRESHOLDS_PP)}")
print(f"Total settings           : {len(SETTINGS)}")
print(f"Total target methods     : {len(TARGET_METHODS)}")
print(f"Total jobs (global)      : {len(BASE_THRESHOLDS_PP) * len(SETTINGS) * len(TARGET_METHODS)}")
print(f"Worker {WORKER_ID} jobs  : {len(jobs)}")

all_users_meta = users_df[users_df["user_id"].isin(sample_users)].copy().set_index("user_id")

for job_idx, (threshold_pp, setting, method) in enumerate(jobs, start=1):
    key = (float(threshold_pp), setting["setting_id"], method)
    if key in done_set:
        print(
            f"[SKIP {job_idx:>3}/{len(jobs)}] "
            f"threshold={threshold_pp:.1f} setting={setting['setting_id']:02d} method={method} already done"
        )
        continue

    source_row = selection_by_method[(float(threshold_pp), method)]
    forget_users, category_user_count = select_forget_users_for_setting(setting, all_users_meta)
    retain_users = np.array(sorted(set(sample_users) - set(forget_users.tolist())), dtype=int)

    forget_user_count = len(forget_users)
    retain_user_count = len(retain_users)

    print(
        f"[RUN  {job_idx:>3}/{len(jobs)}] "
        f"threshold={threshold_pp:.1f} | "
        f"setting={setting['setting_id']:02d} {setting['setting_label']} | "
        f"method={method} | category={category_user_count} | "
        f"forget={forget_user_count} | retain={retain_user_count}"
    )

    if forget_user_count == 0:
        progress_df = mark_progress(
            progress_df, done_set, float(threshold_pp), setting["setting_id"], method, "NO_USERS", ""
        )
        continue

    forget_user_set = set(forget_users.tolist())
    retain_user_set = set(retain_users.tolist())

    forget_trajectories = [t for t in trajectories_all if t["user_id"] in forget_user_set]
    retain_trajectories = [t for t in trajectories_all if t["user_id"] in retain_user_set]

    seed_run = make_seed("ugp_analysis", float(threshold_pp), setting["setting_id"], method)
    set_seed(seed_run)

    trained_model_path = source_row["trained_model_path"]
    if not os.path.exists(trained_model_path):
        raise FileNotFoundError(f"Trained model path not found for {method}: {trained_model_path}")

    hidden_dim = int(source_row["hidden_dim"])
    net = PolicyNet(state_dim, num_actions, hidden_dim=hidden_dim).to(DEVICE)
    net.load_state_dict(torch.load(trained_model_path, map_location=DEVICE))
    net.eval()

    baseline = eval_all_ks(net, retain_trajectories, forget_trajectories, trajectories_all)

    forget_env = MovieLensEnv(forget_trajectories, build_state_fn, candidate_movies)
    retain_env = MovieLensEnv(retain_trajectories, build_state_fn, candidate_movies)

    forget_buffer = ReplayBuffer(capacity=FORGET_BUFFER_STEPS)
    collect_random_experience_forget(forget_env, num_steps=FORGET_BUFFER_STEPS, buffer=forget_buffer)

    retain_buffer = ReplayBuffer(capacity=RETAIN_BUFFER_STEPS)
    collect_policy_experience(retain_env, net, num_steps=RETAIN_BUFFER_STEPS, buffer=retain_buffer)

    net_copy = copy.deepcopy(net)
    unlearned_model_path = build_model_output_path(setting, source_row)
    loss_meta = {
        "base_threshold_pp": float(threshold_pp),
        "setting_id": setting["setting_id"],
        "setting_label": setting["setting_label"],
        "method": method,
        "train_lr": float(source_row["train_lr"]),
        "gamma": float(source_row["gamma"]),
        "hidden_dim": int(source_row["hidden_dim"]),
        "train_batch": int(source_row["train_batch"]),
        "unlearn_lr": float(source_row["unlearn_lr"]),
        "unlearn_iters": int(source_row["unlearn_iters"]),
        "lambda_retain": float(source_row["lambda_retain"]),
    }

    t0 = time.time()
    if method == "Ye_multi":
        lf, lr_, lt = unlearning_finetune_ye_multi(
            net_copy,
            forget_buffer=forget_buffer,
            retain_buffer=retain_buffer,
            num_iters=int(source_row["unlearn_iters"]),
            batch_size=UNLEARN_BATCH,
            lambda_retain=float(source_row["lambda_retain"]),
            lr=float(source_row["unlearn_lr"]),
            log_every=UNLEARN_LOG_INTERVAL,
            loss_log_rows=loss_log_rows,
            loss_log_meta=loss_meta,
        )
    elif method == "New_True_inf":
        lf, lr_, lt = unlearning_finetune_new_true_inf(
            net_copy,
            forget_buffer=forget_buffer,
            retain_buffer=retain_buffer,
            num_iters=int(source_row["unlearn_iters"]),
            batch_size=UNLEARN_BATCH,
            lambda_retain=float(source_row["lambda_retain"]),
            lr=float(source_row["unlearn_lr"]),
            log_every=UNLEARN_LOG_INTERVAL,
            loss_log_rows=loss_log_rows,
            loss_log_meta=loss_meta,
        )
    elif method == "Gradient_Ascent":
        lf, lr_, lt = unlearning_gradient_ascent(
            env=forget_env,
            policy_net=net_copy,
            num_iters=int(source_row["unlearn_iters"]),
            batch_size=int(source_row["train_batch"]),
            lr=float(source_row["unlearn_lr"]),
            gamma=float(source_row["gamma"]),
            max_steps_per_ep=MAX_STEPS,
            log_every=UNLEARN_LOG_INTERVAL,
            loss_log_rows=loss_log_rows,
            loss_log_meta=loss_meta,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    unlearn_time_s = round(time.time() - t0, 2)
    after = eval_all_ks(net_copy, retain_trajectories, forget_trajectories, trajectories_all)

    if SAVE_UNLEARNED_MODELS:
        torch.save(net_copy.state_dict(), unlearned_model_path)

    append_metric_rows(
        metric_rows,
        setting=setting,
        category_user_count=category_user_count,
        forget_user_count=forget_user_count,
        retain_user_count=retain_user_count,
        source_row=source_row,
        seed_run=seed_run,
        unlearn_time_s=unlearn_time_s,
        loss_forget_final=lf,
        loss_retain_final=lr_,
        loss_total_final=lt,
        unlearned_model_path=unlearned_model_path if SAVE_UNLEARNED_MODELS else "",
        baseline=baseline,
        after=after,
    )

    progress_df = mark_progress(
        progress_df,
        done_set,
        float(threshold_pp),
        setting["setting_id"],
        method,
        "DONE",
        unlearned_model_path if SAVE_UNLEARNED_MODELS else "",
    )

    del net, net_copy, forget_buffer, retain_buffer
    torch.cuda.empty_cache()

print("UGP analysis complete.")
