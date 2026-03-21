import pandas as pd
import os

DATA_DIR = "C:/Bob/ml-1m"

ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.dat"),
    sep="::", engine="python", names=["user_id", "movie_id", "rating", "timestamp"])
movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.dat"),
    sep="::", engine="python", names=["movie_id", "title", "genres"], encoding="ISO-8859-1")
users_df = pd.read_csv(os.path.join(DATA_DIR, "users.dat"),
    sep="::", engine="python", names=["user_id", "gender", "age", "occupation", "zip"])

# ── FIX 1: Narrowed sensitive genres ─────────────────────────────────────────
SENSITIVE_GENRES = {"Documentary", "Film-Noir", "War", "Crime"}

movie_sensitive_flag = {
    row["movie_id"]: bool(SENSITIVE_GENRES & set(str(row["genres"]).split("|")))
    for _, row in movies_df.iterrows()
}

total_sensitive = sum(movie_sensitive_flag.values())
print(f"Sensitive movies : {total_sensitive} / {len(movie_sensitive_flag)} "
      f"({100 * total_sensitive / len(movie_sensitive_flag):.1f}%)")
print("  (target: ~10-15%, down from 65.2%)")

# Per-user sensitive fraction
user_sens_frac = {}
for uid, grp in ratings_df.groupby("user_id"):
    total = len(grp)
    sens_count = sum(movie_sensitive_flag.get(mid, False) for mid in grp["movie_id"])
    user_sens_frac[uid] = sens_count / total if total > 0 else 0.0

frac_series = pd.Series(user_sens_frac)
print(f"\nuser_sens_frac distribution:")
print(frac_series.describe())

bins   = [0.0, 0.2, 0.4, 0.7, 1.01]
labels = ["<0.20 (mult=1.00)", "0.20-0.40 (mult=1.15)",
          "0.40-0.70 (mult=1.30)", ">=0.70 (mult=1.50)"]
print("\n  Users per genre-multiplier tier (after fix):")
for i in range(len(bins) - 1):
    n = ((frac_series >= bins[i]) & (frac_series < bins[i+1])).sum()
    print(f"    {labels[i]} : {n} users ({100*n/len(frac_series):.1f}%)")

# ── FIX 2: Compute mean multiplier → auto-calibrate base_prob ────────────────
_users_meta = users_df.set_index("user_id")
all_uids = ratings_df["user_id"].unique()

multipliers = []
for uid in all_uids:
    m = 1.0
    if uid in _users_meta.index:
        row = _users_meta.loc[uid]
        if row["gender"] == "F":          m *= 1.35
        if row["age"] in (1, 18):         m *= 1.25
        elif row["age"] == 56:            m *= 1.15
        if row["occupation"] in (4, 10):  m *= 1.25
    f = user_sens_frac.get(uid, 0.0)
    if f >= 0.70:   m *= 1.50
    elif f >= 0.40: m *= 1.30
    elif f >= 0.20: m *= 1.15
    multipliers.append(m)

mean_mult = pd.Series(multipliers).mean()
print(f"\nMean combined multiplier : {mean_mult:.4f}")
print(f"  (was 1.65 before fix; ideally closer to 1.0–1.3 for good differentiation)")

print("\n  Auto-calibrated base_prob per target %:")
for target_pct in [0.01, 0.02, 0.03, 0.04, 0.05]:
    calibrated = target_pct / mean_mult
    expected_n = calibrated * mean_mult * len(all_uids)
    print(f"    target={int(target_pct*100)}%  →  "
          f"base_prob={calibrated:.5f}  →  expected ~{expected_n:.0f} forget users")

print("\n✓ Re-diagnostic complete.")
