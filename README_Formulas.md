# Unlearning Methods

This experiment compares four unlearning methods in the same MovieLens reinforcement learning pipeline.  
All four methods start from the same trained policy and finetune it using forget-buffer and retain-buffer samples, but they differ in how the forget and retain losses are defined [file:101].

## Method 1: Ye_ApxI

**Ye_ApxI** follows the Appendix I style formulation.  
It uses sampled state-action pairs from the forget and retain buffers rather than the full action-value vector at each state [file:101].

### Loss
\[
L = \mathbb{E}_{(s,a)\sim \tau_u}[Q_{\pi'}(s,a)] + \mathbb{E}_{(s,a)\not\sim \tau_u}\left|Q_{\pi'}(s,a) - Q_{\pi}(s,a)\right|
\]

### Interpretation
- The forget term directly suppresses the value of sampled forget actions.
- The retain term keeps the new policy close to the original policy on sampled retain actions.
- Lambda is fixed to 1.

### Key property
This is the most **trajectory-level** method in the comparison, because it works on sampled `(state, action)` pairs instead of whole Q-vectors.

---

## Method 2: Ye_multi

**Ye_multi** follows Ye's multi-environment formulation more directly.  
It works at the state level and uses the infinity norm over the action-value vector.

### Loss
\[
L = \mathbb{E}_{s \sim S_u}\left[\|Q_{\pi'}(s)\|_\infty\right] + \mathbb{E}_{s \not\sim S_u}\left[\|Q_{\pi'}(s) - Q_{\pi}(s)\|_\infty\right]
\]

### Interpretation
- The forget term suppresses the largest-magnitude action value in forget states.
- The retain term preserves the largest change-sensitive component of the Q-vector in retain states.
- Lambda is fixed to 1.

### Key property
This is the most faithful **state-level multi-environment** version in the comparison.

---

## Method 3: New_True_inf

**New_True_inf** keeps the same state-level multi-environment structure as `Ye_multi`, but modifies the penalty shape.  
Both terms are squared, and the retain term is weighted by a tunable lambda.

### Loss
\[
L = \mathbb{E}_{s \sim S_u}\left[\|Q_{\pi'}(s)\|_\infty^2\right] + \lambda \, \mathbb{E}_{s \not\sim S_u}\left[\|Q_{\pi'}(s) - Q_{\pi}(s)\|_\infty^2\right]
\]

### Interpretation
- The forget term penalizes large-magnitude Q-values more aggressively than `Ye_multi`.
- The retain term also penalizes large deviations more aggressively.
- Lambda controls the tradeoff between forgetting and retention.

### Key property
This is the **principled modified** version of Ye's multi-environment loss.

---

## Method 4: New_Max

**New_Max** is the current implementation-style variant.  
It matches `New_True_inf` on the retain side, but its forget term uses `max(Q)` instead of the true infinity norm `max(|Q|)`.

### Loss
\[
L = \mathbb{E}_{s \sim S_u}\left[\left(\max_a Q_{\pi'}(s,a)\right)^2\right] + \lambda \, \mathbb{E}_{s \not\sim S_u}\left[\|Q_{\pi'}(s) - Q_{\pi}(s)\|_\infty^2\right]
\]

### Interpretation
- The forget term focuses on the most positive action value rather than the largest-magnitude action value.
- The retain term is the same squared infinity-norm retain penalty used in `New_True_inf`.
- Lambda is tuned.

### Key property
This is the closest method to the current implementation baseline and tests whether using `max(Q)` instead of the true infinity norm materially changes results.

---

## Practical summary

- **Ye_ApxI**: action-level, single-environment, lambda fixed to 1.
- **Ye_multi**: state-level, true infinity norm, lambda fixed to 1.
- **New_True_inf**: state-level, true infinity norm, squared terms, lambda tuned.
- **New_Max**: state-level, signed max forget term, squared terms, lambda tuned.

The main comparison separates:
1. action-level vs state-level unlearning,
2. original Ye loss vs modified squared loss,
3. true infinity norm vs signed max surrogate.
