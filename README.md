# Unlearning Methods

This experiment compares four unlearning methods in the same MovieLens reinforcement learning pipeline.  
All four methods start from the same trained policy and finetune it using forget-buffer and retain-buffer samples, but they differ in how the forget and retain losses are defined.

## Method 1: Ye_ApxI

**Ye_ApxI** follows the Appendix I style formulation.  
It uses sampled state-action pairs from the forget and retain buffers rather than the full action-value vector at each state.

### Loss
$$L = \mathbb{E}_{(s,a)\sim \tau_u}[Q_{\pi'}(s,a)] + \mathbb{E}_{(s,a)\not\sim \tau_u}|Q_{\pi'}(s,a) - Q_{\pi}(s,a)|$$

### Notation
* **$L$**: The total loss function being calculated.
* **$\mathbb{E}$**: Expectation, meaning the average value over the sampled distribution.
* **$(s,a)$**: A specific state ($s$) and action ($a$) pair.
* **$\sim \tau_u$**: Sampled from the forget buffer ($\tau_u$ represents the distribution of trajectories to unlearn).
* **$\not\sim \tau_u$**: Sampled from the retain buffer (trajectories to keep).
* **$\pi'$ (Pi Prime)**: The accent mark is called a **"prime"** symbol. Here, $\pi'$ represents the **new, updated policy** that is actively being finetuned (associated with the forget environment).
* **$\pi$ (Regular Pi)**: The **original, baseline trained policy** (associated with the retain environment, acting as an anchor).
* **$Q_{\pi'}(s,a)$ / $Q_{\pi}(s,a)$**: The predicted action-value (Q-value) for taking action $a$ in state $s$ under policy $\pi'$ or $\pi$.
* **$|\cdot|$**: Absolute value norm, representing the simple magnitude of the difference between the two scalar Q-values.

### Interpretation
- The forget term directly suppresses the value of sampled forget actions.
- The retain term keeps the new policy close to the original policy on sampled retain actions.
- $\lambda$ is fixed to 1.

### Key property
This is the most **trajectory-level** method in the comparison, because it works on sampled $(s, a)$ pairs instead of whole Q-vectors.

---

## Method 2: Ye_multi

**Ye_multi** follows Ye's multi-environment formulation more directly.  
It works at the state level and uses the infinity norm over the action-value vector.

### Loss
$$L = \mathbb{E}_{s \sim S_u}[\|Q_{\pi'}(s)\|_\infty] + \mathbb{E}_{s \not\sim S_u}[\|Q_{\pi'}(s) - Q_{\pi}(s)\|_\infty]$$

### Notation
* **$s \sim S_u$**: States ($s$) sampled from the forget buffer's state distribution ($S_u$). This differs from Method 1 as it isolates states rather than specific actions.
* **$s \not\sim S_u$**: States sampled from the retain buffer.
* **$Q_{\pi'}(s)$ / $Q_{\pi}(s)$**: The entire vector of action-values for all possible actions at state $s$.
* **$\|\cdot\|_\infty$ (Infinity Norm)**: Also known as the max norm. It outputs the single largest absolute value from the vector. For a Q-vector at state $s$, it equals $\max_a |Q(s,a)|$.

### Interpretation
- The forget term suppresses the largest-magnitude action value in forget states.
- The retain term preserves the largest change-sensitive component of the Q-vector in retain states.
- $\lambda$ is fixed to 1.

### Key property
This is the most faithful **state-level multi-environment** version in the comparison.

---

## Method 3: New_True_inf

**New_True_inf** keeps the same state-level multi-environment structure as `Ye_multi`, but modifies the penalty shape.  
Both terms are squared, and the retain term is weighted by a tunable lambda.

### Loss
$$L = \mathbb{E}_{s \sim S_u}[\|Q_{\pi'}(s)\|_\infty^2] + \lambda \mathbb{E}_{s \not\sim S_u}[\|Q_{\pi'}(s) - Q_{\pi}(s)\|_\infty^2]$$

### Notation
* **$\|\cdot\|_\infty^2$ (Squared Infinity Norm)**: Takes the maximum absolute value from the vector and squares it. Squaring applies a much harsher penalty to larger values or larger deviations than the standard infinity norm.
* **$\lambda$ (Lambda)**: A tunable weighting hyperparameter that controls the tradeoff between how strongly the model forgets versus how strongly it retains original knowledge.

### Interpretation
- The forget term penalizes large-magnitude Q-values more aggressively than `Ye_multi`.
- The retain term also penalizes large deviations more aggressively.
- $\lambda$ controls the tradeoff between forgetting and retention.

### Key property
This is the **principled modified** version of Ye's multi-environment loss.

---

## Method 4: New_Max

**New_Max** is the current implementation-style variant.  
It matches `New_True_inf` on the retain side, but its forget term uses $\max(Q)$ instead of the true infinity norm $\max(|Q|)$.

### Loss
$$L = \mathbb{E}_{s \sim S_u}[(\max_a Q_{\pi'}(s,a))^2] + \lambda \mathbb{E}_{s \not\sim S_u}[\|Q_{\pi'}(s) - Q_{\pi}(s)\|_\infty^2]$$

### Notation
* **$\max_a Q_{\pi'}(s,a)$**: The maximum signed Q-value across all possible actions ($a$) in state $s$. Unlike the infinity norm (which looks at absolute magnitude), this specifically targets the most *positive* action value. 
* **$(\cdot)^2$**: Squares the maximum signed Q-value.

### Interpretation
- The forget term focuses on the most positive action value rather than the largest-magnitude action value.
- The retain term is the same squared infinity-norm retain penalty used in `New_True_inf`.
- $\lambda$ is tuned.

### Key property
This is the closest method to the current implementation baseline and tests whether using $\max(Q)$ instead of the true infinity norm materially changes results.