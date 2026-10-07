# Theoretical Foundation and Optimization Details

This note explains exactly what the PPO component learns, the mathematical objective it optimizes, how one update is formed, and why the configuration parameters matter. It describes the implemented evidence selector; it does not describe reinforcement learning of the answer-generation model.

## 1. What Is Being Optimized?

The answer generator, `allenai/unifiedqa-t5-base`, is frozen. The evidence text encoder, `sentence-transformers/all-MiniLM-L6-v2`, is also frozen after it produces cached representations. PPO updates the trainable Actor-Critic selector initialized from the supervised selector.

The selector learns a distribution over the next evidence item, or stopping, based on the question and the evidence already selected.

For policy parameters $\theta$, the central objective is the expected discounted return under the selection policy:

```math
J(\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_{t=0}^{T-1} \gamma^t r_t
\right].
```

Here, a trajectory $\tau$ is a sequence of question-specific evidence selections and rewards, $T \leq 3$ is the episode length, and $\gamma$ is the reward discount factor.

PPO provides a practical, conservative way to improve this expected-return objective using sampled trajectories.

The distinction is important: the policy is trained to choose evidence that earns the configured reward. It is not directly trained to maximize answer F1 on every evaluation question, and the frozen T5 parameters are not changed by PPO.

## 2. The Evidence-Selection MDP

Each HybridQA question defines a finite-horizon Markov Decision Process (MDP):

```math
(\mathcal{S}, \mathcal{A}, P, r, \gamma).
```

- **State:**

```math
s_t = (q, E_t, C_t, t).
```

Here, $q$ is the question representation, $E_t = (e_1,\ldots,e_t)$ is the ordered history of selected evidence items, $C_t$ is the current candidate evidence set, and $t$ denotes the current selection depth.

- **Action:**

```math
a_t \in C_t \cup \{\mathrm{STOP}\}.
```

An action corresponds to selecting either a table row or a linked passage from the current candidate set, or choosing $\mathrm{STOP}$.

Invalid or previously selected actions are masked from the action distribution.

- **Transition:**

When an evidence item is selected, it is appended to the selection history:

```math
E_{t+1} = (e_1,\ldots,e_t,e_{t+1}).
```

The candidate set is then updated according to the table structure and linked passages.

The episode terminates when the agent selects $\mathrm{STOP}$, when no valid candidates remain, or when three evidence items have been selected.

- **Reward:**

The transition reward combines terminal answer quality, incremental evidence improvement, and, in the selection-cost ablation, a penalty for selecting additional evidence:

```math
r_t
=
\alpha R_{\mathrm{answer},t}
+
\beta r_{t,\mathrm{evidence}}
-
\lambda \mathbf{1}[a_t \ne \mathrm{STOP}].
```

The contribution of each component depends on the reward variant being evaluated. The answer component is nonzero at episode termination, while the evidence component is computed after evidence-selection actions.

The four reward configurations are described in Section 5.

- **Horizon:**

Let $H=3$ denote the maximum allowed selection horizon. The actual episode length $T$ therefore satisfies

```math
T \leq H = 3.
```

The candidate-generation mechanism is shared across the similarity-based, supervised, and PPO selectors. Therefore, the PPO policy learns to choose among the candidates provided by the retrieval stage; it does not directly search over every table row or passage in the HybridQA corpus.

## 3. State Representation and Actor-Critic Policy

The question embedding initializes a GRU-based history representation.

Let $z_q$ denote the question embedding and $z_{e_i}$ the embedding of the $i$-th selected evidence item.

The initial history representation is

```math
h_0 = \tanh(W_q z_q).
```

After evidence item $e_t$ is selected, the history representation is updated as

```math
h_t = \mathrm{GRU}(z_{e_t}, h_{t-1}).
```

Thus, $h_t$ provides a compact representation of the evidence-selection history.

For each candidate $c_i$, the actor constructs an input representation by combining the question embedding, history representation, candidate embedding, elementwise interactions, candidate type, question-candidate similarity, and normalized selection step:

```math
x_{t,i}
=
\left[
z_q;
h_t;
z_{c_i};
z_q \odot z_{c_i};
h_t \odot z_{c_i};
\tau_i;
\mathrm{sim}(q,c_i);
\frac{t}{H}
\right].
```

Here:

- $z_{c_i}$ is the candidate embedding.
- $z_q \odot z_{c_i}$ represents the elementwise interaction between the question and candidate.
- $h_t \odot z_{c_i}$ represents the interaction between the selection history and candidate.
- $\tau_i$ represents the candidate-type feature.
- $\mathrm{sim}(q,c_i)$ is the question-candidate similarity score.
- $t/H$ is the normalized selection depth.
- $H=3$ is the maximum selection horizon.

An MLP scores each valid candidate and the $\mathrm{STOP}$ action. Invalid actions are masked, and a softmax is applied over the valid action logits to obtain the policy distribution:

```math
\pi_\theta(a_i \mid s_t)
=
\frac{\exp(\ell_{t,i})}
{\sum_{a_j \in \mathcal{A}(s_t)} \exp(\ell_{t,j})}.
```

Here, $\ell_{t,i}$ denotes the policy logit assigned to valid action $a_i$.

The critic estimates the state value

```math
V_\phi(s_t),
```

which represents the expected future discounted return from state $s_t$.

The actor and critic share the state/history construction but have separate scoring heads. The value estimate is used as a training signal for lower-variance policy-gradient updates; it is not used as the final answer score.

## 4. Supervised Initialization Objective

Before PPO training, the selector is trained using weak evidence sets.

Because annotations may provide several acceptable evidence items without specifying a unique order, let $Y_t$ denote all currently available next actions that belong to at least one valid evidence set.

The supervised action loss is the negative log probability assigned to the set of valid next actions:

```math
\mathcal{L}_{\mathrm{sup}}(\theta)
=
-\log
\left(
\sum_{a \in Y_t}
\pi_\theta(a \mid s_t)
\right).
```

After a complete valid evidence set is collected, $\mathrm{STOP}$ is supervised.

Each seed's supervised checkpoint initializes the PPO run with the matching seed.

This reduces the burden of learning useful evidence actions from an initially uninformed policy. It also means that the resulting PPO policy is a supervised-initialized policy rather than a policy learned from scratch using reinforcement learning.

## 5. Reward Objective and Four Ablations

### 5.1 Answer Reward

At a terminal state, the selected evidence is given to the fixed answer-generation model.

For normalized answer exact match $\mathrm{EM}$ and token-level answer F1, the answer reward is

```math
R_{\mathrm{answer}}
=
\frac{1}{2}\mathrm{EM}(\hat{y},y)
+
\frac{1}{2}F_1(\hat{y},y).
```

Here, $y$ is the reference answer and $\hat{y}$ is the generated answer.

The answer reward lies between zero and one.

The answer model's output is cached for repeated question-evidence sequences to avoid duplicate inference during rollouts.

### 5.2 Evidence Reward

Let $\Phi(E_t)$ denote the maximum evidence set-F1 between the selected evidence set and any weak gold alternative.

For a non-STOP selection, the implementation assigns the change in this score:

```math
r_{t,\mathrm{evidence}}
=
\Phi(E_{t+1})
-
\Phi(E_t).
```

The reward can be positive when the newly selected item improves the best evidence match, zero when it does not change the match, or negative when adding the item reduces precision enough to lower set-F1.

The $\mathrm{STOP}$ action itself receives no evidence-improvement reward.

### 5.3 Combined Per-Step Reward

The transition reward is

```math
r_t
=
\alpha R_{\mathrm{answer},t}
+
\beta r_{t,\mathrm{evidence}}
-
\lambda \mathbf{1}[a_t \ne \mathrm{STOP}].
```

The answer term is nonzero at episode termination, while the evidence term is nonzero on evidence-selection transitions.

The selection cost is charged when an evidence item is selected, not when the agent stops.

| Reward variant | $\alpha$ | $\beta$ | $\lambda$ | What the ablation tests |
|---|---:|---:|---:|---|
| `answer_only` | 1 | 0 | 0 | Whether downstream answer reward alone can improve selection |
| `evidence_only` | 0 | 1 | 0 | Whether weak evidence labels can train the sequential policy |
| `combined` | 1 | 1 | 0 | Whether answer and evidence feedback complement one another |
| `combined_step` | 1 | 1 | 0.02 | Whether a small per-selection cost reduces unnecessary selections |

The reward variants are controlled ablations of the training signal. They do not change the evaluation metric or the answer model.

### 5.4 A Shaping Nuance

The implemented evidence reward is a difference in potential. However, the classic policy-invariant potential-based shaping term for discount $\gamma$ is

```math
F(s_t,s_{t+1})
=
\gamma \Phi(s_{t+1})
-
\Phi(s_t).
```

The implementation instead uses

```math
\Phi(s_{t+1})-\Phi(s_t),
```

while the PPO return uses $\gamma=0.99$.

Consequently, this should be described as an **incremental evidence-F1 reward** or **potential-difference reward**, rather than being claimed to preserve the optimal policy under the standard potential-based shaping theorem.

With undiscounted returns, the increments telescope to the terminal potential change. Under discounted returns, they need not.

This distinction matters when interpreting the `evidence_only` condition and the evidence component of the combined reward.

## 6. PPO Update: From Sampled Rollout to Parameter Update

For each PPO update, the current policy samples up to 256 question episodes and stores states, actions, old action log-probabilities, rewards, terminal flags, and value estimates.

The collected rollout is then reused for several optimization epochs.

### 6.1 Temporal-Difference Residual and GAE

For a nonterminal transition, the one-step temporal-difference residual is

```math
\delta_t
=
r_t
+
\gamma V_\phi(s_{t+1})
-
V_\phi(s_t).
```

At termination, the next-state value is zero.

Generalized Advantage Estimation (GAE) combines temporal-difference residuals over multiple horizons:

```math
\hat{A}_t
=
\sum_{l=0}^{T-t-1}
(\gamma \lambda_{\mathrm{GAE}})^l
\delta_{t+l}.
```

The corresponding return target is

```math
\hat{R}_t
=
\hat{A}_t
+
V_\phi(s_t).
```

The parameter $\lambda_{\mathrm{GAE}}$ controls the bias-variance trade-off.

Lower values rely more on short-horizon temporal-difference estimates. Values closer to one use longer reward traces and typically reduce bias at the cost of greater variance.

In this experiment,

```math
\lambda_{\mathrm{GAE}} = 0.95.
```

### 6.2 Clipped Policy Surrogate

The probability ratio between the updated policy and the rollout policy is

```math
\rho_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}.
```

PPO maximizes the clipped surrogate objective

```math
L^{\mathrm{clip}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)\hat{A}_t,
\mathrm{clip}
\left(
\rho_t(\theta),
1-\epsilon,
1+\epsilon
\right)
\hat{A}_t
\right)
\right].
```

Clipping limits how much the updated policy can benefit from a large probability change on one sampled action.

It acts as a surrogate trust-region constraint; it does not provide a hard guarantee that every policy probability changes by at most $\epsilon$.

### 6.3 Value Loss and Entropy Bonus

The critic is trained to predict the rollout return target using squared error:

```math
L_V(\phi)
=
\mathbb{E}_t
\left[
\left(
V_\phi(s_t)-\hat{R}_t
\right)^2
\right].
```

Policy entropy is

```math
\mathcal{H}
\left(
\pi_\theta(\cdot \mid s_t)
\right)
=
-
\sum_a
\pi_\theta(a \mid s_t)
\log
\pi_\theta(a \mid s_t).
```

Entropy encourages a less concentrated action distribution and helps preserve exploration.

The minimized joint loss is

```math
\mathcal{L}(\theta,\phi)
=
-
L^{\mathrm{clip}}(\theta)
+
c_V L_V(\phi)
-
c_H
\mathbb{E}_t
\left[
\mathcal{H}
\left(
\pi_\theta(\cdot \mid s_t)
\right)
\right].
```

The negative sign on the entropy term means that minimizing the loss encourages larger entropy.

Gradients are clipped by their global norm before the optimizer step.

## 7. Optimization Sequence in This Experiment

1. **Prepare representations:** Frozen MiniLM encodes each evidence item and question once. Embeddings and ID-to-row indexes are cached.

2. **Train the supervised selector:** For each configured seed, optimize the multi-action weak-label likelihood for up to three epochs. Validation evidence F1 selects `best_model.pt`.

3. **Initialize PPO:** Build the Actor-Critic model and load the matching seed's supervised actor weights. The critic is initialized during model construction and learned through PPO return targets.

4. **Collect rollouts:** Sample actions from the current policy for batches of question episodes, calculate the selected reward variant, and query the frozen answer model only when the reward requires it.

5. **Estimate returns:** Compute temporal-difference residuals, GAE advantages, and value targets from the sampled rollout.

6. **Optimize for several epochs:** Shuffle rollout transitions, form minibatches, compute clipped policy, value, and entropy losses, clip the gradient norm, and update the model using Adam.

7. **Stop an update early if needed:** A target approximate KL threshold limits excessive drift from the rollout policy during repeated optimization epochs.

8. **Validate and checkpoint:** Periodically evaluate greedy policy reward on 500 validation questions, retain the best validation checkpoint, save the latest resumable state, and early-stop after five validation checks without improvement.

9. **Evaluate after training:** Load each PPO run's best checkpoint and evaluate it on the same official development questions using the fixed answer-generation model.

PPO optimizes a surrogate objective computed from on-policy samples. It does not differentiate through the text generator or through the discrete evidence-selection transitions.

Answer feedback reaches the selector only as scalar rewards.

## 8. Parameter Values and Their Roles

### PPO and Reward Parameters

| Parameter | Value | Role and practical effect |
|---|---:|---|
| PPO learning rate | `3e-4` | Adam step size for Actor-Critic parameters. Too large can destabilize updates; too small can slow policy improvement. |
| Rollout episodes | `256` | Number of question episodes collected per update. Larger rollouts improve reward estimates but require more computation and memory. |
| PPO minibatch size | `128` | Number of transitions used for one gradient estimate. Smaller batches produce noisier updates; larger batches are steadier but give fewer updates per rollout epoch. |
| Update epochs | `4` | Number of passes over one rollout. More passes reuse samples more heavily but increase the risk of excessive policy drift. |
| Discount $\gamma$ | `0.99` | Controls how strongly later rewards contribute relative to earlier rewards. |
| GAE $\lambda_{\mathrm{GAE}}$ | `0.95` | Controls the bias-variance trade-off in advantage estimates. |
| PPO clip $\epsilon$ | `0.20` | Limits incentives for large action-probability changes in the surrogate objective. |
| Value coefficient $c_V$ | `0.5` | Balances critic regression against policy improvement. |
| Entropy coefficient $c_H$ | `0.01` | Encourages exploration and discourages premature policy concentration. |
| Maximum gradient norm | `0.5` | Clips the joint gradient norm to reduce destabilizing updates. |
| Target approximate KL | `0.03` | Stops further PPO epochs during an update when policy drift becomes too large. |
| Maximum episodes | `50,000` | Hard upper limit per run; it is not a target that every run must reach. |
| Validation subset | `500` questions | Keeps periodic greedy validation tractable while providing a consistent stopping signal. |
| Early-stopping patience | `5` checks | Stops a run after five validation checks without a new best score. |
| Checkpoint interval | `2,000` episodes | Limits lost work after interruption and enables training to resume. |
| Seeds | `13, 42, 2026` | Repeats each learned method under different random initialization and sampling conditions. |

### Model and Task Parameters

| Parameter | Value | Role |
|---|---:|---|
| Evidence/question embedding size | `384` | Frozen MiniLM vector dimension consumed by the selector. |
| Actor/Critic hidden size | `256` | Capacity of the learned scoring and value-estimation MLPs. |
| History model | GRU, size `384` | Encodes the order and content of previously selected evidence. |
| Candidate rows | up to `12` | Limits table-row action candidates after similarity-based pre-filtering. |
| Candidate passages | up to `20` | Limits linked-passage candidates. |
| Selection horizon $H$ | `3` | Maximum number of evidence items that may be selected. |
| Supervised epochs | `3` | Number of training passes for seed-matched supervised initialization. |
| Supervised batch size | `32` | Number of episode losses accumulated before a supervised optimizer update. |
| Supervised learning rate | `3e-4` | AdamW step size for weak-label selector initialization. |
| Combined-step cost $\lambda$ | `0.02` | Small penalty for every selected evidence item, used to test whether the policy learns shorter chains. |

These parameter values are experimental choices rather than universal optima.

The experiment uses four reward ablations and three random seeds, but it does not conduct a broad hyperparameter sweep. Therefore, conclusions are conditional on this fixed experimental configuration.

## 9. Interpreting the Ablation Outcomes

The `answer_only` objective achieved the highest mean Answer F1 among the tested PPO variants and a significant improvement over supervised selection.

It selected more evidence items on average and obtained lower weak-label Evidence F1 than the supervised selector.

This is consistent with optimizing downstream answer reward: the selector may include evidence that is useful to the frozen answer generator even when that evidence does not match the weak gold evidence IDs.

The `combined_step` variant selected fewer evidence items than `answer_only` and the two combined variants without a step cost. This behavior is consistent with penalizing each evidence-selection action.

However, shorter chains alone do not establish greater efficiency at equal answer quality. Answer score, evidence recall and precision, token count, and inference latency should be interpreted together.

The `evidence_only` objective optimizes alignment with the weak evidence annotations.

Because these evidence labels may be incomplete or noisy, and because the implemented evidence increment is not the discount-corrected potential-based shaping term, its return is not equivalent to answer correctness or complete-chain success.

## 10. Limits of the Theoretical Claim

- PPO's clipped surrogate is a practical local update objective, not a proof of global convergence or a strict trust region.

- The critic and policy are trained from finite, correlated trajectories. Seed means and question-level bootstrap intervals quantify only part of the uncertainty.

- The discounted incremental evidence reward is not policy-invariant potential-based shaping when $\gamma < 1$, as discussed in Section 5.4.

- Evidence F1 is computed against weak trace-derived alternatives. A low Evidence F1 score can reflect label mismatch as well as genuinely irrelevant evidence selection.

- The selector is optimized for one fixed answer model. A different generator may prefer different evidence.

- The 12 PPO runs are allowed to early-stop using validation reward. Their episode counts may therefore differ. Performance comparisons use the common held-out development evaluation rather than final training-episode reward.

## References

- Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Schulman et al. (2016), [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
- Ng, Harada, and Russell (1999), [Policy Invariance Under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
- Sutton and Barto (2018), [Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/the-book-2nd.html).
