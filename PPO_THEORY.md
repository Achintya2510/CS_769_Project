# Theoretical Foundation and Optimization Details

This note explains exactly what the PPO component learns, the mathematical
objective it optimizes, how one update is formed, and why the configuration
parameters matter. It describes the implemented evidence selector; it does not
describe reinforcement learning of the answer-generation model.

## 1. What is being optimized?

The answer generator, `allenai/unifiedqa-t5-base`, is frozen. The evidence text
encoder, `sentence-transformers/all-MiniLM-L6-v2`, is also frozen after it
produces cached representations. PPO updates the trainable Actor-Critic selector
initialized from the supervised selector. The selector learns a distribution
over the next evidence item (or stopping) from the question and the evidence
already selected.

For policy parameters \(\theta\), the central objective is the expected
discounted return under the selection policy:

$$
J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[
  \sum_{t=0}^{T-1} \gamma^t r_t
\right],
$$

where a trajectory \(\tau\) is a sequence of question-specific evidence
selections and rewards, \(T\leq 3\) is the episode length, and \(\gamma\) is
the reward discount. PPO is a practical, conservative way to improve this
expected-return objective from sampled trajectories.

The distinction is important: the policy is trained to choose evidence that
earns the configured reward. It is not directly trained to maximize answer F1
on every evaluation question, and the frozen T5 parameters are not changed by
PPO.


## 2. The Evidence-Selection MDP

Each HybridQA question defines a finite-horizon Markov Decision Process (MDP),

$$
(\mathcal{S}, \mathcal{A}, P, r, \gamma).
$$

- **State:**

$$
s_t = (q, E_t, C_t, t),
$$

where $q$ is the question representation, $E_t = (e_1, \ldots, e_t)$ is the ordered history of selected evidence items, $C_t$ is the current candidate evidence set, and $t$ denotes the selection depth.

- **Action:**

$$
a_t \in C_t \cup \{\mathrm{STOP}\}.
$$

An action corresponds to selecting either a table row or a linked passage from the current candidate set. Invalid or previously selected actions are masked from the action distribution.

- **Transition:**

When an evidence item is selected, it is appended to the selection history:

$$
E_{t+1} = (e_1, \ldots, e_t, e_{t+1}).
$$

The candidate set is then updated according to the table structure and linked passages. The episode terminates when the agent selects $\mathrm{STOP}$, when no valid candidates remain, or when three evidence items have been selected.

- **Reward:**

The reward is defined as a configurable combination of the terminal answer reward, evidence-quality improvement, and, in one ablation setting, a cost associated with each evidence selection:

$$
r_t
=
r_t^{\mathrm{answer}}
+
\lambda_{\mathrm{ev}} r_t^{\mathrm{evidence}}
-
\lambda_{\mathrm{cost}} r_t^{\mathrm{cost}}.
$$

The weighting coefficients depend on the experimental configuration.

- **Horizon:**

The evidence-selection process has a maximum horizon of three selections:

$$
T \leq 3.
$$

The candidate-generation mechanism is shared across the similarity-based, supervised, and PPO selectors. Therefore, the PPO policy learns to choose among the candidates provided by the retrieval stage; it does not directly search over every table row or passage in the HybridQA corpus.

## 3. State Representation and Actor-Critic Policy

The question embedding initializes a GRU-based history representation. Let $z_q$ denote the question embedding and $z_{e_i}$ the embedding of the $i$-th selected evidence item. The history representation is updated as

$$
h_0 = \tanh(W_q z_q),
\qquad
h_t = \mathrm{GRU}(z_{e_t}, h_{t-1}).
$$

For each candidate $c_i$, the actor constructs an input representation by combining the question embedding, history representation, candidate embedding, elementwise interactions, candidate type, question-candidate similarity, and normalized selection step:

$$
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
$$

Here, $\tau_i$ represents the candidate-type feature, $\mathrm{sim}(q,c_i)$ denotes the similarity between the question and candidate, and $H$ is the maximum selection horizon.

An MLP scores each valid candidate and the $\mathrm{STOP}$ action. After invalid actions are masked, a softmax is applied to obtain the final action probability distribution.

$$
\pi_\theta(a_i\mid s_t)=
\frac{\exp(f_\theta(x_{t,i}))}
{\sum_{j\in\mathcal{A}(s_t)}\exp(f_\theta(x_{t,j}))}.
$$

The critic estimates the state value \(V_\phi(s_t)\), the expected future
discounted return from that state. Actor and critic share the state/history
construction but have separate scoring heads. The value estimate is a training
signal for lower-variance policy-gradient updates; it is not used as the final
answer score.

## 4. Supervised initialization objective

Before PPO, the selector is trained using weak evidence sets. Since annotations
can provide several acceptable evidence items without specifying a unique
order, let \(Y_t\) be all currently available next actions that belong to at
least one valid evidence set. The supervised action loss is the negative log
probability assigned to the set of valid next actions:

$$
\mathcal{L}_{\mathrm{sup}}(\theta)
=-\log\left(\sum_{a\in Y_t}\pi_\theta(a\mid s_t)\right).
$$

After a complete valid set is collected, `STOP` is supervised. Each seed's
supervised checkpoint initializes the PPO run with the matching seed. This
reduces the burden of learning useful evidence actions from an initially
uninformed policy, but also means the PPO result is a supervised-initialized
policy rather than a from-scratch RL result.

## 5. Reward objective and four ablations

### 5.1 Answer reward

At a terminal state, the selected evidence is given to the fixed answer model.
For normalized answer exact match \(\mathrm{EM}\) and token-level answer F1,
the answer reward is

$$
R_{\mathrm{answer}}=
\tfrac{1}{2}\mathrm{EM}(\hat y,y)
+\tfrac{1}{2}F_1(\hat y,y),
$$

where \(y\) is the reference and \(\hat y\) is the generated answer. It lies
between zero and one. The answer model's output is cached for repeated
question/evidence sequences to avoid duplicate inference during rollouts.

### 5.2 Evidence reward

Let \(\Phi(E_t)\) be the maximum evidence set-F1 between the selected set and
any weak gold alternative. On a non-STOP selection the implementation gives
the change in this score:

$$
r_{t,\mathrm{evidence}}=\Phi(E_{t+1})-\Phi(E_t).
$$

The reward can be positive for a useful item, zero if it does not change the
best match, and negative if adding an item reduces precision enough to lower
set-F1. The STOP action itself receives no evidence increment.

### 5.3 Combined per-step reward

The transition reward is

$$
r_t=\alpha R_{\mathrm{answer},t}
+\beta r_{t,\mathrm{evidence}}
-\lambda\,\mathbf{1}[a_t\ne\text{STOP}],
$$

where the answer term is nonzero at episode termination, and the evidence term
is nonzero on evidence-selection transitions. The step cost is charged for
selecting an evidence item, not for stopping.

| Reward variant | \(\alpha\) | \(\beta\) | \(\lambda\) | What the ablation tests |
|---|---:|---:|---:|---|
| `answer_only` | 1 | 0 | 0 | Whether downstream answer reward alone can improve selection |
| `evidence_only` | 0 | 1 | 0 | Whether weak evidence labels can train the sequential policy |
| `combined` | 1 | 1 | 0 | Whether answer and evidence feedback complement one another |
| `combined_step` | 1 | 1 | 0.02 | Whether a small per-selection cost reduces unnecessary selections |

The reward variants are controlled ablations of the training signal. They do
not change the evaluation metric or the answer model.

### 5.4 A shaping nuance

The implemented evidence reward is a difference in potential, but the classic
policy-invariant potential-based shaping term for discount \(\gamma\) is

$$
F(s_t,s_{t+1})=\gamma\Phi(s_{t+1})-\Phi(s_t).
$$

The code uses \(\Phi(s_{t+1})-\Phi(s_t)\), while the PPO return discounts by
\(\gamma=0.99\). Consequently, it should be described as **incremental evidence
F1 reward** or **potential-difference reward**, not claimed to preserve the
optimal policy by the standard shaping theorem. With undiscounted returns, the
increments telescope to the terminal potential change; under discounted
returns, they need not. This distinction matters when interpreting
`evidence_only` and the evidence component of combined reward.

## 6. PPO update: from sampled rollout to parameter update

For each update, the current policy samples up to 256 question episodes and
stores states, actions, old action log-probabilities, rewards, terminal flags,
and value estimates. The collected rollout is then reused for several
optimization epochs.

### 6.1 Temporal-difference residual and GAE

For a nonterminal transition, the one-step TD residual is

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
$$

At termination the next-state value is zero. Generalized Advantage Estimation
(GAE) combines TD residuals at multiple horizons:

$$
\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda_{\mathrm{GAE}})^l\delta_{t+l},
\qquad
\hat R_t=\hat A_t+V_\phi(s_t).
$$

The parameter \(\lambda_{\mathrm{GAE}}\) controls the bias-variance trade-off:
lower values rely more on short-horizon TD estimates; values nearer one use
longer reward traces and usually lower bias but higher variance. Here it is
0.95.

### 6.2 Clipped policy surrogate

The probability ratio between updated and rollout policies is

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

PPO maximizes the clipped surrogate

$$
L^{\mathrm{clip}}(\theta)=\mathbb{E}_t\left[
\min\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)\right].
$$

Clipping limits how much the new policy can benefit from a large probability
change on one sampled action. It is a surrogate trust-region constraint, not a
hard guarantee that every policy distribution changes by at most
\(\epsilon\).

### 6.3 Value loss and entropy bonus

The critic is fit to the rollout return target with squared error:

$$
L_V(\phi)=\mathbb{E}_t[(V_\phi(s_t)-\hat R_t)^2].
$$

Policy entropy is

$$
H(\pi_\theta(\cdot\mid s_t))
=-\sum_a\pi_\theta(a\mid s_t)\log\pi_\theta(a\mid s_t).
$$

Entropy rewards a less concentrated policy and helps retain exploration. The
minimized joint loss is

$$
\mathcal{L}(\theta,\phi)
=-L^{\mathrm{clip}}(\theta)
+c_V L_V(\phi)
-c_H\mathbb{E}_t[H(\pi_\theta(\cdot\mid s_t))].
$$

The negative sign on the entropy term means minimizing the loss encourages
larger entropy. Gradients are clipped by global norm before the optimizer step.

## 7. Optimization sequence in this experiment

1. **Prepare representations:** frozen MiniLM encodes each evidence item and
   question once. Embeddings and ID-to-row indexes are cached.
2. **Train supervised selector:** for each configured seed, optimize the
   multi-action weak-label likelihood for up to three epochs. Validation
   evidence F1 selects `best_model.pt`.
3. **Initialize PPO:** build the Actor-Critic and load that seed's supervised
   actor weights. The critic is initialized by the model construction and
   learned through PPO returns.
4. **Collect rollouts:** sample actions from the current policy for batches of
   question episodes, calculate the selected reward variant, and query the
   frozen answer model only when the reward requires it.
5. **Estimate returns:** compute TD residuals, GAE advantages, and value
   targets from the sampled rollout.
6. **Optimize several epochs:** shuffle rollout transitions, form minibatches,
   compute clipped policy/value/entropy losses, clip gradient norm, and update
   with Adam.
7. **Stop an update early if needed:** target approximate KL limits excessive
   drift from the rollout policy during optimization epochs.
8. **Validate and checkpoint:** periodically evaluate greedy policy reward on
   500 validation questions, retain the best validation checkpoint, save the
   latest resumable state, and early-stop after five checks without improvement.
9. **Evaluate once training is complete:** load each PPO best checkpoint and
   evaluate the same official development questions with the fixed answerer.

PPO optimizes a surrogate computed from on-policy samples; it does not
differentiate through the text generator or through discrete evidence
transitions. Answer feedback reaches the selector as scalar rewards.

## 8. Parameter values and their roles

### PPO and reward parameters

| Parameter | Value | Role and practical effect |
|---|---:|---|
| PPO learning rate | `3e-4` | Adam step size for Actor-Critic parameters. Too large can destabilize updates; too small can slow policy improvement. |
| Rollout episodes | `256` | Number of question episodes collected per update. Larger rollouts improve reward estimates but cost more model/environment work and memory. |
| PPO minibatch size | `128` | Number of transitions used for one gradient estimate. Smaller batches add noisy updates; larger batches are steadier but give fewer updates per rollout epoch. |
| Update epochs | `4` | Number of passes over a rollout. More passes reuse samples more but increase overfitting to stale data and policy drift. |
| Discount \(\gamma\) | `0.99` | Weights later reward relative to earlier reward. Near one is appropriate for a short episode where terminal answer reward matters. |
| GAE \(\lambda_{\mathrm{GAE}}\) | `0.95` | Controls the horizon/bias-variance balance in advantage estimates. |
| Clip \(\epsilon\) | `0.20` | Limits incentives for large action-probability changes in the surrogate. |
| Value coefficient \(c_V\) | `0.5` | Balances critic regression against policy improvement. Too high can let value fitting dominate; too low weakens the baseline. |
| Entropy coefficient \(c_H\) | `0.01` | Encourages action exploration. A larger value delays policy concentration; too large can prevent decisive selection. |
| Max gradient norm | `0.5` | Clips the joint gradient to reduce damaging steps from unusually large gradients. |
| Target approximate KL | `0.03` | Stops further PPO epochs for an update when policy drift is too large. Complements ratio clipping. |
| Maximum episodes | `50,000` | Hard upper limit per run; it is not a target that every run must reach. |
| Validation subset | `500` questions | Keeps periodic greedy validation tractable while providing a consistent stopping signal. |
| Early-stopping patience | `5` checks | Stops a run after five validation checks without a new best score. |
| Checkpoint interval | `2,000` episodes | Bounds lost work after interruption and preserves resumability. |
| Seeds | `13, 42, 2026` | Repeats each learned method under different random initialization/sampling for a limited estimate of variability. |

### Model and task parameters

| Parameter | Value | Role |
|---|---:|---|
| Evidence/question embedding size | `384` | Frozen MiniLM vector dimension consumed by the selector. |
| Actor/Critic hidden size | `256` | Capacity of the learned scoring/value MLPs. |
| History model | GRU, size `384` | Encodes the order and content of prior selected evidence. |
| Candidate rows | up to `12` | Limits row action candidates after similarity pre-filtering. |
| Candidate passages | up to `20` | Limits linked passage candidates. |
| Selection horizon | `3` | Maximum number of evidence items per answer. |
| Supervised epochs | `3` | Training passes for the seed-matched PPO initialization. |
| Supervised batch size | `32` | Number of episode losses accumulated before a supervised optimizer update. |
| Supervised learning rate | `3e-4` | AdamW step size for weak-label selector initialization. |
| Combined-step cost \(\lambda\) | `0.02` | Small penalty for each selected item; intended to test shorter chains. |

Parameter values are experimental choices, not universal optima. The experiment
uses four reward ablations and three seeds, but does not conduct a broad
hyperparameter sweep. Therefore, conclusions are conditional on this frozen
configuration.

## 9. Interpreting the ablation outcomes

The answer-only objective achieved the highest mean Answer F1 among tested PPO
variants and a significant improvement over supervised selection. It selected
more items on average and had lower weak-label Evidence F1 than supervised
selection. This is consistent with optimizing the downstream answer reward:
the selector may include evidence useful to the frozen generator even when that
evidence does not match the weak gold IDs.

The step-cost variant selected fewer items than answer-only and the two
combined variants, which is consistent with penalizing every evidence action.
However, shorter chains alone do not establish higher efficiency at equal
answer quality; answer score, evidence recall/precision, token count, and
latency must be interpreted together.

The evidence-only objective optimizes weak annotation alignment. Since the
evidence labels are incomplete/noisy and the implemented increment is not the
discount-corrected shaping term, its return is not identical to answer
correctness or to complete-chain success.

## 10. Limits of the theoretical claim

- PPO's clipped surrogate is a practical local update objective, not a proof of
  global convergence or a strict trust region.
- The critic and policy are trained from finite, correlated trajectories;
  seed means and question-level bootstrap intervals quantify only part of the
  uncertainty.
- The discounted incremental evidence reward is not policy-invariant shaping
  at \(\gamma<1\), as explained above.
- Evidence F1 is computed against weak trace-derived alternatives. Low score
  can reflect label mismatch as well as genuinely irrelevant selections.
- The selector is optimized for one fixed answer model. A different generator
  can prefer different evidence.
- The 12 PPO runs were allowed to early-stop using validation reward. Their
  episode counts differ; performance comparisons use the common held-out dev
  evaluation, not final training episode reward.

## References

- Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Schulman et al. (2016), [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
- Ng, Harada, and Russell (1999), [Policy Invariance Under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
- Sutton and Barto (2018), [Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/the-book-2nd.html).
