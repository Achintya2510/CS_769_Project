# PPO-Based Sequential Evidence Selection for HybridQA

This repository investigates whether reinforcement learning can improve
multi-hop question answering by learning **which evidence to select before a
fixed answer model is invoked**. The central experiment compares direct
question answering, cosine-similarity retrieval, supervised evidence selection,
and a PPO-trained Actor-Critic selector under a shared HybridQA pipeline.

The project does not fine-tune the answer generator with reinforcement learning.
Instead, PPO operates on explicit intermediate actions: select a table row,
select a linked passage, or stop. This isolates the contribution of sequential
evidence selection from changes to the language model.

## Research question and contribution

> Can a PPO-trained policy improve downstream HybridQA answer quality by
> selecting evidence sequentially, compared with heuristic retrieval and an
> otherwise equivalent supervised selector?

Recent table-reasoning work commonly applies RL to generated reasoning traces,
multimodal table understanding, or executable SQL. This project studies a
different control point: a lightweight policy acts over discrete rows and
passages before answer generation. The claim is deliberately scoped to the
three selected related works and to this controlled experiment; it is not a
claim that this is the first evidence-selection MDP in the broader literature.

The experiment produced a statistically significant improvement in answer
quality over the supervised selector, while also revealing a trade-off: the
answer-oriented PPO policy aligned less closely with the automatically derived
gold evidence sets.

## System overview

```text
Question + Wikipedia table + linked passages
                     |
                     v
        Shared candidate generator
        (rows + passages + STOP)
                     |
         +-----------+-----------+
         |           |           |
    Similarity   Supervised     PPO
      policy       actor      Actor-Critic
         |           |           |
         +-----------+-----------+
                     |
        Ordered evidence chain (<= 3 items)
                     |
                     v
          Frozen UnifiedQA-T5-base
                     |
                     v
       Answer EM/F1 + evidence metrics
```

All learned selectors use the same candidate generator, cached representations,
maximum number of actions, answer model, and evaluation code. Therefore, the
primary supervised-versus-PPO comparison changes the optimization method while
holding the rest of the pipeline fixed.

## Data and evidence representation

The primary dataset is
[HybridQA](https://github.com/wenhuchen/HybridQA), paired with
[WikiTables-WithLinks](https://github.com/wenhuchen/WikiTables-WithLinks).
HybridQA questions require reasoning across structured table content and linked
Wikipedia passages.

The preprocessing pipeline:

1. normalizes questions, tables, rows, and linked passages;
2. assigns stable evidence identifiers to every row and passage;
3. constructs a table-grouped training/validation split to avoid table leakage;
4. derives weak alternative evidence sets from the released traces and links;
5. encodes each question and evidence item once using frozen
   `sentence-transformers/all-MiniLM-L6-v2` embeddings; and
6. stores table data and embeddings in reusable caches.

Candidate generation first ranks rows by question-row cosine similarity, keeps
the top 12 rows, expands passages linked from selected rows, and retains up to 20
passage candidates. The policy may select at most three evidence items.

## Sequential decision process

Evidence selection is represented as a finite-horizon Markov Decision Process.
At step $t$, the state is

$$
s_t = (q, E_t, C_t, t),
$$

where $q$ is the question embedding, $E_t=(e_1,\ldots,e_t)$ is the ordered
evidence history, and $C_t$ is the dynamically generated candidate set. The
action space is

$$
a_t \in C_t \cup \{\text{STOP}\}.
$$

Selecting evidence appends it to the history and updates the linked candidates.
Selecting `STOP`, exhausting the candidates, or reaching three selections ends
the episode.

### Actor-Critic model

The selected-evidence sequence is summarized by a GRU:

$$
h_t = \operatorname{GRU}(e_1,\ldots,e_t;	anh(W_q q)).
$$

For candidate $c_i$, the actor receives

$$
x_{t,i} = [q;h_t;c_i;q\odot c_i;h_t\odot c_i;
           \tau_i;\operatorname{sim}(q,c_i);t/T],
$$

where $\tau_i$ is a learned row/passage/STOP type embedding. A masked MLP
produces action logits and the policy

$$
\pi_\theta(a_i\mid s_t)
= \operatorname{softmax}(f_\theta(x_{t,i})).
$$

The critic estimates $V_\phi(s_t)$ from the question, GRU history, pooled
candidate representation, and normalized step index.

### Supervised initialization

The actor is first trained from weak evidence sets. Because valid evidence order
is not uniquely known, every remaining member of a valid evidence set is
accepted as a correct next action. If \(Y_t\) denotes the valid actions,

$$
\mathcal{L}_{\mathrm{sup}}
= -\log\sum_{a\in Y_t}\pi_\theta(a\mid s_t).
$$

`STOP` becomes the target after a complete valid evidence set has been selected.
PPO starts from the matching supervised checkpoint for each random seed.

## Reward design

Let $\Phi(E_t)$ be the best set-F1 between the selected evidence and any weak
gold alternative. Evidence reward is potential-based shaping:

$$
r_t^{\mathrm{evidence}} = \Phi(E_{t+1})-\Phi(E_t).
$$

At termination, the fixed answer model produces prediction $\hat y$. The
answer reward is

$$
r^{\mathrm{answer}}
= \tfrac{1}{2}\operatorname{EM}(\hat y,y)
+ \tfrac{1}{2}F_1(\hat y,y).
$$

The complete reward is

$$
r_t = \alpha r_t^{\mathrm{answer}}
    + \beta r_t^{\mathrm{evidence}}
    - \lambda\,\mathbf{1}[a_t\neq\text{STOP}].
$$

Four controlled PPO variants were trained:

| Variant | $\alpha$ | $\beta$ | $\lambda$ |
|---|---:|---:|---:|
| Evidence only | 0 | 1 | 0 |
| Answer only | 1 | 0 | 0 |
| Combined | 1 | 1 | 0 |
| Combined + step cost | 1 | 1 | 0.02 |

Answer rewards are cached by `(answer model, question ID, ordered evidence IDs)`
so resumed runs do not repeatedly execute the 892 MB answer model for previously
seen terminal states.

## PPO objective

Generalized Advantage Estimation is used:

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),\qquad
\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda_{\mathrm{GAE}})^l\delta_{t+l}.
$$

With probability ratio

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
     {\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)},
$$

the clipped policy objective is

$$
L^{\mathrm{clip}}(\theta)=
\mathbb{E}_t\left[
\min\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right)\right].
$$

The optimized loss combines the negative policy objective, value regression,
and entropy regularization:

$$
\mathcal{L}
=-L^{\mathrm{clip}}
+c_v\,\mathbb{E}[(V_\phi(s_t)-\hat R_t)^2]
-c_H\,\mathbb{E}[H(\pi_\theta(\cdot\mid s_t))].
$$

## Training configuration

| Setting | Value |
|---|---:|
| Evidence encoder | `all-MiniLM-L6-v2`, frozen |
| Answer generator | `allenai/unifiedqa-t5-base`, frozen |
| Embedding dimension | 384 |
| GRU/hidden dimension | 256 |
| Maximum evidence selections | 3 |
| Supervised epochs | 3 |
| Supervised batch size | 32 |
| Supervised learning rate | $3\times10^{-4}$ |
| PPO learning rate | $3\times10^{-4}$ |
| Rollout episodes/update | 256 |
| PPO minibatch size | 128 |
| Update epochs | 4 |
| Discount $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| PPO clip $\epsilon$ | 0.20 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| Target approximate KL | 0.03 |
| Maximum episodes | 50,000 |
| Early-stopping patience | 5 validation checks |
| Seeds | 13, 42, 2026 |

Checkpoints are written every 2,000 configured episodes and contain the model,
optimizer, progress, logs, and random-number-generator states. Training resumes
from `latest.pt`; completed or early-stopped runs are skipped.

### Observed PPO stopping points

All 12 runs stopped after validation failed to improve for the configured
patience, rather than being forced to consume all 50,000 episodes.

| Reward | Seed 13 | Seed 42 | Seed 2026 |
|---|---:|---:|---:|
| Evidence only | 28,160 | 12,032 | 12,032 |
| Answer only | 20,224 | 36,096 | 26,112 |
| Combined | 20,224 | 12,032 | 20,224 |
| Combined + step cost | 12,032 | 12,032 | 22,016 |

## Evaluation protocol

Final answer generation used the 3,466 questions in the official HybridQA
development split. Every method used the same frozen answer generator and answer
normalization. Learned methods were evaluated for all three seeds.

Reported metrics are:

- answer Exact Match and token F1;
- evidence precision, recall, F1, and complete-chain recall;
- average selected-evidence count and latency; and
- paired bootstrap confidence intervals over aligned questions.

Evidence labels are weak alternatives derived from released traces and links.
Consequently, Evidence F1 measures annotation alignment, not every possible
semantically sufficient reasoning path.

## Results

Method-level values below are means across three seeds for learned systems.

| Method | Answer EM | Answer F1 | Evidence F1 | Avg. evidence |
|---|---:|---:|---:|---:|
| Direct QA | 3.32% | 8.26% | N/A | 49.20 |
| Similarity retrieval | 6.35% | 11.31% | 26.16% | 3.00 |
| Supervised selector | 16.07% | 21.45% | **36.85%** | **1.47** |
| PPO evidence only | 16.79% | 22.14% | 35.29% | 1.66 |
| PPO answer only | **17.46%** | **23.17%** | 32.50% | 2.25 |
| PPO combined | 17.16% | 22.71% | 34.95% | 1.89 |
| PPO combined + step cost | 16.77% | 22.43% | 35.41% | 1.71 |

The best individual checkpoint was `ppo_answer_only_seed_42/best.pt`:

| Answer EM | Answer F1 | Evidence F1 | Avg. evidence |
|---:|---:|---:|---:|
| 18.41% | **24.52%** | 32.72% | 2.43 |

### Statistical comparison with supervised selection

Paired bootstrap tests used 10,000 samples. Answer comparisons contain all 3,466
questions; evidence comparisons contain the 3,374 questions with finite paired
weak-evidence metrics.

| PPO variant | Answer-F1 difference | 95% CI | Evidence-F1 difference | 95% CI |
|---|---:|---:|---:|---:|
| Evidence only | +0.69 pp | [+0.11, +1.28] | -1.56 pp | [-2.14, -0.98] |
| Answer only | **+1.72 pp** | **[+0.96, +2.48]** | -4.36 pp | [-5.06, -3.62] |
| Combined | +1.26 pp | [+0.58, +1.96] | -1.90 pp | [-2.55, -1.26] |
| Combined + step cost | +0.99 pp | [+0.38, +1.61] | -1.44 pp | [-2.04, -0.83] |

Every PPO variant produced a statistically positive Answer-F1 difference, while
every PPO variant produced lower weak-label Evidence F1 than the supervised
selector.

## Plots and interpretation

![Final method comparison](outputs/analysis/method_comparison.png)

The answer-quality panel shows that all PPO objectives outperform the supervised
mean, with answer-only reward producing the highest downstream F1. The evidence
panel reverses the ordering: supervised selection best matches the derived gold
evidence. The efficiency panel shows why Direct QA performs poorly—it passes an
average of about 49 evidence items, whereas learned selectors pass roughly one
to two.

![Answer/evidence trade-off](outputs/analysis/answer_evidence_tradeoff.png)

The trade-off plot demonstrates that optimizing answer reward and optimizing
weak evidence alignment are not equivalent. PPO answer-only selected more items
than supervised selection and obtained slightly higher evidence recall, but its
additional selections lowered precision and Evidence F1. Some selections may
also be useful to UnifiedQA without matching the automatically derived evidence
IDs.

![Selected training diagnostics](outputs/analysis/selected_training_diagnostics.png)

The selected seed-42 answer-only run early-stopped at 36,096 episodes. Rollout
reward remained noisy, as expected for sampled questions and terminal answer
rewards. The validation objective peaked before the final checkpoint, while the
approximate KL was usually near the configured target with occasional spikes.
The retained `best.pt`, not the final policy state, is used for selection and
evaluation.

### Main interpretation

The supported conclusion is:

> Within a controlled architecture and fixed answer-generation setup, PPO
> significantly improves final answer accuracy over supervised evidence
> selection. The improvement does not correspond to stronger weak-label
> evidence alignment; it exposes a measurable answer/evidence trade-off.

This is not evidence that the system reproduces or outperforms specialised
HybridQA architectures. Results from systems using different encoders, readers,
retrieval budgets, or training objectives are contextual references rather than
direct baselines.

## Reproducing the experiment

### 1. Local CPU environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the official repositories at:

```text
data/raw/HybridQA
data/raw/WikiTables-WithLinks
```

Open `local_workflow.ipynb`, enable the audit/preprocessing cells when starting
from raw data, and produce `data/processed`.

### 2. Complete GPU workflow

Open `colab_gpu.ipynb`. The merged notebook contains all stages:

1. repository checkout and dependency installation;
2. Kaggle/Colab path and persistent-storage setup;
3. artifact verification;
4. fresh embedding generation;
5. supervised training for three seeds;
6. resumable PPO training;
7. official-dev answer generation;
8. inference-bundle export; and
9. visible result aggregation.

Expensive operations are guarded by `RUN_*` switches. Enable one stage at a
time for a fresh reproduction. For the existing asset-based Kaggle workflow,
keep embedding and supervised stages disabled, attach the private asset dataset,
and enable PPO or final evaluation as needed.

Kaggle inputs are read-only. Checkpoints are therefore restored into
`/kaggle/working/cs769_outputs`. Saving a notebook version persists that
directory as downloadable notebook output and allows it to be attached to the
next run for resumption.

### 3. Local final analysis

After downloading and extracting `cs769_outputs` and the evaluation archive into
`outputs/`, execute `local_workflow.ipynb`. It verifies all artifacts, creates
method and run-level tables, performs paired bootstrap analysis, selects the
final checkpoint, and writes figures and CSV files under `outputs/analysis/`.

## Repository structure

```text
CS_769_Project/
|-- README.md
|-- IMPLEMENTATION_PLAN.md
|-- config.yaml
|-- requirements.txt
|-- data_pipeline.py          # audit, preprocessing, weak evidence, candidates
|-- models.py                 # frozen encoder/answerer and Actor-Critic network
|-- training.py               # environment, supervised learning, PPO, checkpoints
|-- evaluation.py             # baselines, QA/evidence metrics, bootstrap, plots
|-- colab_gpu.ipynb           # complete reproducible GPU workflow
|-- local_workflow.ipynb      # executed CPU analysis with embedded outputs
|-- data/processed/           # generated processed dataset
`-- outputs/
    |-- colab_runs/           # supervised and PPO checkpoints
    |-- results/              # 17 prediction JSONL files and metrics
    `-- analysis/             # final tables, plots, and selected-model record
```

## Limitations

- Evidence supervision is derived from weak traces and links, not a manually
  validated complete set of all valid reasoning paths.
- Only HybridQA was completed; the optional OTT-QA generalization experiment was
  outside the final scope.
- Candidate pre-filtering can make a required evidence item unreachable.
- UnifiedQA is frozen, so absolute answer performance is constrained by its
  ability to use the selected context.
- The reward model uses the same fixed answer generator later used for
  evaluation; the result therefore measures optimization for this particular
  downstream model.
- Three seeds quantify some training variance but do not exhaust PPO
  hyperparameter uncertainty.

## Related work

- Zheyuan Yang, Lyuhao Chen, Arman Cohan, and Yilun Zhao.
  [Table-R1: Inference-Time Scaling for Table Reasoning Tasks](https://aclanthology.org/2025.emnlp-main.1040/),
  EMNLP 2025.
- Xiaoqiang Kang et al.
  [Can GRPO Boost Complex Multimodal Table Understanding?](https://aclanthology.org/2025.emnlp-main.637/),
  EMNLP 2025.
- Josefa Lia Stoisser, Marc Boubnovski Martell, and Julien Fauqueur.
  [Sparks of Tabular Reasoning via Text2SQL Reinforcement Learning](https://aclanthology.org/2025.trl-1.20/),
  TRL 2025.
- Wenhu Chen et al.
  [HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data](https://aclanthology.org/2020.findings-emnlp.91/),
  Findings of EMNLP 2020.
