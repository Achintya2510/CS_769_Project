# Implementation Plan

## Reinforcement Learning for Sequential Evidence Selection in HybridQA

## Project goal

Test whether a PPO-trained sequential evidence selector improves multi-hop table-text question answering compared with:

1. Direct question answering.
2. Similarity-based evidence retrieval.
3. Supervised evidence selection.

This is a small experimental research project. The four “agents” in the project description are logical pipeline stages, not separate software agents or services. The implementation will use two notebooks and four readable Python files.

### Main scope

- Primary dataset: HybridQA.
- Secondary dataset: OTT-QA only if HybridQA is complete.
- Maximum evidence-selection depth: three actions.
- One shared evidence representation and candidate generator.
- One fixed answer model for all methods.
- Three random seeds for learned methods.
- Four PPO reward ablations.

### Not required

- UI, API, database, deployment, or authentication.
- General-purpose table processing.
- Separate agent frameworks.
- End-to-end LLM reinforcement learning.
- Full GPU training on the laptop.

---

## A. Architecture and data flow

```text
Question + table + linked passages
                 |
                 v
       Normalize and link evidence
                 |
                 v
       Shared candidate generator
                 |
      +----------+----------+
      |          |          |
 Similarity  Supervised    PPO
      |          |          |
      +----------+----------+
                 |
                 v
        Fixed answer generator
                 |
                 v
       Answer and evidence metrics
```

Similarity, supervised, and PPO selectors will use the same candidates, cached embeddings, three-step limit, answer generator, and evaluation code. This isolates the evidence-selection method.

Initial model choices:

| Component | Choice |
|---|---|
| Evidence encoder | Frozen `sentence-transformers/all-MiniLM-L6-v2` |
| Answer generator | Frozen `allenai/unifiedqa-t5-base` |
| Actor-Critic | Small GRU plus MLP scoring/value heads |
| PPO initialization | Matching supervised-selector checkpoint |
| Seeds | 13, 42, 2026 |

---

## B. Dataset and preprocessing

Use the [official HybridQA repository](https://github.com/wenhuchen/HybridQA) and [WikiTables-WithLinks](https://github.com/wenhuchen/WikiTables-WithLinks).

Required files:

```text
HybridQA/released_data/train.json
HybridQA/released_data/dev.json
HybridQA/released_data/test.json
HybridQA/released_data/train.traced.json
HybridQA/released_data/dev.traced.json
HybridQA/released_data/dev_reference.json
WikiTables-WithLinks/tables_tok/<table_id>.json
WikiTables-WithLinks/request_tok/<table_id>.json
```

The traced `answer-node` values are approximate answer locations, not complete verified reasoning chains. They may be used as weak training labels but must not be presented as perfect gold evidence.

### Data split

1. Split official training data into 80% `train_core` and 20% `train_val`.
2. Group by `table_id` to avoid placing the same table in both splits.
3. Use `train_val` for all design choices and checkpoint selection.
4. Keep official `dev` untouched until the experiment is fixed.
5. Use official test only through CodaLab if available.

### Data audit

Before training, check:

- exact example and table counts;
- duplicate question IDs;
- missing tables or passages;
- malformed rows, cells, or hyperlinks;
- trace coverage and invalid trace coordinates;
- train-validation-development overlap.

Save the result as `outputs/data_audit.json`. Preserve original text, IDs, row numbers, and links in processed data.

---

## C. Evidence and candidate generation

Use two evidence types.

### Table row

```text
ID: row::<table_id>::<row_number>
Text: table title | header: value | header: value | ...
```

### Linked passage

```text
ID: passage::<table_id>::<wikipedia_id>
Text: passage text from request_tok
```

Table hyperlinks provide the row-to-passage links. Do not add fuzzy or LLM-generated links in the main experiment.

### Weak evidence labels

- Table trace: `{traced_row}`.
- Passage trace: `{row_containing_link, passage_id}`.
- Preserve multiple trace locations as alternative evidence sets.

If the audit shows that these labels are too incomplete for evidence evaluation, manually validate a small stratified development subset. Use weak traces only for training and clearly label weak-trace metrics as auxiliary.

### Candidate procedure

1. Rank rows by question-row cosine similarity.
2. At step 0, offer the top 12 rows. `STOP` is masked.
3. After selecting a row, add up to 20 passages linked from that row.
4. Keep remaining top rows available.
5. Mask previously selected evidence.
6. End on `STOP` or after three selections.

Candidate generation must not access answers or evidence labels during evaluation.

Before training, measure evidence-chain reachability. If it is below 85% on `train_val`, adjust candidate limits before freezing the design.

---

## D. Supervised model and PPO

### State and actions

- **State:** question embedding, selected-evidence history, candidate embeddings, action mask, and step number.
- **Action:** one candidate or `STOP`.
- **Transition:** add selected evidence, expand linked candidates, remove repeats, and increment the step.
- **End:** `STOP` or three selections.

### Actor-Critic

- MiniLM produces frozen 384-dimensional embeddings.
- A small GRU summarizes selected evidence.
- The actor MLP scores each candidate.
- The critic MLP estimates the state value.
- Invalid actions receive negative-infinity logits before sampling.

### Supervised training

Train the actor with weak evidence sets. Since evidence order is not known, allow any remaining valid evidence item as the next supervised target. Train `STOP` after a valid set is collected.

For each seed, save the best supervised actor. Use the same checkpoint as the supervised baseline and PPO starting policy.

### PPO rewards

```text
answer_reward = 0.5 * EM + 0.5 * answer_F1
evidence_reward_t = evidence_F1(S_t) - evidence_F1(S_t-1)
step_cost = 0.02 per evidence action
```

Run these configurations:

| PPO run | Reward |
|---|---|
| A | Answer only |
| B | Evidence only |
| C | Answer + evidence |
| D | Answer + evidence − step cost |

PPO D is the primary method; A–C are ablations.

Initial PPO settings:

- rollout: 256 episodes;
- `gamma=0.99`, GAE lambda `0.95`;
- clip ratio `0.20`;
- four update epochs;
- learning rate `3e-4`;
- entropy coefficient `0.01`;
- value coefficient `0.5`;
- gradient clipping `0.5`;
- maximum 50,000 episodes with `train_val` early stopping.

Cache answer-model rewards using question ID, ordered evidence IDs, and model settings.

Required sanity checks:

- invalid/repeated actions cannot be sampled;
- `STOP` and maximum depth terminate correctly;
- hand-computed GAE and PPO losses match the code;
- checkpoint resume restores training state;
- PPO learns a tiny synthetic two-hop task.

---

## E. Baselines and answer generation

### Direct QA

Pass table rows and linked passages in deterministic source order until the answer-model token limit is reached.

### Similarity retrieval

Use the shared candidate generator and select the highest-similarity evidence for up to three steps.

### Supervised selector

Use the trained actor without PPO. Select greedily until `STOP` or three actions.

### PPO selector

Load the supervised actor and fine-tune it with PPO.

### Fixed answer settings

All methods use:

- `allenai/unifiedqa-t5-base` at a pinned revision;
- input format `question \n context`;
- 512 input tokens and 32 output tokens;
- greedy decoding without sampling;
- identical context formatting and answer normalization.

Before main training, compare answer performance using gold, random, and empty evidence. If gold evidence does not clearly help, change the answer model before freezing the experiment.

---

## F. Evaluation

### Metrics

- Answer: official HybridQA EM and token F1.
- Evidence: precision, recall, F1, and complete-chain recall.
- Efficiency: selected item count, context tokens, steps, latency, and GPU memory.
- PPO behavior: reward, entropy, approximate KL, value loss, and chain length.

### Statistical reporting

- Run supervised and PPO A–D with seeds 13, 42, and 2026.
- Report mean and standard deviation.
- Use paired bootstrap confidence intervals for PPO-versus-supervised differences.
- Do not claim improvement if the confidence interval includes zero.

Save predictions and evidence chains before calculating aggregate metrics. Never remove failed questions from evaluation.

Final table:

| Method | Answer EM | Answer F1 | Evidence F1 | Complete-chain recall | Avg. evidence | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Direct QA | TBD | TBD | N/A | N/A | N/A | TBD |
| Similarity | TBD | TBD | TBD | TBD | TBD | TBD |
| Supervised | TBD | TBD | TBD | TBD | TBD | TBD |
| PPO D | TBD | TBD | TBD | TBD | TBD | TBD |

Do not insert predicted or invented results.

---

## G. Project files

```text
CS_769_Project/
|-- README.md
|-- IMPLEMENTATION_PLAN.md
|-- requirements.txt
|-- config.yaml
|-- local_workflow.ipynb
|-- colab_gpu.ipynb
|-- data_pipeline.py
|-- models.py
|-- training.py
|-- evaluation.py
|-- data/
`-- outputs/
```

Responsibilities:

- `local_workflow.ipynb`: data audit, preprocessing, sanity checks, final metrics, plots, and local inference.
- `colab_gpu.ipynb`: only GPU tasks—full embeddings, neural training, answer generation, and GPU timing.
- `data_pipeline.py`: loading, preprocessing, evidence, and candidates.
- `models.py`: selector, Actor-Critic, and fixed answerer wrapper.
- `training.py`: supervised/PPO training, environment, rewards, and checkpoints.
- `evaluation.py`: baselines, metrics, statistics, and plots.

Do not add a package hierarchy, CLI framework, agent classes, or separate files for small helpers.

---

## H. Implementation steps

### Step 1 — Setup

Create the two notebooks, four Python files, `config.yaml`, and `requirements.txt`. Confirm imports and device detection locally and in Colab.

### Step 2 — Data pipeline

Download and audit HybridQA locally. Create the table-grouped split, processed records, evidence IDs, weak labels, and row-passage links.

### Step 3 — Candidates and metrics

Implement candidate generation, official answer scoring, evidence metrics, and small CPU sanity checks. Use Colab to create the full embedding cache, then check candidate reachability.

### Step 4 — Baselines

Prepare direct and similarity contexts locally. Run batched answer generation in Colab. Evaluate on `train_val`; do not use official dev yet.

### Step 5 — Supervised selector

Implement locally, verify it can overfit a tiny sample, then train three seeds in Colab. Save resumable and inference checkpoints.

### Step 6 — PPO

Verify environment, masking, rewards, GAE, and PPO locally on a synthetic task. Run one small Colab pilot, freeze settings, then train A–D for three seeds.

### Step 7 — Final evaluation

Run all final answer generation and GPU timing in Colab on identical official-dev IDs. Download predictions, evidence chains, weights, and timing data.

### Step 8 — Analysis

Locally calculate metrics, confidence intervals, tables, plots, and error analysis. State whether PPO helped, hurt, or was inconclusive.

OTT-QA is optional and begins only after Step 8.

---

## I. Compute requirements

Approximate Colab requirements:

| Task | Estimated time | GPU memory |
|---|---:|---:|
| Full MiniLM embeddings | 0.5–2 hours | 2–4 GB |
| Answer generation | 1–4 hours per full split | 3–6 GB |
| Supervised training | 1–3 hours per seed | 2–4 GB |
| PPO training | 3–8 hours per run | 4–8 GB |

The full four-ablation, three-seed PPO matrix may require 36–96 GPU hours. Replace these estimates with measured pilot values. Reserve about 40 GB in Google Drive for data, caches, checkpoints, and predictions.

The laptop handles CPU preprocessing, small sanity checks, metrics, plots, and local selector inference. It does not repeat supervised or PPO training.

---

## J. Colab-to-local handoff

### Colab saves two checkpoint types

1. **Resume checkpoint:** actor, critic, optimizer, progress, random states, and config. Keep this in Google Drive for interrupted Colab sessions.
2. **Inference bundle:** only the files required to use the trained selector locally.

```text
outputs/final_bundle/
|-- policy_state.pt
|-- policy_config.yaml
|-- evidence_embeddings.npy
|-- evidence_index.json
|-- preprocessing_manifest.json
|-- environment.txt
|-- manifest.json
`-- README.txt
```

`policy_state.pt` contains the actor and history-GRU state dictionary only. Load it in `models.py` using CPU mapping and `weights_only=True`.

The bundle records the exact MiniLM and UnifiedQA revisions. Their pretrained weights can be downloaded normally on the laptop. If offline execution is required, save their Hugging Face folders in the bundle once.

### Colab workflow

1. Clone the exact Git commit.
2. Open `colab_gpu.ipynb` and mount Google Drive.
3. Install `requirements.txt`.
4. Load processed files created locally.
5. Run only embedding, training, answer-generation, and GPU-evaluation cells.
6. Save checkpoints every 2,000 episodes and at validation points.
7. Export the final inference bundle and download predictions/logs.

### Final local run

The **Run downloaded model** section of `local_workflow.ipynb` will:

1. verify bundle checksums;
2. load the config, evidence index, and final policy on CPU;
3. select evidence without calling the training loop;
4. optionally run a few UnifiedQA examples on CPU;
5. use saved Colab predictions for full-development analysis.

Therefore, no high-GPU training file needs to run in VS Code.

---

## K. Main risks

| Risk | Action |
|---|---|
| Weak traces are incomplete | Use them for training only; validate a small evidence subset if needed |
| Candidates miss required evidence | Adjust shared limits on `train_val` and report candidate recall |
| Answerer ignores evidence | Replace it before main experiments |
| PPO is unstable | Use supervised initialization, reward ablations, KL/entropy monitoring, and resume checkpoints |
| Colab disconnects | Save frequent checkpoints to Drive |
| GPU memory is insufficient | Reduce batch size; do not change evidence budgets between methods |
| Experiment matrix is too slow | Apply the same early-stopping rule to every PPO run |
| Official test is unavailable | Use official dev as the clearly labeled held-out evaluation |

---

## Completion checklist

- [ ] Data audit and table-grouped splits are saved.
- [ ] Direct and similarity baselines run end to end.
- [ ] Three supervised seeds are complete.
- [ ] PPO A–D are complete for three seeds.
- [ ] Official-dev predictions and evidence chains are saved.
- [ ] Final policy bundle loads on the laptop without training.
- [ ] Metrics, confidence intervals, plots, and error analysis are generated.
- [ ] The report presents the observed result, including a negative or inconclusive result.

## Primary sources

- [HybridQA repository](https://github.com/wenhuchen/HybridQA)
- [WikiTables-WithLinks](https://github.com/wenhuchen/WikiTables-WithLinks)
- [HybridQA paper](https://aclanthology.org/2020.findings-emnlp.91/)
- [OTT-QA repository](https://github.com/wenhuchen/OTT-QA)
