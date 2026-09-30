# Sequential Evidence Selection for HybridQA

Small research implementation for comparing direct QA, similarity retrieval,
supervised evidence selection, and PPO evidence selection on HybridQA.

## Files

- `local_workflow.ipynb`: CPU preprocessing, inspection, sanity checks, analysis,
  and loading the final downloaded policy.
- `colab_gpu.ipynb`: GPU-only embedding, training, answer generation, and timing.
- `data_pipeline.py`: dataset loading, auditing, preprocessing, and candidates.
- `models.py`: evidence encoder, answerer, and Actor-Critic model.
- `training.py`: supervised and PPO training utilities and checkpoints.
- `evaluation.py`: baselines, metrics, bootstrap analysis, and plots.

## Local setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place or clone the official repositories at:

```text
data/raw/HybridQA
data/raw/WikiTables-WithLinks
```

Then open `local_workflow.ipynb` and run the data audit and preprocessing cells.
Do not run full supervised or PPO training locally.

## Colab

Open `colab_gpu.ipynb`, mount Google Drive, clone the same Git commit, install
`requirements.txt`, and point `config.yaml` at the processed data and output
directories. Colab saves resumable checkpoints and a lightweight
`outputs/final_bundle/` for local CPU evidence selection.

See `IMPLEMENTATION_PLAN.md` for the controlled experiment design.
