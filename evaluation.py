"""Baselines, HybridQA metrics, statistical analysis, and result plots."""

from __future__ import annotations

import json
import math
import re
import string
import time
import atexit
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from data_pipeline import CandidateGenerator, load_processed_table
from models import EvidenceActorCritic, FixedAnswerer, render_context
from training import STOP_ID, SequentialEvidenceEnv, make_policy_state, policy_forward, set_f1


def normalize_answer(text: str) -> str:
    """Match the normalization used by the official HybridQA evaluator."""

    lowered = text.lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def answer_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def answer_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    overlap = Counter(predicted_tokens) & Counter(reference_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(predicted_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def evidence_metrics(
    selected: Sequence[str], alternatives: Sequence[Sequence[str]]
) -> dict[str, float]:
    if not alternatives:
        return {"precision": math.nan, "recall": math.nan, "f1": math.nan, "complete": math.nan}
    selected_set = set(selected)
    scored = []
    for gold in alternatives:
        gold_set = set(gold)
        overlap = len(selected_set & gold_set)
        precision = overlap / len(selected_set) if selected_set else 0.0
        recall = overlap / len(gold_set) if gold_set else 0.0
        f1 = set_f1(selected, gold)
        scored.append((f1, precision, recall, float(gold_set.issubset(selected_set))))
    best = max(scored, key=lambda item: item[0])
    return {"f1": best[0], "precision": best[1], "recall": best[2], "complete": max(x[3] for x in scored)}


def score_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No predictions to score")
    em = [answer_exact_match(str(row["prediction"]), str(row["answer"])) for row in rows]
    f1 = [answer_f1(str(row["prediction"]), str(row["answer"])) for row in rows]
    evidence = [evidence_metrics(row.get("selected_ids", []), row.get("gold_sets", [])) for row in rows]

    def nanmean(values: Sequence[float]) -> float:
        array = np.asarray(values, dtype="float64")
        return float(np.nanmean(array)) if not np.isnan(array).all() else math.nan

    return {
        "count": float(len(rows)),
        "answer_em": float(np.mean(em)),
        "answer_f1": float(np.mean(f1)),
        "evidence_precision": nanmean([x["precision"] for x in evidence]),
        "evidence_recall": nanmean([x["recall"] for x in evidence]),
        "evidence_f1": nanmean([x["f1"] for x in evidence]),
        "complete_chain_recall": nanmean([x["complete"] for x in evidence]),
        "average_evidence_count": float(np.mean([len(row.get("selected_ids", [])) for row in rows])),
        "average_latency_ms": float(np.mean([float(row.get("latency_ms", 0.0)) for row in rows])),
    }


def paired_bootstrap_difference(
    first: Sequence[float],
    second: Sequence[float],
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    a, b = np.asarray(first, dtype="float64"), np.asarray(second, dtype="float64")
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Paired inputs must be one-dimensional and have equal length")
    generator = np.random.default_rng(seed)
    differences = np.empty(samples, dtype="float64")
    for index in range(samples):
        sampled = generator.integers(0, len(a), size=len(a))
        differences[index] = np.mean(a[sampled] - b[sampled])
    return {
        "difference": float(np.mean(a - b)),
        "ci_low": float(np.percentile(differences, 2.5)),
        "ci_high": float(np.percentile(differences, 97.5)),
    }


def direct_context(table: Mapping[str, Any]) -> tuple[list[str], str]:
    items = list(table["rows"]) + list(table["passages"])
    return [str(item["evidence_id"]) for item in items], render_context(items)


def similarity_select(
    generator: CandidateGenerator,
    question_embedding: np.ndarray,
    max_steps: int = 3,
) -> list[str]:
    selected: list[str] = []
    for _ in range(max_steps):
        candidates = generator.candidates(question_embedding, selected)
        if not candidates:
            break
        selected.append(candidates[0])
    return selected


def similarity_select_environment(env: SequentialEvidenceEnv) -> list[str]:
    """Greedy similarity policy using the exact dynamic environment candidates."""

    env.reset()
    while not env.done:
        candidates = [item for item in env.candidate_ids() if item != STOP_ID]
        if not candidates:
            break
        # CandidateGenerator returns candidates in descending state-query similarity.
        env.step(candidates[0])
    return list(env.selected)


@torch.inference_mode()
def policy_select(
    model: EvidenceActorCritic,
    env: SequentialEvidenceEnv,
    device: torch.device | str = "cpu",
) -> list[str]:
    device = torch.device(device)
    model.to(device).eval()
    env.reset()
    while not env.done:
        state = make_policy_state(env).to(device)
        logits, _ = policy_forward(model, state)
        action_index = int(torch.argmax(logits).item())
        action_id = state.candidate_ids[action_index]
        env.step(action_id)
    return list(env.selected)


def make_answer_inputs(
    examples: Sequence[Mapping[str, Any]],
    selected_by_question: Mapping[str, Sequence[str]],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    inputs = []
    for example in examples:
        question_id = str(example["question_id"])
        selected_ids = list(selected_by_question[question_id])
        items = [evidence_by_id[item] for item in selected_ids]
        inputs.append(
            {
                "question_id": question_id,
                "question": example["question"],
                "answer": example.get("answer", ""),
                "selected_ids": selected_ids,
                "gold_sets": example.get("weak_evidence_sets", []),
                "context": render_context(items),
            }
        )
    return inputs


def prepare_direct_inputs(
    examples: Sequence[Mapping[str, Any]], processed_dir: str | Path
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        table = load_processed_table(processed_dir, str(example["table_id"]))
        selected_ids, context = direct_context(table)
        rows.append(
            {
                "question_id": str(example["question_id"]),
                "question": str(example["question"]),
                "answer": str(example.get("answer", "")),
                "selected_ids": selected_ids,
                "gold_sets": [],  # direct QA has no explicit selector
                "context": context,
            }
        )
    return rows


def prepare_similarity_inputs(
    examples: Sequence[Mapping[str, Any]],
    env_factory: Callable[[str], SequentialEvidenceEnv],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = {
        str(example["question_id"]): similarity_select_environment(
            env_factory(str(example["question_id"]))
        )
        for example in examples
    }
    return make_answer_inputs(examples, selected, evidence_by_id)


def prepare_policy_inputs(
    model: EvidenceActorCritic,
    examples: Sequence[Mapping[str, Any]],
    env_factory: Callable[[str], SequentialEvidenceEnv],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    device: torch.device | str,
) -> list[dict[str, Any]]:
    selected = {
        str(example["question_id"]): policy_select(
            model, env_factory(str(example["question_id"])), device
        )
        for example in examples
    }
    return make_answer_inputs(examples, selected, evidence_by_id)


def generate_answers(
    answerer: FixedAnswerer,
    inputs: Sequence[Mapping[str, Any]],
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    questions = [str(row["question"]) for row in inputs]
    contexts = [str(row["context"]) for row in inputs]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    predictions = answerer.answer(questions, contexts, batch_size=batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    per_item = elapsed_ms / max(1, len(inputs))
    return [dict(row, prediction=prediction, latency_ms=per_item) for row, prediction in zip(inputs, predictions)]


class AnswerRewardCache:
    """Persistent cache for terminal answer rewards during PPO rollouts."""

    def __init__(
        self,
        path: str | Path,
        answerer: FixedAnswerer,
        questions: Mapping[str, Mapping[str, Any]],
        evidence_by_id: Mapping[str, Mapping[str, Any]],
        model_key: str,
    ) -> None:
        self.path = Path(path)
        self.answerer = answerer
        self.questions = questions
        self.evidence_by_id = evidence_by_id
        self.model_key = model_key
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.pending = 0
        atexit.register(self.flush)

    def __call__(self, question_id: str, ordered_evidence_ids: Sequence[str]) -> float:
        key = json.dumps([self.model_key, question_id, list(ordered_evidence_ids)], separators=(",", ":"))
        if key in self.cache:
            return float(self.cache[key]["reward"])
        example = self.questions[question_id]
        items = [self.evidence_by_id[item] for item in ordered_evidence_ids]
        prediction = self.answerer.answer([example["question"]], [render_context(items)], batch_size=1)[0]
        em = answer_exact_match(prediction, example["answer"])
        f1 = answer_f1(prediction, example["answer"])
        reward = 0.5 * em + 0.5 * f1
        self.cache[key] = {"prediction": prediction, "em": em, "f1": f1, "reward": reward}
        self.pending += 1
        if self.pending >= 100:
            self.flush()
        return reward

    def flush(self) -> None:
        if self.pending == 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)
        self.pending = 0


def save_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def aggregate_result_directory(result_dir: str | Path) -> dict[str, Any]:
    """Score all saved prediction files and summarize repeated seeds."""

    root = Path(result_dir)
    per_run: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, float]]] = {}
    for path in sorted(root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        metrics = score_predictions(rows)
        per_run[path.stem] = metrics
        method = re.sub(r"_seed_\d+$", "", path.stem)
        grouped.setdefault(method, []).append(metrics)
    summary: dict[str, dict[str, float]] = {}
    for method, runs in grouped.items():
        names = sorted(runs[0])
        method_summary: dict[str, float] = {}
        for name in names:
            values = np.asarray([run[name] for run in runs], dtype="float64")
            method_summary[f"{name}_mean"] = float(np.nanmean(values))
            method_summary[f"{name}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        summary[method] = method_summary
    output = {"per_run": per_run, "summary": summary}
    (root / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def plot_training_log(rows: Sequence[Mapping[str, float]], output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        name
        for name in ("reward", "total", "answer", "evidence", "entropy", "approximate_kl")
        if any(name in row for row in rows)
    ]
    if not metrics:
        raise ValueError("No recognized training metrics to plot")
    figure, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.5 * len(metrics)), squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        x = [index for index, row in enumerate(rows) if metric in row]
        y = [float(row[metric]) for row in rows if metric in row]
        axis.plot(x, y)
        axis.set_title(metric)
        axis.set_xlabel("update")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
