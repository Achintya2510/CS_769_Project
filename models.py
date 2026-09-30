"""Neural components for evidence selection and fixed answer generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import json

import numpy as np
import torch
from torch import nn


class FrozenTextEncoder:
    """Small Sentence Transformers wrapper used to create reusable embeddings."""

    def __init__(self, model_name: str, revision: str | None = None, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        kwargs = {"device": device}
        if revision:
            kwargs["revision"] = revision
        self.model_name = model_name
        self.revision = revision
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode(
        self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = True
    ) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def encode_evidence(
        self, evidence_items: Iterable[Mapping[str, Any]], batch_size: int = 64
    ) -> tuple[list[str], np.ndarray]:
        items = list(evidence_items)
        ids = [str(item["evidence_id"]) for item in items]
        matrix = self.encode([str(item["text"]) for item in items], batch_size=batch_size)
        return ids, matrix


def create_embedding_cache(
    encoder: FrozenTextEncoder,
    processed_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 64,
) -> dict[str, int]:
    """Encode every unique evidence item and processed question once."""

    processed = Path(processed_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table_index = json.loads((processed / "table_index.json").read_text(encoding="utf-8"))
    evidence_items: list[dict[str, Any]] = []
    for filename in table_index.values():
        table = json.loads((processed / "tables" / filename).read_text(encoding="utf-8"))
        evidence_items.extend(table["rows"])
        evidence_items.extend(table["passages"])
    unique_evidence = {str(item["evidence_id"]): item for item in evidence_items}
    evidence_ids = sorted(unique_evidence)
    evidence_matrix = encoder.encode(
        [str(unique_evidence[item]["text"]) for item in evidence_ids], batch_size=batch_size
    )
    np.save(output / "evidence_embeddings.npy", evidence_matrix)
    (output / "evidence_index.json").write_text(
        json.dumps({item: index for index, item in enumerate(evidence_ids)}, indent=2),
        encoding="utf-8",
    )

    questions: dict[str, str] = {}
    for path in sorted((processed / "questions").glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    questions[str(row["question_id"])] = str(row["question"])
    question_ids = sorted(questions)
    question_matrix = encoder.encode([questions[item] for item in question_ids], batch_size=batch_size)
    np.save(output / "question_embeddings.npy", question_matrix)
    (output / "question_index.json").write_text(
        json.dumps({item: index for index, item in enumerate(question_ids)}, indent=2),
        encoding="utf-8",
    )
    return {"evidence": len(evidence_ids), "questions": len(question_ids)}


def load_embedding_cache(cache_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    root = Path(cache_dir)
    evidence_matrix = np.load(root / "evidence_embeddings.npy", mmap_mode="r")
    evidence_index = json.loads((root / "evidence_index.json").read_text(encoding="utf-8"))
    question_matrix = np.load(root / "question_embeddings.npy", mmap_mode="r")
    question_index = json.loads((root / "question_index.json").read_text(encoding="utf-8"))
    evidence = {key: evidence_matrix[index] for key, index in evidence_index.items()}
    questions = {key: question_matrix[index] for key, index in question_index.items()}
    return evidence, questions


class EvidenceActorCritic(nn.Module):
    """Masked variable-candidate Actor-Critic with an ordered evidence history."""

    def __init__(
        self,
        embedding_dim: int = 384,
        type_embedding_dim: int = 8,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.type_embedding_dim = type_embedding_dim
        self.hidden_dim = hidden_dim
        self.type_embedding = nn.Embedding(3, type_embedding_dim)  # row, passage, STOP
        self.question_to_history = nn.Linear(embedding_dim, embedding_dim)
        self.history_gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        actor_input = embedding_dim * 5 + type_embedding_dim + 2
        self.actor = nn.Sequential(
            nn.LayerNorm(actor_input),
            nn.Linear(actor_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        critic_input = embedding_dim * 3 + 1
        self.critic = nn.Sequential(
            nn.LayerNorm(critic_input),
            nn.Linear(critic_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.stop_embedding = nn.Parameter(torch.randn(embedding_dim) * 0.02)

    def history_vector(self, question: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        """Return one history vector; gradients flow through the complete selected sequence."""

        initial = torch.tanh(self.question_to_history(question)).view(1, 1, -1)
        if selected.numel() == 0:
            return initial.view(-1)
        sequence = selected.view(1, -1, self.embedding_dim)
        _, hidden = self.history_gru(sequence, initial.transpose(0, 1))
        return hidden.view(-1)

    def forward(
        self,
        question: torch.Tensor,
        selected: torch.Tensor,
        candidates: torch.Tensor,
        candidate_types: torch.Tensor,
        similarities: torch.Tensor,
        step_fraction: float | torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidates.ndim != 2 or candidates.shape[-1] != self.embedding_dim:
            raise ValueError("candidates must have shape [num_candidates, embedding_dim]")
        history = self.history_vector(question, selected)
        count = candidates.shape[0]
        q = question.view(1, -1).expand(count, -1)
        h = history.view(1, -1).expand(count, -1)
        type_features = self.type_embedding(candidate_types.long())
        similarity_column = similarities.float().view(-1, 1)
        step = torch.as_tensor(step_fraction, dtype=question.dtype, device=question.device)
        step_column = step.expand(count).view(-1, 1)
        features = torch.cat(
            [q, h, candidates, q * candidates, h * candidates, type_features, similarity_column, step_column],
            dim=-1,
        )
        logits = self.actor(features).squeeze(-1)
        logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)

        valid = action_mask.float().view(-1, 1)
        pooled = (candidates * valid).sum(0) / valid.sum().clamp_min(1.0)
        critic_features = torch.cat([question, history, pooled, step.view(1)], dim=0)
        value = self.critic(critic_features).squeeze(-1)
        return logits, value

    def inference_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("critic.")
        }


def build_actor_critic(config: Mapping[str, Any]) -> EvidenceActorCritic:
    settings = config["models"]
    return EvidenceActorCritic(
        embedding_dim=int(settings["embedding_dim"]),
        type_embedding_dim=int(settings["type_embedding_dim"]),
        hidden_dim=int(settings["hidden_dim"]),
    )


def load_policy_weights(
    model: EvidenceActorCritic, path: str | Path, device: str | torch.device = "cpu"
) -> EvidenceActorCritic:
    state = torch.load(Path(path), map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    critic_missing = {name for name in missing if name.startswith("critic.")}
    if set(missing) != critic_missing or unexpected:
        raise RuntimeError(f"Policy checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    return model


class FixedAnswerer:
    """Frozen UnifiedQA wrapper. It is shared by all retrieval methods."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str = "cpu",
        max_input_tokens: int = 512,
        max_output_tokens: int = 32,
    ) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        kwargs = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        self.model.to(device).eval()
        self.device = torch.device(device)
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

    @torch.inference_mode()
    def answer(self, questions: Sequence[str], contexts: Sequence[str], batch_size: int = 16) -> list[str]:
        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must have the same length")
        predictions: list[str] = []
        prompts = [f"{question}\n{context}" for question, context in zip(questions, contexts)]
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                return_tensors="pt",
            ).to(self.device)
            generated = self.model.generate(
                **tokens,
                max_new_tokens=self.max_output_tokens,
                do_sample=False,
                num_beams=1,
            )
            predictions.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return predictions


def render_context(evidence_items: Sequence[Mapping[str, Any]]) -> str:
    labels = {"row": "ROW", "passage": "PASSAGE"}
    return "\n".join(
        f"[{labels.get(str(item.get('type')), 'EVIDENCE')} {item['evidence_id']}] {item['text']}"
        for item in evidence_items
    )
