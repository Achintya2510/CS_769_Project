"""Sequential environment, supervised learning, PPO, and checkpoint utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from data_pipeline import CandidateGenerator, load_processed_table
from models import EvidenceActorCritic


STOP_ID = "__STOP__"


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_f1(selected: Sequence[str], gold: Sequence[str]) -> float:
    selected_set, gold_set = set(selected), set(gold)
    if not selected_set and not gold_set:
        return 1.0
    if not selected_set or not gold_set:
        return 0.0
    overlap = len(selected_set & gold_set)
    precision = overlap / len(selected_set)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def best_evidence_f1(selected: Sequence[str], alternatives: Sequence[Sequence[str]]) -> float:
    if not alternatives:
        return 0.0
    return max(set_f1(selected, gold) for gold in alternatives)


class SequentialEvidenceEnv:
    """Finite-horizon evidence environment with dynamic candidates."""

    def __init__(
        self,
        question_id: str,
        question_embedding: np.ndarray,
        evidence_embeddings: Mapping[str, np.ndarray],
        evidence_types: Mapping[str, str],
        candidate_fn: Callable[[Sequence[str]], Sequence[str]],
        gold_sets: Sequence[Sequence[str]] | None = None,
        answer_reward_fn: Callable[[str, Sequence[str]], float] | None = None,
        max_steps: int = 3,
        answer_weight: float = 1.0,
        evidence_weight: float = 1.0,
        step_cost: float = 0.02,
    ) -> None:
        self.question_id = question_id
        self.question_embedding = np.asarray(question_embedding, dtype="float32")
        self.evidence_embeddings = evidence_embeddings
        self.evidence_types = evidence_types
        self.candidate_fn = candidate_fn
        self.gold_sets = [list(items) for items in (gold_sets or [])]
        self.answer_reward_fn = answer_reward_fn
        self.max_steps = max_steps
        self.answer_weight = answer_weight
        self.evidence_weight = evidence_weight
        self.step_cost = step_cost
        self.selected: list[str] = []
        self.done = False

    def reset(self) -> list[str]:
        self.selected = []
        self.done = False
        return self.candidate_ids()

    def candidate_ids(self) -> list[str]:
        candidates = [item for item in self.candidate_fn(self.selected) if item not in self.selected]
        if self.selected:
            candidates.append(STOP_ID)
        return candidates

    def step(self, action_id: str) -> tuple[list[str], float, bool, dict[str, float]]:
        if self.done:
            raise RuntimeError("Cannot step a terminated episode")
        valid = self.candidate_ids()
        if action_id not in valid:
            raise ValueError(f"Invalid action {action_id!r}")

        previous_phi = best_evidence_f1(self.selected, self.gold_sets)
        stopped = action_id == STOP_ID
        if not stopped:
            self.selected.append(action_id)
        self.done = stopped or len(self.selected) >= self.max_steps
        new_phi = best_evidence_f1(self.selected, self.gold_sets)
        evidence_reward = new_phi - previous_phi if not stopped else 0.0
        answer_reward = 0.0
        if self.done and self.answer_reward_fn is not None:
            answer_reward = float(self.answer_reward_fn(self.question_id, tuple(self.selected)))
        cost = 0.0 if stopped else self.step_cost
        reward = self.answer_weight * answer_reward + self.evidence_weight * evidence_reward - cost
        components = {
            "answer": answer_reward,
            "evidence": evidence_reward,
            "step_cost": cost,
            "total": reward,
        }
        return ([] if self.done else self.candidate_ids()), reward, self.done, components


def build_environment_factory(
    examples: Sequence[Mapping[str, Any]],
    processed_dir: str | Path,
    evidence_embeddings: Mapping[str, np.ndarray],
    question_embeddings: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    answer_reward_fn: Callable[[str, Sequence[str]], float] | None = None,
    reward_variant: str = "combined_step",
) -> tuple[Callable[[str], SequentialEvidenceEnv], dict[str, Mapping[str, Any]]]:
    """Create environments from processed records and cached embeddings."""

    examples_by_id = {str(row["question_id"]): dict(row) for row in examples}
    table_cache: dict[str, dict[str, Any]] = {}
    generator_cache: dict[str, CandidateGenerator] = {}
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for table_id in sorted({str(row["table_id"]) for row in examples}):
        table = load_processed_table(processed_dir, table_id)
        table_cache[table_id] = table
        items = list(table["rows"]) + list(table["passages"])
        evidence_by_id.update({str(item["evidence_id"]): item for item in items})
        table_embeddings = {
            item["evidence_id"]: evidence_embeddings[item["evidence_id"]]
            for item in items
            if item["evidence_id"] in evidence_embeddings
        }
        generator_cache[table_id] = CandidateGenerator(
            table,
            table_embeddings,
            top_rows=int(config["candidates"]["top_rows"]),
            top_passages=int(config["candidates"]["top_passages"]),
        )

    reward_settings = {
        "answer_only": (1.0, 0.0, 0.0),
        "evidence_only": (0.0, 1.0, 0.0),
        "combined": (1.0, 1.0, 0.0),
        "combined_step": (1.0, 1.0, float(config["rewards"]["step_cost"])),
    }
    if reward_variant not in reward_settings:
        raise ValueError(f"Unknown reward variant: {reward_variant}")
    answer_weight, evidence_weight, step_cost = reward_settings[reward_variant]

    def factory(question_id: str) -> SequentialEvidenceEnv:
        example = examples_by_id[question_id]
        table_id = str(example["table_id"])
        generator = generator_cache[table_id]
        # Use the current table's items directly. Scanning the global evidence
        # dictionary for every episode makes training unnecessarily quadratic.
        types = {
            str(item_id): str(item["type"])
            for item_id, item in generator.items.items()
        }
        return SequentialEvidenceEnv(
            question_id=question_id,
            question_embedding=question_embeddings[question_id],
            evidence_embeddings=generator.embeddings,
            evidence_types=types,
            candidate_fn=lambda selected: generator.candidates(question_embeddings[question_id], selected),
            gold_sets=example.get("weak_evidence_sets", []),
            answer_reward_fn=answer_reward_fn,
            max_steps=int(config["candidates"]["max_steps"]),
            answer_weight=answer_weight,
            evidence_weight=evidence_weight,
            step_cost=step_cost,
        )

    return factory, evidence_by_id


@dataclass
class PolicyState:
    question: torch.Tensor
    selected: torch.Tensor
    candidate_base: torch.Tensor
    candidate_types: torch.Tensor
    similarities: torch.Tensor
    action_mask: torch.Tensor
    candidate_ids: list[str]
    step_fraction: float

    def to(self, device: torch.device) -> "PolicyState":
        return PolicyState(
            question=self.question.to(device),
            selected=self.selected.to(device),
            candidate_base=self.candidate_base.to(device),
            candidate_types=self.candidate_types.to(device),
            similarities=self.similarities.to(device),
            action_mask=self.action_mask.to(device),
            candidate_ids=self.candidate_ids,
            step_fraction=self.step_fraction,
        )


def make_policy_state(env: SequentialEvidenceEnv) -> PolicyState:
    # Cached embeddings are memory-mapped read-only arrays. ``torch.tensor``
    # makes a writable copy and avoids undefined behavior warnings from
    # ``torch.as_tensor``.
    question = torch.tensor(np.asarray(env.question_embedding), dtype=torch.float32)
    selected_vectors = [env.evidence_embeddings[item] for item in env.selected]
    selected = torch.tensor(np.asarray(selected_vectors), dtype=torch.float32)
    if not selected_vectors:
        selected = torch.empty((0, question.numel()), dtype=torch.float32)

    candidate_ids = env.candidate_ids()
    base_vectors: list[np.ndarray] = []
    types: list[int] = []
    similarities: list[float] = []
    q_norm = np.linalg.norm(env.question_embedding) + 1e-8
    for item in candidate_ids:
        if item == STOP_ID:
            base_vectors.append(np.zeros_like(env.question_embedding))
            types.append(2)
            similarities.append(0.0)
            continue
        vector = np.asarray(env.evidence_embeddings[item], dtype="float32")
        base_vectors.append(vector)
        types.append(0 if env.evidence_types[item] == "row" else 1)
        similarities.append(float(np.dot(env.question_embedding, vector) / (q_norm * (np.linalg.norm(vector) + 1e-8))))
    return PolicyState(
        question=question,
        selected=selected,
        candidate_base=torch.tensor(np.asarray(base_vectors), dtype=torch.float32),
        candidate_types=torch.tensor(types, dtype=torch.long),
        similarities=torch.tensor(similarities, dtype=torch.float32),
        action_mask=torch.ones(len(candidate_ids), dtype=torch.bool),
        candidate_ids=candidate_ids,
        step_fraction=len(env.selected) / env.max_steps,
    )


def _materialize_candidates(state: PolicyState, model: EvidenceActorCritic) -> torch.Tensor:
    vectors = state.candidate_base.clone()
    stop_rows = state.candidate_types == 2
    if stop_rows.any():
        vectors[stop_rows] = model.stop_embedding
    return vectors


def policy_forward(model: EvidenceActorCritic, state: PolicyState) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        state.question,
        state.selected,
        _materialize_candidates(state, model),
        state.candidate_types,
        state.similarities,
        state.step_fraction,
        state.action_mask,
    )


def supervised_episode_loss(
    model: EvidenceActorCritic, env: SequentialEvidenceEnv, device: torch.device
) -> torch.Tensor | None:
    env.reset()
    losses: list[torch.Tensor] = []
    possible_sets = [set(items) for items in env.gold_sets]
    while not env.done:
        state = make_policy_state(env).to(device)
        logits, _ = policy_forward(model, state)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_set = set(env.selected)
        compatible = [gold for gold in possible_sets if selected_set.issubset(gold)]
        complete = any(gold.issubset(selected_set) for gold in compatible)
        if complete and STOP_ID in state.candidate_ids:
            target_ids = [STOP_ID]
        else:
            target_ids = sorted(
                {
                    item
                    for gold in compatible
                    for item in gold - selected_set
                    if item in state.candidate_ids
                }
            )
        if not target_ids:
            break
        target_indices = torch.tensor(
            [state.candidate_ids.index(item) for item in target_ids], device=device
        )
        losses.append(-torch.logsumexp(log_probs[target_indices], dim=0))
        chosen_index = target_indices[torch.argmax(log_probs[target_indices])].item()
        chosen_id = state.candidate_ids[chosen_index]
        env.step(chosen_id)
        if chosen_id != STOP_ID:
            possible_sets = [gold for gold in compatible if chosen_id in gold]
    return torch.stack(losses).mean() if losses else None


def train_supervised(
    model: EvidenceActorCritic,
    env_factory: Callable[[str], SequentialEvidenceEnv],
    question_ids: Sequence[str],
    epochs: int,
    learning_rate: float,
    device: torch.device,
    validation_fn: Callable[[EvidenceActorCritic], float] | None = None,
    batch_size: int = 32,
    output_dir: str | Path | None = None,
) -> list[dict[str, float]]:
    from tqdm.auto import tqdm

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_validation = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    start_epoch = 0
    checkpoint_dir = Path(output_dir) if output_dir is not None else None
    latest_path = checkpoint_dir / "latest_supervised.pt" if checkpoint_dir else None
    if latest_path is not None and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = list(checkpoint.get("history", []))
        best_validation = float(checkpoint.get("best_validation", -float("inf")))
        best_state = checkpoint.get("best_model")
        start_epoch = int(checkpoint.get("epoch", 0))
        if checkpoint.get("python_random_state") is not None:
            random.setstate(checkpoint["python_random_state"])
        if checkpoint.get("numpy_random_state") is not None:
            np.random.set_state(checkpoint["numpy_random_state"])
        if checkpoint.get("torch_random_state") is not None:
            torch.set_rng_state(checkpoint["torch_random_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])

    for epoch in range(start_epoch, epochs):
        # Validation uses evaluation mode; every new epoch must explicitly
        # restore training mode before the cuDNN GRU backward pass.
        model.train()
        losses = []
        shuffled = list(question_ids)
        random.shuffle(shuffled)
        pending_losses: list[torch.Tensor] = []
        progress = tqdm(shuffled, desc=f"supervised epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad(set_to_none=True)
        for question_id in progress:
            loss = supervised_episode_loss(model, env_factory(question_id), device)
            if loss is None:
                continue
            losses.append(float(loss.detach().cpu()))
            pending_losses.append(loss)
            if len(pending_losses) >= batch_size:
                torch.stack(pending_losses).mean().backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_losses.clear()
                progress.set_postfix(loss=f"{np.mean(losses[-100:]):.4f}")
        if pending_losses:
            torch.stack(pending_losses).mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        row = {"epoch": float(epoch + 1), "loss": float(np.mean(losses)) if losses else float("nan")}
        improved = validation_fn is None
        if validation_fn is not None:
            validation = float(validation_fn(model))
            row["validation_evidence_f1"] = validation
            if validation > best_validation:
                best_validation = validation
                best_state = copy.deepcopy(model.state_dict())
                improved = True
        history.append(row)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if improved:
                best_path = checkpoint_dir / "best_model.pt"
                best_temporary = best_path.with_suffix(best_path.suffix + ".tmp")
                torch.save(best_state or model.state_dict(), best_temporary)
                os.replace(best_temporary, best_path)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "best_validation": best_validation,
                "best_model": best_state,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            temporary = latest_path.with_suffix(latest_path.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, latest_path)
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.inference_mode()
def greedy_episode(
    model: EvidenceActorCritic,
    env: SequentialEvidenceEnv,
    device: torch.device,
) -> tuple[list[str], float]:
    was_training = model.training
    model.to(device).eval()
    try:
        env.reset()
        total_reward = 0.0
        while not env.done:
            state = make_policy_state(env).to(device)
            logits, _ = policy_forward(model, state)
            action_id = state.candidate_ids[int(torch.argmax(logits).item())]
            _, reward, _, _ = env.step(action_id)
            total_reward += reward
        return list(env.selected), total_reward
    finally:
        model.train(was_training)


def mean_greedy_evidence_f1(
    model: EvidenceActorCritic,
    env_factory: Callable[[str], SequentialEvidenceEnv],
    question_ids: Sequence[str],
    device: torch.device,
) -> float:
    scores = []
    for question_id in question_ids:
        env = env_factory(question_id)
        selected, _ = greedy_episode(model, env, device)
        if env.gold_sets:
            scores.append(best_evidence_f1(selected, env.gold_sets))
    return float(np.mean(scores)) if scores else 0.0


def mean_greedy_episode_reward(
    model: EvidenceActorCritic,
    env_factory: Callable[[str], SequentialEvidenceEnv],
    question_ids: Sequence[str],
    device: torch.device,
) -> float:
    rewards = [greedy_episode(model, env_factory(question_id), device)[1] for question_id in question_ids]
    return float(np.mean(rewards)) if rewards else 0.0


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    next_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
    next_v = torch.tensor(next_value, dtype=values.dtype, device=values.device)
    for index in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_v * nonterminal - values[index]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[index] = last_advantage
        next_v = values[index]
    return advantages, advantages + values


@dataclass
class Transition:
    state: PolicyState
    action_index: int
    old_log_probability: float
    old_value: float
    reward: float
    done: bool
    advantage: float = 0.0
    return_value: float = 0.0


def collect_ppo_rollouts(
    model: EvidenceActorCritic,
    env_factory: Callable[[str], SequentialEvidenceEnv],
    question_ids: Sequence[str],
    episode_count: int,
    device: torch.device,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[Transition], list[dict[str, float]]]:
    model.to(device).eval()
    transitions: list[Transition] = []
    episode_logs: list[dict[str, float]] = []
    for episode_index in range(episode_count):
        question_id = question_ids[episode_index % len(question_ids)]
        env = env_factory(question_id)
        env.reset()
        episode: list[Transition] = []
        component_totals = {"answer": 0.0, "evidence": 0.0, "step_cost": 0.0, "total": 0.0}
        while not env.done:
            cpu_state = make_policy_state(env)
            state = cpu_state.to(device)
            with torch.no_grad():
                logits, value = policy_forward(model, state)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                log_probability = distribution.log_prob(action)
            _, reward, done, components = env.step(cpu_state.candidate_ids[int(action.item())])
            for key in component_totals:
                component_totals[key] += components[key]
            episode.append(
                Transition(
                    state=cpu_state,
                    action_index=int(action.item()),
                    old_log_probability=float(log_probability.cpu()),
                    old_value=float(value.cpu()),
                    reward=reward,
                    done=done,
                )
            )
        rewards = torch.tensor([item.reward for item in episode], dtype=torch.float32)
        values = torch.tensor([item.old_value for item in episode], dtype=torch.float32)
        dones = torch.tensor([float(item.done) for item in episode], dtype=torch.float32)
        advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)
        for item, advantage, return_value in zip(episode, advantages, returns):
            item.advantage = float(advantage)
            item.return_value = float(return_value)
        transitions.extend(episode)
        component_totals["steps"] = float(len(episode))
        episode_logs.append(component_totals)
    return transitions, episode_logs


def ppo_update(
    model: EvidenceActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    device: torch.device,
    update_epochs: int = 4,
    minibatch_size: int = 128,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.03,
) -> dict[str, float]:
    if minibatch_size < 1:
        raise ValueError("minibatch_size must be positive")
    model.to(device).train()
    raw_advantages = torch.tensor([item.advantage for item in transitions], dtype=torch.float32)
    normalized = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std(unbiased=False) + 1e-8)
    metric_rows: list[dict[str, float]] = []
    for _ in range(update_epochs):
        order = torch.randperm(len(transitions)).tolist()
        epoch_kls: list[float] = []
        for start in range(0, len(order), minibatch_size):
            positions = order[start : start + minibatch_size]
            policy_losses, value_losses, entropies, kls, clip_fractions = [], [], [], [], []
            for position in positions:
                item = transitions[position]
                state = item.state.to(device)
                logits, value = policy_forward(model, state)
                distribution = Categorical(logits=logits)
                action = torch.tensor(item.action_index, device=device)
                new_log_probability = distribution.log_prob(action)
                old_log_probability = torch.tensor(item.old_log_probability, device=device)
                log_ratio = new_log_probability - old_log_probability
                ratio = torch.exp(log_ratio)
                advantage = normalized[position].to(device)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
                policy_losses.append(-torch.minimum(unclipped, clipped))
                target_value = torch.tensor(item.return_value, device=device)
                value_losses.append((value - target_value).pow(2))
                entropies.append(distribution.entropy())
                kls.append((ratio - 1.0) - log_ratio)
                clip_fractions.append((torch.abs(ratio - 1.0) > clip_ratio).float())
            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy = torch.stack(entropies).mean()
            approximate_kl = torch.stack(kls).mean()
            clip_fraction = torch.stack(clip_fractions).mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            row = {
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "approximate_kl": float(approximate_kl.detach().cpu()),
                "clip_fraction": float(clip_fraction.detach().cpu()),
            }
            metric_rows.append(row)
            epoch_kls.append(row["approximate_kl"])
        if epoch_kls and float(np.mean(epoch_kls)) > target_kl:
            break
    return {
        name: float(np.mean([row[name] for row in metric_rows]))
        for name in metric_rows[0]
    }


def train_ppo(
    model: EvidenceActorCritic,
    env_factory: Callable[[str], SequentialEvidenceEnv],
    question_ids: Sequence[str],
    config: Mapping[str, Any],
    output_dir: str | Path,
    device: torch.device,
    validation_fn: Callable[[EvidenceActorCritic], float] | None = None,
) -> list[dict[str, float]]:
    """Complete PPO loop. Intended for ``colab_gpu.ipynb`` only."""

    from tqdm.auto import tqdm

    settings = config["training"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["ppo_learning_rate"]))
    episode = 0
    best_validation = -float("inf")
    logs: list[dict[str, float]] = []
    checks_without_improvement = 0
    checkpoint_every = int(settings["checkpoint_every_episodes"])
    latest_path = output / "latest.pt"
    if latest_path.exists():
        progress = load_training_checkpoint(latest_path, model, optimizer, device=device)
        episode = int(progress.get("episode", 0))
        logs = list(progress.get("logs", []))
        best_validation = float(progress.get("best_validation", -float("inf")))
        checks_without_improvement = int(progress.get("checks_without_improvement", 0))
    progress_bar = tqdm(total=int(settings["max_episodes"]), initial=episode, desc="PPO episodes")
    while episode < int(settings["max_episodes"]):
        rollout_size = min(int(settings["rollout_episodes"]), int(settings["max_episodes"]) - episode)
        sampled_ids = random.choices(list(question_ids), k=rollout_size)
        transitions, episode_logs = collect_ppo_rollouts(
            model,
            env_factory,
            sampled_ids,
            rollout_size,
            device,
            gamma=float(settings["gamma"]),
            gae_lambda=float(settings["gae_lambda"]),
        )
        update_metrics = ppo_update(
            model,
            optimizer,
            transitions,
            device,
            update_epochs=int(settings["update_epochs"]),
            minibatch_size=int(settings.get("ppo_minibatch_size", 128)),
            clip_ratio=float(settings["clip_ratio"]),
            value_coefficient=float(settings["value_coefficient"]),
            entropy_coefficient=float(settings["entropy_coefficient"]),
            max_grad_norm=float(settings["max_grad_norm"]),
            target_kl=float(settings["target_kl"]),
        )
        episode += rollout_size
        reward_mean = float(np.mean([row["total"] for row in episode_logs]))
        log_row = {"episode": float(episode), "reward": reward_mean, **update_metrics}
        logs.append(log_row)
        progress_bar.update(rollout_size)
        progress_bar.set_postfix(reward=f"{reward_mean:.4f}")
        should_checkpoint = episode % checkpoint_every < rollout_size or episode >= int(settings["max_episodes"])
        if should_checkpoint:
            validation = validation_fn(model) if validation_fn is not None else reward_mean
            log_row["validation"] = float(validation)
            improved = validation > best_validation
            if improved:
                best_validation = float(validation)
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            progress = {
                "episode": episode,
                "logs": logs,
                "best_validation": best_validation,
                "checks_without_improvement": checks_without_improvement,
            }
            save_training_checkpoint(latest_path, model, optimizer, progress, config)
            if improved:
                save_training_checkpoint(output / "best.pt", model, optimizer, progress, config)
            patience = int(settings.get("ppo_early_stopping_patience", 0))
            if patience > 0 and checks_without_improvement >= patience:
                break
    progress_bar.close()
    return logs


def save_training_checkpoint(
    path: str | Path,
    model: EvidenceActorCritic,
    optimizer: torch.optim.Optimizer,
    progress: Mapping[str, Any],
    config: Mapping[str, Any],
    scheduler: Any | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "progress": dict(progress),
        "config": dict(config),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, temporary)
    os.replace(temporary, target)


def load_training_checkpoint(
    path: str | Path,
    model: EvidenceActorCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
    return dict(checkpoint["progress"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_inference_bundle(
    output_dir: str | Path,
    model: EvidenceActorCritic,
    policy_config: Mapping[str, Any],
    evidence_embeddings: np.ndarray,
    evidence_index: Mapping[str, int],
    preprocessing_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    import yaml

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    torch.save(model.inference_state_dict(), root / "policy_state.pt")
    with (root / "policy_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(policy_config), handle, sort_keys=False)
    np.save(root / "evidence_embeddings.npy", np.asarray(evidence_embeddings, dtype="float32"))
    with (root / "evidence_index.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(evidence_index), handle, indent=2)
    with (root / "preprocessing_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(preprocessing_manifest), handle, indent=2)
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "not-a-git-checkout"
    environment = f"torch={torch.__version__}\nnumpy={np.__version__}\ngit_commit={git_commit}\n"
    (root / "environment.txt").write_text(environment, encoding="utf-8")
    (root / "README.txt").write_text(
        "Load policy_state.pt with models.load_policy_weights(..., device='cpu').\n"
        "Verify manifest.json before use. Training is not required locally.\n",
        encoding="utf-8",
    )
    files = [path for path in root.iterdir() if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files
        }
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_inference_bundle(path: str | Path) -> None:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for filename, metadata in manifest["files"].items():
        file_path = root / filename
        if not file_path.exists() or _sha256(file_path) != metadata["sha256"]:
            raise RuntimeError(f"Inference bundle checksum failed: {filename}")


def sanity_check_training_math() -> None:
    rewards = torch.tensor([0.0, 1.0])
    values = torch.tensor([0.2, 0.3])
    dones = torch.tensor([0.0, 1.0])
    advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=1.0)
    expected_advantages = torch.tensor([0.8, 0.7])
    if not torch.allclose(advantages, expected_advantages, atol=1e-6):
        raise AssertionError(f"GAE mismatch: {advantages} != {expected_advantages}")
    if not torch.allclose(returns, torch.tensor([1.0, 1.0]), atol=1e-6):
        raise AssertionError("Return calculation mismatch")
    logits = torch.tensor([1.0, torch.finfo(torch.float32).min])
    distribution = Categorical(logits=logits)
    if distribution.probs[1].item() != 0.0:
        raise AssertionError("Masked action has non-zero probability")


def run_synthetic_ppo_sanity(seed: int = 42, updates: int = 150) -> float:
    """Tiny two-action PPO check. Defined for validation; not run during preprocessing."""

    set_seed(seed)
    logits = nn.Parameter(torch.zeros(2))
    value = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([logits, value], lr=0.05)
    for _ in range(updates):
        old_distribution = Categorical(logits=logits.detach())
        actions = old_distribution.sample((64,))
        rewards = (actions == 1).float()
        old_log_probs = old_distribution.log_prob(actions)
        advantages = rewards - value.detach()
        distribution = Categorical(logits=logits)
        ratios = torch.exp(distribution.log_prob(actions) - old_log_probs)
        surrogate = torch.minimum(ratios * advantages, torch.clamp(ratios, 0.8, 1.2) * advantages)
        loss = -surrogate.mean() + 0.5 * (value - rewards.mean()).pow(2) - 0.01 * distribution.entropy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    probability = float(torch.softmax(logits.detach(), dim=0)[1])
    if probability < 0.9:
        raise AssertionError(f"Synthetic PPO failed to learn the rewarding action: p={probability:.3f}")
    return probability
