"""HybridQA loading, auditing, preprocessing, and candidate generation.

The module intentionally uses plain dictionaries and small functions.  Dataset
preprocessing can run without importing PyTorch or Transformers.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    import yaml

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = config_path.parent
    overrides = {
        "hybridqa_root": os.getenv("HYBRIDQA_ROOT"),
        "wikitables_root": os.getenv("WIKITABLES_ROOT"),
        "processed_dir": os.getenv("HYBRID_RL_PROCESSED_DIR"),
        "output_dir": os.getenv("HYBRID_RL_OUTPUT_DIR"),
    }
    for key, value in list(config["paths"].items()):
        resolved = overrides.get(key) or value
        resolved_path = Path(resolved)
        config["paths"][key] = str(
            resolved_path if resolved_path.is_absolute() else (base / resolved_path).resolve()
        )
    config["config_path"] = str(config_path)
    return config


def normalize_text(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _repo_json(repo_root: Path, relative_path: str) -> Any:
    """Read a repository file even when the Windows worktree is not checked out.

    WikiTables-WithLinks contains a few filenames that Windows cannot represent.
    Clone it with ``git clone --no-checkout``; this fallback reads blobs directly
    from Git while processed tables are saved under safe hash filenames.
    """

    normal_path = repo_root / Path(relative_path)
    if normal_path.exists():
        return _read_json(normal_path)
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        raise FileNotFoundError(
            f"Missing {normal_path}. The repository may be cloned with --no-checkout "
            "on Windows so files can be read from Git objects."
        )
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"HEAD:{relative_path}"],
        check=True,
        capture_output=True,
    )
    return json.loads(result.stdout.decode("utf-8"))


class RepositoryJSONReader:
    """Efficiently stream JSON blobs from a normal or no-checkout Git repository."""

    def __init__(self, repo_root: str | Path) -> None:
        self.root = Path(repo_root)
        self.process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "RepositoryJSONReader":
        if (self.root / ".git").exists():
            self.process = subprocess.Popen(
                ["git", "-C", str(self.root), "cat-file", "--batch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        return self

    def __exit__(self, *_: Any) -> None:
        if self.process is not None:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=5)

    def read(self, relative_path: str) -> Any:
        normal_path = self.root / Path(relative_path)
        if normal_path.exists():
            return _read_json(normal_path)
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            return _repo_json(self.root, relative_path)
        self.process.stdin.write(f"HEAD:{relative_path}\n".encode("utf-8"))
        self.process.stdin.flush()
        header = self.process.stdout.readline().decode("utf-8").strip()
        if header.endswith(" missing"):
            raise FileNotFoundError(f"Git object is missing: {relative_path}")
        parts = header.split()
        if len(parts) != 3 or parts[1] != "blob":
            raise RuntimeError(f"Unexpected git cat-file response for {relative_path}: {header}")
        size = int(parts[2])
        content = self.process.stdout.read(size)
        self.process.stdout.read(1)  # trailing newline added by --batch
        return json.loads(content.decode("utf-8"))


def load_questions(hybridqa_root: str | Path, split: str, traced: bool = False) -> list[dict[str, Any]]:
    suffix = ".traced.json" if traced else ".json"
    path = Path(hybridqa_root) / "released_data" / f"{split}{suffix}"
    return _read_json(path)


def load_table_resources(
    wikitables_root: str | Path,
    table_id: str,
    reader: RepositoryJSONReader | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    root = Path(wikitables_root)
    read = reader.read if reader is not None else lambda path: _repo_json(root, path)
    table = read(f"tables_tok/{table_id}.json")
    passages = read(f"request_tok/{table_id}.json")
    return table, passages


def _safe_table_filename(table_id: str) -> str:
    return hashlib.sha1(table_id.encode("utf-8")).hexdigest() + ".json"


def split_by_table(
    examples: Sequence[Mapping[str, Any]], validation_fraction: float = 0.1, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table_ids = sorted({str(example["table_id"]) for example in examples})
    random.Random(seed).shuffle(table_ids)
    validation_count = max(1, round(len(table_ids) * validation_fraction))
    validation_tables = set(table_ids[:validation_count])
    train = [dict(x) for x in examples if str(x["table_id"]) not in validation_tables]
    validation = [dict(x) for x in examples if str(x["table_id"]) in validation_tables]
    return train, validation


def audit_dataset(config: Mapping[str, Any], inspect_tables: bool = True) -> dict[str, Any]:
    hybridqa = Path(config["paths"]["hybridqa_root"])
    wikitables = Path(config["paths"]["wikitables_root"])
    output = Path(config["paths"]["output_dir"]) / "data_audit.json"

    report: dict[str, Any] = {
        "hybridqa_root": str(hybridqa),
        "wikitables_root": str(wikitables),
        "splits": {},
        "errors": [],
    }
    all_questions: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        path = hybridqa / "released_data" / f"{split}.json"
        if not path.exists():
            report["errors"].append(f"missing: {path}")
            continue
        examples = _read_json(path)
        ids = [str(x.get("question_id", "")) for x in examples]
        tables = [str(x.get("table_id", "")) for x in examples]
        normalized_questions = {normalize_text(x.get("question", "")).lower() for x in examples}
        all_questions[split] = normalized_questions
        report["splits"][split] = {
            "examples": len(examples),
            "unique_question_ids": len(set(ids)),
            "duplicate_question_ids": len(ids) - len(set(ids)),
            "unique_tables": len(set(tables)),
            "sha256": sha256_file(path),
        }

    if "train" in all_questions and "dev" in all_questions:
        report["normalized_question_overlap_train_dev"] = len(
            all_questions["train"] & all_questions["dev"]
        )

    traced_path = hybridqa / "released_data" / "train.traced.json"
    traced = _read_json(traced_path) if traced_path.exists() else []
    trace_types: Counter[str] = Counter()
    empty_traces = 0
    invalid_nodes = 0
    table_ids: set[str] = set()
    for example in traced:
        table_ids.add(str(example.get("table_id", "")))
        nodes = example.get("answer-node") or []
        empty_traces += int(not nodes)
        for node in nodes:
            if not isinstance(node, list) or len(node) < 4:
                invalid_nodes += 1
                continue
            trace_types[str(node[3])] += 1
    report["traces"] = {
        "examples": len(traced),
        "empty": empty_traces,
        "invalid_nodes": invalid_nodes,
        "types": dict(trace_types),
    }

    if inspect_tables and table_ids:
        missing: list[str] = []
        malformed: list[str] = []
        row_counts: list[int] = []
        link_count = 0
        with RepositoryJSONReader(wikitables) as reader:
            for table_id in sorted(table_ids):
                try:
                    table, passages = load_table_resources(wikitables, table_id, reader)
                    rows = table.get("data", [])
                    if not isinstance(rows, list) or not isinstance(passages, dict):
                        malformed.append(table_id)
                        continue
                    row_counts.append(len(rows))
                    link_count += sum(
                        len(cell[1])
                        for row in rows
                        for cell in row
                        if isinstance(cell, list) and len(cell) >= 2 and isinstance(cell[1], list)
                    )
                except (FileNotFoundError, subprocess.CalledProcessError):
                    missing.append(table_id)
        report["tables"] = {
            "checked": len(table_ids),
            "missing": missing,
            "malformed": malformed,
            "min_rows": min(row_counts) if row_counts else 0,
            "max_rows": max(row_counts) if row_counts else 0,
            "mean_rows": sum(row_counts) / len(row_counts) if row_counts else 0.0,
            "total_links": link_count,
        }

    _write_json(output, report)
    return report


def _cell_text(cell: Any) -> str:
    if isinstance(cell, list) and cell:
        return normalize_text(cell[0])
    return normalize_text(cell)


def _cell_links(cell: Any) -> list[str]:
    if isinstance(cell, list) and len(cell) > 1 and isinstance(cell[1], list):
        return [str(link) for link in cell[1]]
    return []


def build_table_evidence(
    table_id: str, table: Mapping[str, Any], passages: Mapping[str, str], max_passage_chars: int = 12000
) -> dict[str, Any]:
    title = normalize_text(table.get("title", table_id))
    headers = [_cell_text(cell) for cell in table.get("header", [])]
    rows: list[dict[str, Any]] = []
    passage_to_rows: dict[str, list[str]] = {}
    for row_number, raw_row in enumerate(table.get("data", [])):
        row_id = f"row::{table_id}::{row_number}"
        values = [_cell_text(cell) for cell in raw_row]
        pairs = [f"{headers[i] if i < len(headers) else f'column_{i}'}: {value}" for i, value in enumerate(values)]
        links = sorted({link for cell in raw_row for link in _cell_links(cell)})
        rows.append(
            {
                "evidence_id": row_id,
                "type": "row",
                "row_number": row_number,
                "text": f"Table: {title} | " + " | ".join(pairs),
                "raw_values": values,
                "links": links,
            }
        )
        for link in links:
            passage_to_rows.setdefault(link, []).append(row_id)

    passage_items = []
    for passage_id, text in passages.items():
        passage_items.append(
            {
                "evidence_id": f"passage::{table_id}::{passage_id}",
                "type": "passage",
                "passage_id": passage_id,
                "text": normalize_text(text)[:max_passage_chars],
                "linked_rows": passage_to_rows.get(passage_id, []),
            }
        )
    return {
        "table_id": table_id,
        "title": title,
        "url": table.get("url"),
        "headers": headers,
        "rows": rows,
        "passages": passage_items,
    }


def derive_weak_evidence_sets(example: Mapping[str, Any]) -> list[list[str]]:
    table_id = str(example["table_id"])
    alternatives: set[tuple[str, ...]] = set()
    for node in example.get("answer-node") or []:
        if not isinstance(node, list) or len(node) < 4:
            continue
        location, passage_id, node_type = node[1], node[2], str(node[3]).lower()
        if not isinstance(location, (list, tuple)) or len(location) < 1:
            continue
        row_id = f"row::{table_id}::{int(location[0])}"
        if node_type == "passage" and passage_id:
            evidence_set = tuple(sorted((row_id, f"passage::{table_id}::{passage_id}")))
        else:
            evidence_set = (row_id,)
        alternatives.add(evidence_set)
    return [list(items) for items in sorted(alternatives)]


def _question_record(example: Mapping[str, Any], split: str) -> dict[str, Any]:
    return {
        "question_id": str(example["question_id"]),
        "question": normalize_text(example["question"]),
        "table_id": str(example["table_id"]),
        "answer": normalize_text(example.get("answer-text", "")),
        "split": split,
        "weak_evidence_sets": derive_weak_evidence_sets(example),
    }


def preprocess_hybridqa(
    config: Mapping[str, Any], max_examples: int | None = None, include_dev: bool = True
) -> dict[str, Any]:
    """Preprocess HybridQA into compact question JSONL and per-table evidence JSON."""

    hybridqa = Path(config["paths"]["hybridqa_root"])
    wikitables = Path(config["paths"]["wikitables_root"])
    processed = Path(config["paths"]["processed_dir"])
    question_dir, table_dir = processed / "questions", processed / "tables"
    question_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    train = load_questions(hybridqa, "train", traced=True)
    train_core, train_val = split_by_table(
        train,
        float(config["data"]["validation_fraction"]),
        int(config["data"]["split_seed"]),
    )
    split_examples: dict[str, list[dict[str, Any]]] = {
        "train_core": train_core,
        "train_val": train_val,
    }
    if include_dev:
        split_examples["dev"] = load_questions(hybridqa, "dev", traced=True)
    if max_examples is not None:
        split_examples = {name: rows[:max_examples] for name, rows in split_examples.items()}

    table_ids = sorted({str(x["table_id"]) for rows in split_examples.values() for x in rows})
    table_index: dict[str, str] = {}
    failures: dict[str, str] = {}
    with RepositoryJSONReader(wikitables) as reader:
        for table_id in table_ids:
            try:
                table, passages = load_table_resources(wikitables, table_id, reader)
                evidence = build_table_evidence(
                    table_id,
                    table,
                    passages,
                    int(config["data"].get("max_passage_chars", 12000)),
                )
                filename = _safe_table_filename(table_id)
                _write_json(table_dir / filename, evidence)
                table_index[table_id] = filename
            except Exception as error:  # preserve failures for inspection; do not silently drop
                failures[table_id] = f"{type(error).__name__}: {error}"

    counts: dict[str, int] = {}
    for split, examples in split_examples.items():
        records = (
            _question_record(example, split)
            for example in examples
            if str(example["table_id"]) in table_index
        )
        counts[split] = _write_jsonl(question_dir / f"{split}.jsonl", records)

    _write_json(processed / "table_index.json", table_index)
    manifest = {
        "schema_version": 1,
        "max_examples_per_split": max_examples,
        "question_counts": counts,
        "tables_written": len(table_index),
        "table_failures": failures,
        "split_seed": int(config["data"]["split_seed"]),
        "validation_fraction": float(config["data"]["validation_fraction"]),
    }
    _write_json(processed / "preprocessing_manifest.json", manifest)
    return manifest


def load_processed_table(processed_dir: str | Path, table_id: str) -> dict[str, Any]:
    root = Path(processed_dir)
    index = _read_json(root / "table_index.json")
    return _read_json(root / "tables" / index[table_id])


class CandidateGenerator:
    """Dynamic row/passage candidates using a shared cached embedding map."""

    def __init__(
        self,
        table_evidence: Mapping[str, Any],
        embeddings: Mapping[str, Any],
        top_rows: int = 12,
        top_passages: int = 20,
    ) -> None:
        import numpy as np

        self.np = np
        self.table = table_evidence
        self.embeddings = embeddings
        self.top_rows = top_rows
        self.top_passages = top_passages
        self.items = {
            item["evidence_id"]: item
            for item in list(table_evidence["rows"]) + list(table_evidence["passages"])
        }
        self.row_to_passages: dict[str, list[str]] = {row["evidence_id"]: [] for row in table_evidence["rows"]}
        self.passage_to_rows: dict[str, list[str]] = {}
        for passage in table_evidence["passages"]:
            pid = passage["evidence_id"]
            linked_rows = list(passage.get("linked_rows", []))
            self.passage_to_rows[pid] = linked_rows
            for row_id in linked_rows:
                self.row_to_passages.setdefault(row_id, []).append(pid)

    def _score(self, query: Any, evidence_id: str) -> float:
        vector = self.np.asarray(self.embeddings[evidence_id], dtype="float32")
        return float(self.np.dot(query, vector) / (self.np.linalg.norm(query) * self.np.linalg.norm(vector) + 1e-8))

    def candidates(self, question_embedding: Any, selected_ids: Sequence[str]) -> list[str]:
        np = self.np
        question = np.asarray(question_embedding, dtype="float32")
        if selected_ids:
            selected_vectors = [np.asarray(self.embeddings[item], dtype="float32") for item in selected_ids]
            query = question + np.mean(selected_vectors, axis=0)
        else:
            query = question

        rows = [row["evidence_id"] for row in self.table["rows"]]
        ranked_rows = sorted(rows, key=lambda item: (-self._score(question, item), item))[: self.top_rows]
        available: set[str] = set(ranked_rows)
        linked_passages: set[str] = set()
        for selected in selected_ids:
            linked_passages.update(self.row_to_passages.get(selected, []))
            available.update(self.passage_to_rows.get(selected, []))
        ranked_passages = sorted(
            linked_passages, key=lambda item: (-self._score(query, item), item)
        )[: self.top_passages]
        available.update(ranked_passages)
        available.difference_update(selected_ids)
        return sorted(available, key=lambda item: (-self._score(query, item), item))
