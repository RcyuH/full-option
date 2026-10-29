"""Durable per-example tapes and immutable run provenance for safe resumption."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

from loopsd_search.config import ExperimentConfig


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def model_fingerprint(path: Path) -> dict[str, str]:
    files = sorted(set(path.glob("*.json")) | set(path.glob("*.safetensors")))
    if not any(item.suffix == ".safetensors" for item in files):
        raise ValueError("Expected a local safetensors checkpoint")
    files += sorted(path.glob("*.txt"))
    return {item.name: digest_file(item) for item in files}


def directory_fingerprint(path: Path) -> str:
    """Hash source files by relative path, without requiring Git metadata."""
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
    )
    if not files:
        raise ValueError(f"No source files found under {path}")
    inventory = {
        item.relative_to(path).as_posix(): digest_file(item) for item in files
    }
    return digest_json(inventory)


def make_manifest(cfg: ExperimentConfig, split_ids: dict[str, list[str]]) -> dict:
    root = Path(__file__).resolve().parent
    source = {item.name: digest_file(item) for item in sorted(root.glob("*.py"))}
    nemo_package = (cfg.nemo_rl_path / "nemo_rl").resolve()
    value = {
        "schema_version": 1,
        "config": cfg.model_dump(mode="json"),
        "model_sha256": model_fingerprint(cfg.model.path),
        "train_sha256": digest_file(cfg.data.train),
        "test_sha256": digest_file(cfg.data.test),
        "source_sha256": source,
        # Hash the actual package files. This works for both Git clones and
        # source snapshots/archives where nested .git metadata is unavailable.
        "nemo_source_sha256": directory_fingerprint(nemo_package),
        "split_ids": split_ids,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "numpy", "pyarrow", "pydantic")
        },
        "assumptions": [
            "Prompt/cue are reconstruction assumptions; author strings are absent.",
            "Shuffle: numpy.default_rng(seed).permutation; author library is unspecified.",
            "Thinking stops at </think> or cap; injected closure/cue are not generated tokens.",
            "Premature EOS is terminal and produces an empty answer, scored incorrect.",
            "Scorer: answer stage only; last boxed, answer marker, or numeric final line.",
            "Loop ties: Python random.Random(seed) on lexicographically sorted IDs.",
            "Same token budget is not a matched-FLOP comparison.",
        ],
    }
    value["fingerprint"] = digest_json(value)
    return value


class RunStore:
    """Callers hold a file lock for the entire run, including manifest validation."""

    # These modules changed in the follow-up that added configurable loop
    # factors. The actual generation/search implementation is separately
    # fingerprinted and must still match the original run.
    _RESUME_COMPATIBLE_SOURCE_FILES = {
        "artifacts.py",
        "cli.py",
        "config.py",
        "report.py",
        "routes.py",
    }

    def __init__(
        self, root: Path, manifest: dict, *, allow_batch_size_change: bool = False
    ) -> None:
        self.root = root
        self.fingerprint = manifest["fingerprint"]
        path = root / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old != manifest:
                if not allow_batch_size_change or not self._compatible_batch_resume(old, manifest):
                    raise ValueError(
                        "Run provenance changed. Choose a fresh output_dir; refusing mixed results."
                    )
                # Existing tapes and selected artifacts retain the original
                # fingerprint. Only missing examples are generated with the
                # explicitly requested, lower runtime batch size.
                self.fingerprint = old["fingerprint"]
                print(
                    "Resuming existing run with a different batch_size; retaining its original fingerprint.",
                    flush=True,
                )
        else:
            if allow_batch_size_change:
                raise ValueError("--resume-batch-size requires an existing manifest.json")
            if any(root.glob("predictions/**/*.jsonl")) or (root / "selected_route.json").exists():
                raise ValueError("Existing results without a manifest; choose a fresh output_dir")
            atomic_json(path, manifest)

    @classmethod
    def _compatible_batch_resume(cls, old: dict, new: dict) -> bool:
        """Permit only a batch-size change plus known loop-factor release metadata."""
        if old.get("schema_version") != new.get("schema_version"):
            return False
        for candidate in (old, new):
            fingerprint = candidate.get("fingerprint")
            payload = {key: value for key, value in candidate.items() if key != "fingerprint"}
            if fingerprint != digest_json(payload):
                return False

        old_config = json.loads(json.dumps(old.get("config", {})))
        new_config = json.loads(json.dumps(new.get("config", {})))
        try:
            old_batch = old_config["model"]["batch_size"]
            new_batch = new_config["model"]["batch_size"]
            if old_batch == new_batch:
                return False
            old_config["model"]["batch_size"] = new_config["model"]["batch_size"]
            # loop_factors was added later; absent in old manifests means [2].
            old_config.setdefault("search", {}).setdefault("loop_factors", [2])
            new_config.setdefault("search", {}).setdefault("loop_factors", [2])
        except (KeyError, TypeError):
            return False
        if old_config != new_config:
            return False

        old_sources = old.get("source_sha256", {})
        new_sources = new.get("source_sha256", {})
        if old_sources.keys() != new_sources.keys():
            return False
        changed_sources = {
            name for name in old_sources if old_sources[name] != new_sources[name]
        }
        if not changed_sources.issubset(cls._RESUME_COMPATIBLE_SOURCE_FILES):
            return False

        old_without_fingerprint = {k: v for k, v in old.items() if k != "fingerprint"}
        new_without_fingerprint = {k: v for k, v in new.items() if k != "fingerprint"}
        old_without_fingerprint["config"] = old_config
        new_without_fingerprint["config"] = new_config
        # Compare everything else exactly, allowing source hashes only for
        # the explicitly listed, non-generation modules above.
        old_without_fingerprint["source_sha256"] = {}
        new_without_fingerprint["source_sha256"] = {}
        return old_without_fingerprint == new_without_fingerprint

    def tape_path(self, stage: str, route_id: str) -> Path:
        return self.root / "predictions" / stage / (route_id.replace(":", "_") + ".jsonl")

    def read_tape(self, stage: str, route_id: str) -> dict[str, dict]:
        path = self.tape_path(stage, route_id)
        if not path.exists():
            return {}
        records = {}
        # Only a final incomplete write is recoverable. Corruption in a complete row is fatal.
        with path.open("r+b") as stream:
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(start)
                    break
                row = json.loads(line)
                if row["run_fingerprint"] != self.fingerprint or row["route"] != route_id:
                    raise ValueError(f"Mismatched provenance in {path}")
                if row["id"] in records:
                    raise ValueError(f"Duplicate example {row['id']} in {path}")
                records[row["id"]] = row
        return records

    def append(self, stage: str, route_id: str, rows: list[dict]) -> None:
        path = self.tape_path(stage, route_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
