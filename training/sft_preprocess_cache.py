"""Disk cache for SFT chat-template formatting + tokenization."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from data.swe_utils import load_jsonl, prepare_gemma4_chat_messages

CACHE_VERSION = 1
READY_MARKER = ".ready"
MANIFEST_NAME = "manifest.json"
DATASET_DIR_NAME = "dataset"


def _train_path_fingerprint(train_path: Path) -> dict[str, int]:
    stat = train_path.stat()
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def cache_fingerprint(
    *,
    train_path: str | Path,
    model_path: str,
    max_seq_length: int,
    max_samples: int | None,
) -> str:
    train_path = Path(train_path)
    fp = _train_path_fingerprint(train_path)
    payload = (
        f"v{CACHE_VERSION}|{train_path.resolve()}|{fp['mtime_ns']}|{fp['size']}|"
        f"{model_path}|{max_seq_length}|{max_samples or ''}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def resolve_cache_dir(base_dir: str | Path, fingerprint: str) -> Path:
    return Path(base_dir) / fingerprint


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / MANIFEST_NAME


def _ready_path(cache_dir: Path) -> Path:
    return cache_dir / READY_MARKER


def _dataset_path(cache_dir: Path) -> Path:
    return cache_dir / DATASET_DIR_NAME


def build_manifest(
    *,
    train_path: str | Path,
    model_path: str,
    max_seq_length: int,
    max_samples: int | None,
    num_samples: int,
    tokenizer_eos_token: str | None,
) -> dict[str, Any]:
    train_path = Path(train_path)
    manifest = {
        "version": CACHE_VERSION,
        "train_path": str(train_path.resolve()),
        "model_path": model_path,
        "max_seq_length": max_seq_length,
        "max_samples": max_samples,
        "num_samples": num_samples,
        "tokenizer_eos_token": tokenizer_eos_token,
    }
    manifest.update(_train_path_fingerprint(train_path))
    return manifest


def manifest_matches(
    manifest: dict[str, Any],
    *,
    train_path: str | Path,
    model_path: str,
    max_seq_length: int,
    max_samples: int | None,
) -> bool:
    train_path = Path(train_path)
    expected = build_manifest(
        train_path=train_path,
        model_path=model_path,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
        num_samples=manifest.get("num_samples", -1),
        tokenizer_eos_token=manifest.get("tokenizer_eos_token"),
    )
    keys = ("version", "mtime_ns", "size", "model_path", "max_seq_length", "max_samples")
    return all(manifest.get(k) == expected.get(k) for k in keys)


def cache_is_ready(cache_dir: Path) -> bool:
    return (
        _ready_path(cache_dir).is_file()
        and _manifest_path(cache_dir).is_file()
        and _dataset_path(cache_dir).is_dir()
    )


def should_use_cache(
    *,
    cache_dir: Path,
    train_path: str | Path,
    model_path: str,
    max_seq_length: int,
    max_samples: int | None,
    use_preprocessed: bool,
    force_preprocess: bool,
) -> bool:
    if force_preprocess:
        return False
    if not use_preprocessed:
        return False
    if not cache_is_ready(cache_dir):
        return False
    manifest = json.loads(_manifest_path(cache_dir).read_text(encoding="utf-8"))
    return manifest_matches(
        manifest,
        train_path=train_path,
        model_path=model_path,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
    )


def _format_and_tokenize_row(row: dict[str, Any], tokenizer) -> dict[str, list[int]]:
    messages = prepare_gemma4_chat_messages(row["messages"])
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    eos = tokenizer.eos_token
    if eos and not text.endswith(eos):
        text = text + eos
    return {"input_ids": tokenizer(text)["input_ids"]}


def build_tokenized_dataset(
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    num_proc: int = 1,
) -> Dataset:
    dataset = Dataset.from_list(rows)

    def _map_fn(example: dict[str, Any]) -> dict[str, list[int]]:
        return _format_and_tokenize_row(example, tokenizer)

    map_kwargs: dict[str, Any] = {"remove_columns": dataset.column_names}
    if num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    return dataset.map(_map_fn, **map_kwargs)


def save_preprocessed_cache(
    dataset: Dataset,
    cache_dir: Path,
    manifest: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_dir / "_building"
    if tmp_dir.exists():
        import shutil

        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    dataset.save_to_disk(str(tmp_dir / DATASET_DIR_NAME))
    (tmp_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ready = tmp_dir / READY_MARKER
    ready.write_text("ok\n", encoding="utf-8")

    import shutil

    for name in (DATASET_DIR_NAME, MANIFEST_NAME, READY_MARKER):
        src = tmp_dir / name
        dst = cache_dir / name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
    shutil.rmtree(tmp_dir, ignore_errors=True)


def load_preprocessed_cache(cache_dir: Path) -> Dataset:
    if not cache_is_ready(cache_dir):
        raise FileNotFoundError(f"Preprocessed cache not ready: {cache_dir}")
    return load_from_disk(str(_dataset_path(cache_dir)))


def wait_for_cache(cache_dir: Path, *, timeout_s: float = 7200.0, poll_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if cache_is_ready(cache_dir):
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for preprocessed cache: {cache_dir}")


def prepare_sft_dataset(
    *,
    train_path: str,
    model_path: str,
    max_seq_length: int,
    max_samples: int | None,
    tokenizer,
    cache_base_dir: str | Path,
    use_preprocessed: bool,
    force_preprocess: bool,
    preprocess_num_proc: int,
    is_main_process: bool,
) -> tuple[Dataset, Path, bool]:
    """Return (dataset, cache_dir, loaded_from_cache)."""
    rows = load_jsonl(train_path)
    if max_samples:
        rows = rows[:max_samples]
    if not rows:
        raise SystemExit(f"No training rows in {train_path}")

    fingerprint = cache_fingerprint(
        train_path=train_path,
        model_path=model_path,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
    )
    cache_dir = resolve_cache_dir(cache_base_dir, fingerprint)

    if should_use_cache(
        cache_dir=cache_dir,
        train_path=train_path,
        model_path=model_path,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
        use_preprocessed=use_preprocessed,
        force_preprocess=force_preprocess,
    ):
        if is_main_process:
            print(f"==> Loading preprocessed SFT cache: {cache_dir}")
        if not is_main_process:
            wait_for_cache(cache_dir)
        return load_preprocessed_cache(cache_dir), cache_dir, True

    if is_main_process and _ready_path(cache_dir).exists():
        _ready_path(cache_dir).unlink()

    if is_main_process:
        print(f"==> Building preprocessed SFT cache: {cache_dir}")
        print(f"    samples={len(rows)} num_proc={preprocess_num_proc}")
        t0 = time.time()
        dataset = build_tokenized_dataset(rows, tokenizer, num_proc=preprocess_num_proc)
        manifest = build_manifest(
            train_path=train_path,
            model_path=model_path,
            max_seq_length=max_seq_length,
            max_samples=max_samples,
            num_samples=len(rows),
            tokenizer_eos_token=tokenizer.eos_token,
        )
        save_preprocessed_cache(dataset, cache_dir, manifest)
        print(f"    saved in {time.time() - t0:.1f}s")
    else:
        wait_for_cache(cache_dir)

    return load_preprocessed_cache(cache_dir), cache_dir, False


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0
