"""PromptDataset with on-disk cache for length filtering."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from datasets import load_from_disk
from loguru import logger
from skyrl.train.dataset.dataset import PromptDataset


def _cache_root() -> Path:
    return Path(
        os.environ.get(
            "SKYRL_PROMPT_DATASET_CACHE",
            "data/rl/skyrl_parquet/.prompt_filter_cache",
        )
    )


def _cache_key(datasets: list[str], tokenizer_name: str, max_prompt_length: int) -> str:
    payload = {
        "datasets": [str(Path(p).resolve()) for p in datasets],
        "mtimes": [os.path.getmtime(p) if os.path.isfile(p) else 0 for p in datasets],
        "tokenizer": tokenizer_name,
        "max_prompt_length": max_prompt_length,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class CachedPromptDataset(PromptDataset):
    """Cache filtered parquet rows so RL restarts skip apply_chat_template filtering."""

    def __init__(
        self,
        datasets,
        tokenizer,
        max_prompt_length: int,
        num_workers: int = 8,
        prompt_key: str = "prompt",
        env_class_key: str = "env_class",
        cache_dir: str | None = None,
        force_recache: bool = False,
        disable_cache: bool = False,
    ):
        self._cache_dir = Path(cache_dir) if cache_dir else _cache_root()
        self._force_recache = force_recache or os.environ.get("SKYRL_PROMPT_DATASET_FORCE_RECACHE", "0") == "1"
        self._disable_cache = disable_cache or os.environ.get("SKYRL_PROMPT_DATASET_DISABLE_CACHE", "0") == "1"
        self._tokenizer_name = getattr(tokenizer, "name_or_path", str(tokenizer))
        super().__init__(
            datasets=datasets,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            num_workers=num_workers,
            prompt_key=prompt_key,
            env_class_key=env_class_key,
        )

    def _cache_path(self) -> Path | None:
        if self._disable_cache:
            return None
        key = _cache_key(self.datasets, self._tokenizer_name, self.max_prompt_length)
        return self._cache_dir / key

    def _read_files_and_tokenize(self):
        cache_path = self._cache_path()
        if cache_path is not None and cache_path.is_dir() and not self._force_recache:
            logger.info(f"Loading filtered prompt dataset from cache: {cache_path}")
            self.dataframe = load_from_disk(str(cache_path))
            logger.info(f"Filtered dataset size (cached): {len(self.dataframe)}")
            return

        super()._read_files_and_tokenize()

        if cache_path is None:
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        logger.info(f"Saving filtered prompt dataset cache: {cache_path}")
        self.dataframe.save_to_disk(str(temp_path))
        if cache_path.exists():
            shutil.rmtree(cache_path)
        temp_path.rename(cache_path)
