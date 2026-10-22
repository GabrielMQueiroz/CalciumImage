# -*- coding: utf-8 -*-
"""Stage manifest and deterministic hashing for resumable pipeline execution."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io import InputSpec

logger = logging.getLogger("decovo")

MANIFEST_FILENAME = "decovo_manifest.json"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(*parts: Any) -> str:
    """Compute a 16-char SHA-256 hex digest over arbitrary JSON-serializable objects."""
    h = hashlib.sha256()
    for part in parts:
        h.update(_canonical_json(part).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


def input_files_signature(spec: InputSpec) -> Dict[str, Any]:
    """Build a signature of input files (basename, size, mtime_ns) for cache invalidation."""
    file_entries = []
    for f in spec.files:
        try:
            st = os.stat(f)
            file_entries.append({
                "name": os.path.basename(f),
                "size": int(st.st_size),
                "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
            })
        except OSError:
            file_entries.append({"name": os.path.basename(f), "size": -1, "mtime_ns": -1})
    return {
        "kind": spec.kind,
        "n_frames": int(spec.n_frames),
        "frame_shape": list(spec.frame_shape),
        "dtype": str(spec.dtype),
        "files": file_entries,
    }


class StageCache:
    """Reads and writes `decovo_manifest.json` in `work_dir` to support stage skipping on resume."""

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir).resolve()
        self.manifest_path = self.work_dir / MANIFEST_FILENAME
        self.data: Dict[str, Any] = {"version": 2, "stages": {}}
        self._load()

    def _load(self) -> None:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and "stages" in loaded:
                    self.data = loaded
            except Exception as e:
                logger.warning("Could not read manifest %s (%s); starting fresh.", self.manifest_path, e)

    def save(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self.manifest_path)

    def clear(self) -> None:
        self.data = {"version": 2, "stages": {}}
        if self.manifest_path.exists():
            try:
                self.manifest_path.unlink()
            except OSError:
                pass

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return self.work_dir / path

    def _relativize(self, p: str | Path) -> str:
        path = Path(p).resolve()
        try:
            return str(path.relative_to(self.work_dir))
        except ValueError:
            return str(path)

    def is_valid(self, stage: str, expected_hash: str) -> bool:
        """Return True if `stage` has `expected_hash` and all its recorded output files exist."""
        entry = self.data.get("stages", {}).get(stage)
        if not entry or entry.get("hash") != expected_hash:
            return False
        outputs: List[str] = entry.get("outputs", [])
        if not outputs:
            return False
        for out in outputs:
            if not self._resolve_path(out).exists():
                return False
        return True

    def get_outputs(self, stage: str) -> List[str]:
        entry = self.data.get("stages", {}).get(stage, {})
        return [str(self._resolve_path(p)) for p in entry.get("outputs", [])]

    def get_meta(self, stage: str) -> Dict[str, Any]:
        entry = self.data.get("stages", {}).get(stage, {})
        return dict(entry.get("meta", {}))

    def get_hash(self, stage: str) -> Optional[str]:
        entry = self.data.get("stages", {}).get(stage, {})
        return entry.get("hash")

    def record(
        self,
        stage: str,
        stage_hash: str,
        outputs: Iterable[str | Path],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data.setdefault("stages", {})[stage] = {
            "hash": stage_hash,
            "outputs": [self._relativize(p) for p in outputs],
            "meta": meta or {},
        }
        self.save()
