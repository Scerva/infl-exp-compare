"""Loader for config/sources.yaml, the single source-catalog of truth."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES_YAML = PROJECT_ROOT / "config" / "sources.yaml"


@dataclass(frozen=True)
class SourceDef:
    source_family: str
    series_name: str
    series_id: str
    provider: str
    horizon_months: Any
    inflation_concept: Optional[str]
    frequency: str
    units: Optional[str]
    availability_start: Optional[str]
    download_method: Optional[str]
    release_timing_method: Optional[str]
    target_anchor: Optional[float]
    notes: str = ""
    group: str = ""


def load_source_catalog(path: Path = SOURCES_YAML) -> list[SourceDef]:
    raw = yaml.safe_load(path.read_text())
    out: list[SourceDef] = []
    for group, entries in raw.items():
        for e in entries:
            out.append(SourceDef(group=group, **e))
    return out


def get_source(series_id: str, path: Path = SOURCES_YAML) -> SourceDef:
    for s in load_source_catalog(path):
        if s.series_id == series_id:
            return s
    raise KeyError(f"series_id {series_id!r} not found in {path}")


def sources_by_provider(provider: str, path: Path = SOURCES_YAML) -> list[SourceDef]:
    return [s for s in load_source_catalog(path) if s.provider == provider]


def sources_by_family(source_family: str, path: Path = SOURCES_YAML) -> list[SourceDef]:
    return [s for s in load_source_catalog(path) if s.source_family == source_family]
