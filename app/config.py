"""Settings read from environment variables, with defaults for local work."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    path = Path(os.getenv("DATA_DIR", "data"))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=_data_dir)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_path(self) -> Path:
        return self.data_dir / "processed" / "paysim.parquet"


settings = Settings()
