# src/hqp/common/contracts.py
# -----------------------------------------------------------------------------
# Pydantic models describing minimal “contracts” for common rows/keys.
# Useful for validating ad-hoc ingestion or for tight, typed adapters.
# Not required for the main CLI path, but handy in tests or prototyping.
# -----------------------------------------------------------------------------
from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, StrictFloat, StrictInt, StrictStr, validator

SEED: int = 42  # central place to share a default seed for reproducibility


class RaceKey(BaseModel):
    """Single-key contract with uppercase normalization."""

    race_id: StrictStr

    @validator("race_id")
    def _norm(cls, v: str) -> str:  # noqa: N805
        v = str(v).strip()
        if not v:
            raise ValueError("race_id is empty")
        return v.upper()


class HorseKey(BaseModel):
    """Single-key contract with uppercase normalization."""

    horse_id: StrictStr

    @validator("horse_id")
    def _norm(cls, v: str) -> str:  # noqa: N805
        v = str(v).strip()
        if not v:
            raise ValueError("horse_id is empty")
        return v.upper()


class EntryKey(BaseModel):
    """Composite-key contract for (race_id, horse_id)."""

    race_id: StrictStr
    horse_id: StrictStr

    @validator("race_id", "horse_id", pre=True)
    def _norm(cls, v: str) -> str:  # noqa: N805
        v = str(v).strip()
        if not v:
            raise ValueError("key component is empty")
        return v.upper()


class RaceRow(BaseModel):
    """
    Minimal canonical runner row; extend downstream as needed.

    Note: Strict types ensure early failure on malformed fields;
    validators normalize whitespace/upper-casing and coerce numerics safely.
    """

    # keys
    race_id: StrictStr = Field(..., description="Unique race identifier")
    horse_id: StrictStr = Field(..., description="Unique horse identifier")

    # core fields
    race_datetime: datetime
    course: StrictStr
    horse_name: StrictStr

    # numerics (optional)
    draw: StrictInt | None = None
    age: StrictInt | None = None
    weight_lbs: StrictInt | None = None
    official_rating: StrictInt | None = None
    n_runners: StrictInt | None = None
    distance_yards: StrictInt | None = None

    # market / result (optional)
    sp: StrictFloat | None = None
    position: StrictInt | None = None

    # categoricals / text (optional)
    going: StrictStr | None = None
    sex: Literal["M", "F", "G", "C", "H"] | None = None
    trainer: StrictStr | None = None
    jockey: StrictStr | None = None

    @validator("race_id", "horse_id", pre=True)
    def _norm_keys(cls, v: str) -> str:  # noqa: N805
        return str(v).strip().upper()

    @validator("course", "horse_name", pre=True)
    def _strip(cls, v: str) -> str:  # noqa: N805
        return str(v).strip()

    @validator(
        "draw",
        "age",
        "weight_lbs",
        "official_rating",
        "n_runners",
        "distance_yards",
        "position",
        pre=True,
    )
    def _int_or_none(cls, v: object) -> int | None:  # noqa: N805
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return None
            return int(s)
        if isinstance(v, bool):
            return int(v)  # avoid silent bool→int surprises
        if isinstance(v, (int, float)):
            if isinstance(v, float) and v != v:  # NaN
                return None
            return int(v)
        try:
            return int(str(v).strip())
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Cannot coerce {v!r} to int") from e

    @validator("sp", pre=True)
    def _float_or_none(cls, v: object) -> float | None:  # noqa: N805
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return None
            return float(s)
        if isinstance(v, (int, float)):
            if isinstance(v, float) and v != v:  # NaN
                return None
            return float(v)
        try:
            return float(str(v).strip())
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Cannot coerce {v!r} to float") from e
