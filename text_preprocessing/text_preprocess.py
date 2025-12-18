import re
from typing import Iterable

import pandas as pd


def normalize_text(text: str) -> str:
    """Lowercase, strip, and collapse whitespace for simple normalization."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _coerce_skills(skills: Iterable[str]) -> Iterable[str]:
    """Ensure skills is an iterable of tokens (split comma-separated strings if needed)."""
    if isinstance(skills, str):
        return [s.strip() for s in skills.split(",") if s.strip()]
    return skills or []


def build_combined_text(skills: Iterable[str], desc: str, category: str) -> str:
    safe_skills = _coerce_skills(skills)
    skills_text = " ".join(safe_skills) if safe_skills else ""
    parts = [skills_text, desc or "", category or ""]
    return normalize_text(" ".join(parts))


def add_combined_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combined_text"] = df.apply(
        lambda row: build_combined_text(row.get("skills", []), row.get("desc", ""), row.get("category", "")),
        axis=1,
    )
    return df
