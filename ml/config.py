"""Shared project paths and lightweight limits."""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_RESUME_BYTES = 10 * 1024 * 1024
REQUIRED_SKILL_WEIGHT = 2.0
PREFERRED_SKILL_WEIGHT = 1.0
