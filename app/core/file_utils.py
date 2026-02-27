from __future__ import annotations

import re
from pathlib import Path


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    """
    Prevent path traversal and keep filenames filesystem-friendly.
    """
    name = Path(filename).name
    name = _FILENAME_SAFE_RE.sub("_", name).strip("._")
    if not name:
        return "upload"
    return name[:200]

