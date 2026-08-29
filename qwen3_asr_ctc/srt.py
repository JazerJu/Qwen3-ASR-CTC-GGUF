"""极简 SRT 写出（无第三方依赖）。"""
from __future__ import annotations

from pathlib import Path


def _ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(units, path: str | Path) -> None:
    """units: [(文本, 起秒, 止秒)]"""
    lines = []
    for i, (text, start, end) in enumerate(units, 1):
        lines += [str(i), f"{_ts(start)} --> {_ts(end)}", text, ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
