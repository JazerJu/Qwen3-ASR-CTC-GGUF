"""Minimal SRT writer for [(unit, start_s, end_s)] triples (no dependencies)."""


def _ts(t: float) -> str:
    ms = max(int(round(t * 1000)), 0)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(units, path: str, max_units_per_line: int = 14, gap: float = 0.4):
    lines, idx, buf, start = [], 0, [], None
    prev_end = 0.0

    def flush():
        nonlocal idx, buf
        if not buf:
            return
        idx += 1
        text = "".join(u[0] for u in buf)
        lines.append(f"{idx}\n{_ts(buf[0][1])} --> {_ts(buf[-1][2])}\n{text}\n")
        buf.clear()

    for u in units:
        if buf and (u[1] - prev_end > gap or len(buf) >= max_units_per_line):
            flush()
        if not buf:
            start = u[1]
        buf.append(u)
        prev_end = u[2]
    flush()

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
