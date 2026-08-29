"""紧凑词表的字节级反解。

**不能按字符串拼。** 紧凑词表里保留了 89 个字节原语做兜底，一个汉字可能由多个
字节 token 拼成，`"".join(pieces)` 会得到乱码。tokens.txt 存的是每个紧凑 id 对应
的**原始字节的 base64**，解码时拼 bytes 再统一 decode。
"""
from __future__ import annotations

import base64
from pathlib import Path


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 那套 byte<->unicode 映射，Qwen3 的 BPE 用的就是它。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def load_tokens(path: str | Path) -> dict[int, bytes]:
    """tokens.txt（`<base64 原始字节>\\t<紧凑 id>`）-> {id: bytes}"""
    out: dict[int, bytes] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            b64, idx = line.rsplit("\t", 1)
            out[int(idx)] = base64.b64decode(b64)
    return out


def decode_bytes(id2bytes: dict[int, bytes], ids) -> str:
    return b"".join(id2bytes[int(i)] for i in ids).decode("utf-8", errors="replace")
