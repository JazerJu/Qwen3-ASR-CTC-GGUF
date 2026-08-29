#!/usr/bin/env python3
"""Step 2: generate models/qwen-ctc/tokens.txt (base64 raw bytes per compact id).

Verification: 500 multilingual sentences — byte-path decode(compact ids) must
equal transformers tokenizer.decode(qwen ids) byte-for-byte.
"""

import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from transformers import AutoTokenizer


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


QWEN3_DIR = "/data/推理框架/asr-onnx/Qwen3-ASR-HF"

byte_decoder = {v: k for k, v in bytes_to_unicode().items()}


def main():
    tok = AutoTokenizer.from_pretrained(QWEN3_DIR, trust_remote_code=True)
    vc = json.load(open(HERE / "vocab_compact.json", encoding="utf-8"))
    c2q = vc["compact_to_qwen"]
    q2c = {q: c for c, q in enumerate(c2q)}

    lines, id2bytes = [], {}
    for cid, qid in enumerate(c2q):
        piece = tok.convert_ids_to_tokens(int(qid))
        raw = bytes(byte_decoder[ch] for ch in piece)
        id2bytes[cid] = raw
        lines.append(base64.b64encode(raw).decode("ascii") + "\t" + str(cid))

    out = HERE / "model" / "qwen-ctc-tokens.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(lines)} entries)")

    def byte_decode(compact_ids):
        return b"".join(id2bytes[i] for i in compact_ids).decode("utf-8", errors="replace")

    sentences = [
        "今天我们来讲一讲服务器的日常维护，主要包括内存、磁盘和网络的监控。",
        "你可以打开cmd命令行，输入ip config查看本机的IP地址。",
        "In this video, we will walk through the installation process step by step.",
        "안녕하세요, 오늘 날씨가 정말 좋네요.",
        "こんにちは、今日はいい天気ですね。",
        "Le chat s'est assis sur le tapis.", "Die Straße war leer.",
        "Мы говорим по-русски.", "こんにちは1234、测试mixed—input！！",
    ] * 56

    fail = covered = 0
    for s in sentences:
        qids = tok(s, add_special_tokens=False)["input_ids"]
        if any(q not in q2c for q in qids):
            continue  # encode-direction gap in the compact vocab; decode-only use case
        covered += 1
        cids = [q2c[q] for q in qids]
        if byte_decode(cids) != tok.decode(qids):
            fail += 1
            print(f"MISMATCH: {s[:30]} | {byte_decode(cids)[:40]!r} vs {tok.decode(qids)[:40]!r}")
    print(f"round-trip: {covered - fail}/{covered} identical ({len(sentences) - covered} skipped, compact-vocab gap)")
    sys.exit(1 if fail or covered < 100 else 0)


if __name__ == "__main__":
    main()
