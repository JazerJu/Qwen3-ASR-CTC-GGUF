#!/usr/bin/env python3
"""Step 1 gate: ONNX encoder vs PyTorch on real audio at lengths != trace input.

Catches the mask-materialization class of bugs (QWEN3-CTC-导出与推理指示.md §1.3):
trace used mel 2800; validation runs 800 / ~1900 / 2500 frames.
Gate: cosine >= 0.999 on the valid frame range for each length.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import soundfile as sf
import torch

import qwen3_compat  # noqa: F401
from qwen_asr import Qwen3ASRModel
from transformers import WhisperFeatureExtractor

from modeling_ctc import patch_qwen3_attention_mask
from qwen3_compat import repair_positional_embedding, valid_frames

QWEN3_DIR = "/data/推理框架/asr-onnx/Qwen3-ASR-HF"
MEL_FRAMES = 3000

FE = WhisperFeatureExtractor.from_pretrained(QWEN3_DIR)


def mel_of(audio):
    return FE(audio, sampling_rate=16000, padding=False, return_tensors="np").input_features[0]


def main():
    import onnxruntime as ort

    sources = ["/tmp/seg5s.wav", "/tmp/sg_ds.wav"]
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "30", "-t", "12", "-i",
             "/data/其他模型/小型语言模型/自建fq节点.mp4", "-ar", "16000", "-ac", "1", f"{td}/12s.wav"],
            check=True, capture_output=True)
        sources.append(f"{td}/12s.wav")

        patch_qwen3_attention_mask()
        m = Qwen3ASRModel.from_pretrained(QWEN3_DIR, dtype=torch.float32, device_map=None)
        tower = m.model.thinker.audio_tower.eval()
        repair_positional_embedding(tower)

        sess = ort.InferenceSession(str(HERE / "model" / "Qwen3-ASR-Encoder.fp32.onnx"),
                                     providers=["CPUExecutionProvider"])

        ok = True
        for wav in sources:
            audio, _ = sf.read(wav, dtype="float32")
            mel = mel_of(audio)
            T = mel.shape[1]
            with torch.no_grad():
                ref = tower(torch.from_numpy(mel), feature_lens=torch.tensor([T]))[0].numpy()

            padded = np.zeros((128, MEL_FRAMES), dtype=np.float32)
            padded[:, :T] = mel
            hyp = sess.run(None, {"input_features": padded,
                                  "feature_length": np.array([T], dtype=np.int64)})[0]
            valid = int(valid_frames(torch.tensor([T]))[0])
            ref_v, hyp_v = ref[:valid], hyp[:valid]
            cos = float(ref_v.ravel() @ hyp_v.ravel() / (np.linalg.norm(ref_v) * np.linalg.norm(hyp_v)))
            gate = cos >= 0.999
            ok &= gate
            print(f"mel={T:5d} -> out={valid:3d}f  cosine={cos:.6f}  {'PASS' if gate else 'FAIL'}")

        print("GATE:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
