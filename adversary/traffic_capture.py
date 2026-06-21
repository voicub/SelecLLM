"""adversary/traffic_capture.py
42-dim feature vector from encrypted packet traces.

Features designed to be language-discriminative without seeing payload:
  - 12 packet-size statistics
  - 8 inter-arrival time statistics
  - 6 burst statistics (burst count correlates with TTS speaking rate)
  - 16-bin log-spaced size histogram (ASR/MT packet sizes differ by language)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from typing import List, Tuple
from config import CFG

_N_BINS = 16
_SIZE_BINS = np.logspace(np.log10(64), np.log10(16384), _N_BINS + 1)


def extract_features(packets) -> np.ndarray:
    """Return 42-dim float32 feature vector: 12 size + 8 IAT + 6 burst + 16 hist."""
    if not packets:
        return np.zeros(42, dtype=np.float32)

    sizes = np.array([p.byte_size    for p in packets], dtype=np.float32)
    times = np.array([p.timestamp_ms for p in packets], dtype=np.float32)

    # ── Packet size features (12) ─────────────────────────────────────────────
    sf = np.array([
        np.mean(sizes),
        np.std(sizes) + 1e-6,
        np.min(sizes),
        np.max(sizes),
        np.percentile(sizes, 25),
        np.percentile(sizes, 50),
        np.percentile(sizes, 75),
        np.percentile(sizes, 90),
        float(len(sizes)),
        float(np.sum(sizes)),                          # total bytes → utterance length
        float(np.sum(sizes > CFG.mtu_bytes * 0.8)),   # large-packet count (TTS)
        float(np.sum(sizes < 300)),                    # small-packet count (ASR text)
    ], dtype=np.float32)

    # ── Inter-arrival time features (8) ──────────────────────────────────────
    # IAT gap between ASR packet and TTS burst differs by language speaking rate
    if len(times) > 1:
        iats = np.diff(times)
        iatf = np.array([
            np.mean(iats),
            np.std(iats) + 1e-6,
            np.min(iats),
            np.max(iats),
            np.percentile(iats, 25),
            np.percentile(iats, 75),
            float(np.sum(iats < 5.0)),     # rapid bursts  (TTS chunks)
            float(np.sum(iats > 80.0)),    # stage-boundary gaps (ASR→MT→TTS)
        ], dtype=np.float32)
    else:
        iatf = np.zeros(8, dtype=np.float32)

    # ── Burst features (6) ────────────────────────────────────────────────────
    # TTS burst size correlates with speaking rate → language-specific
    bsizes, blens = _bursts(times, sizes, gap=15.0)
    if bsizes:
        bf = np.array([
            float(len(bsizes)),
            float(np.mean(bsizes)),
            float(np.std(bsizes) + 1e-6),
            float(np.mean(blens)),
            float(np.max(bsizes)),
            float(np.sum(bsizes)) / (float(np.sum(sizes)) + 1e-6),
        ], dtype=np.float32)
    else:
        bf = np.zeros(6, dtype=np.float32)

    # ── Size histogram (16) ──────────────────────────────────────────────────
    # Different languages produce characteristically different ASR/TTS sizes
    hist, _ = np.histogram(sizes, bins=_SIZE_BINS)
    hf = (hist / (len(sizes) + 1e-6)).astype(np.float32)

    feat = np.concatenate([sf, iatf, bf, hf])
    assert len(feat) == 42, f"Feature length {len(feat)}"
    return feat


def _bursts(times, sizes, gap=15.0):
    if len(times) == 0:
        return [], []
    bs, bl = [], []
    cs, cl = float(sizes[0]), 1
    for i in range(1, len(times)):
        if times[i] - times[i-1] > gap:
            bs.append(cs); bl.append(cl)
            cs, cl = float(sizes[i]), 1
        else:
            cs += float(sizes[i]); cl += 1
    bs.append(cs); bl.append(cl)
    return bs, bl


def build_feature_matrix(results):
    X, yl, ys, ysp = [], [], [], []
    for r in results:
        X.append(extract_features(r.packets))
        yl.append(r.source_lang)
        ys.append(getattr(r, "sentiment", "neutral"))
        ysp.append(getattr(r, "speaker_id", 0))
    return np.array(X, dtype=np.float32), yl, ys, ysp
