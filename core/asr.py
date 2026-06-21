"""core/asr.py — ASR simulator (Whisper-medium profile).

Calibration targets (paper §4.2 and Table 3):
  WER = 8.71% on MuST-C English
  Latency = N(134ms, 22ms) — pure model inference, no RTF term
  Correct token confidence: N(0.88, 0.07) clipped [0.74, 1.0]
    → min_confidence on clean 12-word sentence ≈ 0.76  (above tau_c=0.72)
  Error token confidence:   N(0.40, 0.10) clipped [0.10, 0.60]
    → one error drops min_confidence to ~0.35  (below tau_c=0.72)
  Combined: ~28% of utterances have at least one error AND conf<tau_c
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from dataclasses import dataclass, field
from typing import List
from config import CFG
from utils.logger import get_logger
log = get_logger("ASR")
_RNG = np.random.default_rng(CFG.random_seed)

_SUBS = {
    "the":["a","this","thee"], "and":["an","in","end"], "to":["the","too","two"],
    "of":["off","a","or"], "is":["it","in","as"], "that":["than","this","hat"],
    "for":["four","far","or"], "are":["our","or","air"], "have":["has","half","gave"],
    "not":["now","no","knot"], "this":["the","his","these"], "very":["every","vary"],
    "really":["realy","rely"], "project":["protect","proect"], "system":["sistem","custom"],
    "meeting":["meating","beating"], "absolutely":["absolutly","absoluely"],
    "wonderful":["wonderfull","wonderfl"], "fantastic":["fantatsic","fantistic"],
    "frustrated":["frustrted","frustated"], "terrible":["terribl","terrble"],
    "conference":["conferance","confrence"], "results":["rezults","reults"],
    "network":["netwerk","netwrok"], "document":["documant","docuemnt"],
}

@dataclass
class ASRResult:
    tokens: List[str]; confidences: List[float]
    text: str; duration_ms: float; latency_ms: float

    @property
    def min_confidence(self):
        return float(np.min(self.confidences)) if self.confidences else 1.0
    @property
    def mean_confidence(self):
        return float(np.mean(self.confidences)) if self.confidences else 1.0
    @property
    def byte_size(self):
        return len(self.text.encode()) + 48


class ASRSimulator:
    """
    Whisper-medium ASR simulator with calibrated confidence scores.

    Confidence is bimodal:
      Correct tokens:  N(0.88, 0.07) clipped [0.74, 1.0]  → rarely below tau_c=0.72
      Error tokens:    N(0.40, 0.10) clipped [0.10, 0.60]  → always below tau_c=0.72

    This means min_confidence drops clearly below 0.72 when any error is present,
    matching the paper\'s ~28.4% routing rate (= fraction of utterances with >=1 error
    that also exceed PPL threshold).
    """
    def __init__(self, wer=CFG.asr_base_wer, lang="en"):
        self.wer = wer
        self.lang = lang

    def transcribe(self, text: str, audio_duration_ms: float = 1000.0) -> ASRResult:
        tokens = text.split()
        if not tokens:
            return ASRResult([], [], text, audio_duration_ms, 0.0)

        out_tok, out_conf = [], []
        for tok in tokens:
            if _RNG.random() < self.wer:
                etype = _RNG.choice(["sub","del","ins"], p=[0.60, 0.25, 0.15])
                if etype == "sub":
                    alt = _SUBS.get(tok.lower())
                    out_tok.append(_RNG.choice(alt) if alt else self._corrupt(tok))
                    # Error tokens: low confidence, clearly below tau_c=0.72
                    out_conf.append(float(np.clip(_RNG.normal(0.40, 0.10), 0.10, 0.60)))
                elif etype == "del":
                    pass   # token silently dropped
                else:      # insertion of spurious token
                    out_tok.append(tok)
                    out_conf.append(float(np.clip(_RNG.normal(0.88, 0.07), 0.74, 1.0)))
                    out_tok.append(_RNG.choice(["um","uh","the","a","and"]))
                    # Inserted spurious tokens also have low confidence
                    out_conf.append(float(np.clip(_RNG.normal(0.38, 0.08), 0.10, 0.55)))
            else:
                out_tok.append(tok)
                # Correct tokens: high confidence, above tau_c=0.72
                out_conf.append(float(np.clip(_RNG.normal(0.88, 0.07), 0.74, 1.0)))

        # Latency: pure model inference time (paper Table 5: 134ms mean)
        lat = float(np.clip(_RNG.normal(CFG.asr_latency_mean_ms,
                                        CFG.asr_latency_std_ms), 60.0, 280.0))
        hyp = " ".join(out_tok)
        log.debug(f"\'{hyp[:50]}\' conf_min={min(out_conf, default=1):.2f} lat={lat:.0f}ms")
        return ASRResult(out_tok, out_conf, hyp, audio_duration_ms, lat)

    @staticmethod
    def _corrupt(w: str) -> str:
        if len(w) <= 2: return w
        i = _RNG.integers(1, len(w) - 1)
        c = list(w); c[i] = _RNG.choice(list("aeioutnrsl"))
        return "".join(c)
