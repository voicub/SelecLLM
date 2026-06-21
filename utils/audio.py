"""utils/audio.py — synthetic speech waveform generator."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from config import CFG

_RNG = np.random.default_rng(CFG.random_seed)
_SYL_RATES = {"en":4.5,"de":4.2,"fr":5.5,"es":5.8,"it":5.6,
               "nl":4.3,"pt":5.4,"ro":4.7,"ru":4.1,"zh":4.0}

def synthetic_utterance(text: str, lang: str = "en",
                         sr: int = CFG.sample_rate) -> "np.ndarray":
    words = text.split()
    rate = _SYL_RATES.get(lang, 4.5)
    dur_s = float(np.clip(max(1, int(len(words)*1.8)) / rate, 0.5, 30.0))
    n = int(dur_s * sr)
    t = np.linspace(0, dur_s, n, dtype=np.float32)
    sig = (0.4*np.sin(2*np.pi*180*t) + 0.2*np.sin(2*np.pi*360*t)
           + 0.1*np.sin(2*np.pi*720*t))
    sig *= (0.5 + 0.5*np.sin(2*np.pi*rate*t))
    sig += _RNG.normal(0, 0.02, n).astype(np.float32)
    pk = np.max(np.abs(sig))
    return sig / pk if pk > 0 else sig

def audio_duration_ms(audio: "np.ndarray", sr: int = CFG.sample_rate) -> float:
    return len(audio) / sr * 1000.0
