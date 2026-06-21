"""core/tts.py — TTS simulator (VITS profile)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from dataclasses import dataclass
from config import CFG
from utils.logger import get_logger
log = get_logger("TTS")
_RNG = np.random.default_rng(CFG.random_seed+3)
_RATES={"en":150,"de":130,"fr":160,"es":170,"it":165,"nl":140,"pt":155,"ro":145,"ru":140,"zh":200}

@dataclass
class TTSResult:
    text: str; lang: str; audio: "np.ndarray"; sample_rate: int
    audio_duration_ms: float; latency_ms: float
    @property
    def byte_size(self): return len(self.audio)*2
    @property
    def n_packets(self): return max(1,self.byte_size//CFG.mtu_bytes)

class TTSSimulator:
    def __init__(self, lang="en", sr=CFG.sample_rate):
        self.lang=lang; self.sr=sr; self.wpm=_RATES.get(lang,150)
    def synthesise(self, text: str) -> TTSResult:
        n_words=max(1,len(text.split()))
        dur_s=float(np.clip(n_words/self.wpm*60+_RNG.normal(0,0.1),0.2,60.0))
        n=int(dur_s*self.sr); t=np.linspace(0,dur_s,n,dtype=np.float32)
        r=self.wpm/60
        audio=((0.5*np.sin(2*np.pi*180*t)+0.2*np.sin(2*np.pi*360*t)+
                0.1*np.sin(2*np.pi*720*t))*(0.5+0.5*np.sin(2*np.pi*r*1.5*t))
               +_RNG.normal(0,0.01,n).astype(np.float32)).astype(np.float32)
        lat=float(np.clip(_RNG.normal(CFG.tts_latency_mean_ms,CFG.tts_latency_std_ms),25,120))
        log.debug(f"{n_words}w → {dur_s*1000:.0f}ms audio lat={lat:.0f}ms")
        return TTSResult(text,self.lang,audio,self.sr,dur_s*1000,lat)
