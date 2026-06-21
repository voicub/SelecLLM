"""core/mt.py — MT simulator (NLLB-200 1.3B profile)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from dataclasses import dataclass
from config import CFG
from utils.logger import get_logger
log = get_logger("MT")
_RNG = np.random.default_rng(CFG.random_seed+2)

_RATIOS = {("en","de"):1.08,("en","fr"):1.10,("en","es"):1.05,("en","it"):1.07,
            ("en","nl"):1.03,("en","pt"):1.09,("en","ro"):1.06,("en","ru"):0.92,
            ("en","zh"):0.62,("de","en"):0.93,("fr","en"):0.91,("es","en"):0.95}
_WMAP  = {"de":{"hello":"hallo","and":"und","the":"die","is":"ist","very":"sehr",
                 "not":"nicht","great":"grossartig","today":"heute"},
           "fr":{"hello":"bonjour","and":"et","the":"le","is":"est","very":"tres",
                 "not":"pas","great":"formidable","today":"aujourd'hui"},
           "es":{"hello":"hola","and":"y","the":"el","is":"es","very":"muy",
                 "not":"no","great":"genial","today":"hoy"}}

@dataclass
class MTResult:
    source_text: str; translated_text: str; src_lang: str; tgt_lang: str; latency_ms: float
    @property
    def byte_size(self): return len(self.translated_text.encode())+48

class MTSimulator:
    def __init__(self, src_lang="en", tgt_lang="de"):
        self.src=src_lang; self.tgt=tgt_lang
    def translate(self, text: str) -> MTResult:
        words=text.split(); ratio=_RATIOS.get((self.src,self.tgt),1.0)
        n_out=max(1,int(len(words)*ratio+_RNG.normal(0,1)))
        wmap=_WMAP.get(self.tgt,{})
        out=[wmap.get(w.lower(),w) for w in words]
        while len(out)<n_out: out.append(words[len(out)%max(1,len(words))])
        out=out[:n_out]
        lat=float(np.clip(_RNG.normal(CFG.mt_latency_mean_ms,CFG.mt_latency_std_ms),20,300)+len(words)*0.5)
        log.debug(f"{len(words)}→{n_out} words lat={lat:.0f}ms")
        return MTResult(text," ".join(out),self.src,self.tgt,lat)
