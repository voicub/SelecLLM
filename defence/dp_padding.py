"""defence/dp_padding.py — (epsilon,delta)-DP Gaussian packet-padding (paper §5, Eqs 5-7).
sigma >= delta_l * sqrt(2*ln(1.25/delta)) / epsilon
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import math, numpy as np
from dataclasses import dataclass
from typing import List
from config import CFG
from utils.logger import get_logger
log = get_logger("DEFENCE")

@dataclass
class PaddingStats:
    n_packets: int; original_bytes: int; padded_bytes: int
    overhead_fraction: float; sigma: float; epsilon: float; delta: float
    def __str__(self):
        return (f"DP-Pad(eps={self.epsilon}, delta={self.delta:.0e}) "
                f"sigma={self.sigma:.0f}B overhead={self.overhead_fraction:.1%}")

class DPPaddingDefence:
    def __init__(self, epsilon=CFG.dp_epsilon, delta=CFG.dp_delta,
                 sensitivity=CFG.dp_sensitivity_bytes, granularity=CFG.dp_pad_granularity,
                 seed=CFG.random_seed):
        self.epsilon=epsilon; self.delta=delta; self.sensitivity=sensitivity
        self.granularity=granularity; self._rng=np.random.default_rng(seed)
        self.sigma = sensitivity * math.sqrt(2*math.log(1.25/delta)) / epsilon
        log.info(f"DP-Pad eps={epsilon} delta={delta:.0e} sens={sensitivity}B => sigma={self.sigma:.1f}B")

    def pad(self, sz: int) -> int:
        eta=self._rng.normal(0.0, self.sigma)
        padded=sz+max(0,math.ceil(eta))
        rem=padded%self.granularity
        return padded+(self.granularity-rem) if rem else padded

    def pad_trace(self, sizes: List[int]) -> List[int]: return [self.pad(s) for s in sizes]

    def expected_overhead_fraction(self, mean_payload=5000.0) -> float:
        return (self.sigma/math.sqrt(2*math.pi))/mean_payload

    def analyse_trace(self, sizes: List[int]) -> PaddingStats:
        padded=self.pad_trace(sizes); ot=sum(sizes); pt=sum(padded)
        return PaddingStats(len(sizes),ot,pt,(pt-ot)/max(1,ot),self.sigma,self.epsilon,self.delta)

    def privacy_guarantee(self) -> str:
        return (f"({self.epsilon:.2f},{self.delta:.0e})-DP: "
                f"sensitivity={self.sensitivity}B sigma={self.sigma:.1f}B "
                f"prob-ratio-bound=e^{self.epsilon:.2f}={math.exp(self.epsilon):.2f}x")
