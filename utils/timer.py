"""utils/timer.py — latency measurement context manager."""
import time
from dataclasses import dataclass, field
from typing import List

@dataclass
class LatencyTimer:
    _records: List[tuple] = field(default_factory=list)
    def record(self, stage: str, ms: float): self._records.append((stage, ms))
    def total_ms(self) -> float: return sum(ms for _, ms in self._records)
    def summary(self) -> dict:
        from collections import defaultdict
        agg = defaultdict(list)
        for s, ms in self._records: agg[s].append(ms)
        return {s: sum(v)/len(v) for s, v in agg.items()}

class StageTimer:
    def __init__(self, stage, acc=None):
        self.stage = stage; self.acc = acc; self.elapsed_ms = 0.0
    def __enter__(self): self._t0 = time.perf_counter(); return self
    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter()-self._t0)*1000.0
        if self.acc: self.acc.record(self.stage, self.elapsed_ms)
