"""core/vad.py — energy-based Voice Activity Detector."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from dataclasses import dataclass
from typing import List
from config import CFG
from utils.logger import get_logger
log = get_logger("VAD")

@dataclass
class VADSegment:
    start: int; end: int; sr: int
    @property
    def duration_ms(self): return (self.end-self.start)/self.sr*1000.0
    def audio(self, wav): return wav[self.start:self.end]

class EnergyVAD:
    def __init__(self, sr=CFG.sample_rate, frame_ms=20.0,
                 thresh=CFG.vad_energy_threshold, min_ms=CFG.vad_min_duration_ms,
                 hangover_ms=100.0):
        self.sr=sr; self.fsize=int(frame_ms/1000*sr); self.thresh=thresh
        self.min_samp=int(min_ms/1000*sr); self.hangover=max(1,int(hangover_ms/frame_ms))

    def detect(self, wav: np.ndarray) -> List[VADSegment]:
        nf = len(wav)//self.fsize
        if nf == 0: return [VADSegment(0,len(wav),self.sr)]
        E = np.array([np.sqrt(np.mean(wav[i*self.fsize:(i+1)*self.fsize]**2)) for i in range(nf)])
        active = E > self.thresh
        sm, hang = np.zeros(nf,bool), 0
        for i,a in enumerate(active):
            if a: hang=self.hangover; sm[i]=True
            elif hang>0: sm[i]=True; hang-=1
        segs, in_sp, s0 = [], False, 0
        for i,f in enumerate(sm):
            if f and not in_sp: in_sp=True; s0=i*self.fsize
            elif not f and in_sp:
                in_sp=False; e=i*self.fsize
                if e-s0>=self.min_samp: segs.append(VADSegment(s0,e,self.sr))
        if in_sp:
            e=nf*self.fsize
            if e-s0>=self.min_samp: segs.append(VADSegment(s0,e,self.sr))
        if not segs: segs=[VADSegment(0,len(wav),self.sr)]
        log.debug(f"{len(segs)} segment(s) detected"); return segs
