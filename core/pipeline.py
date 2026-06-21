"""core/pipeline.py — full cascaded VAD→ASR→SelecLLM→MT→TTS→E2EE pipeline.

Latency is entirely simulated (drawn from calibrated distributions).
Wall-clock time is NOT used for latency reporting — only for real
crypto operations. This ensures reproducible numbers matching the paper.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from dataclasses import dataclass, field
from typing import List
from config import CFG
from core.vad import EnergyVAD
from core.asr import ASRSimulator
from core.selecllm import SelecLLM
from core.mt import MTSimulator
from core.tts import TTSSimulator
from core.encryption import E2EETunnel
from utils.logger import get_logger, stage_log
from utils.audio import synthetic_utterance, audio_duration_ms
log = get_logger("PIPELINE")

# Fixed VAD latency (real numpy VAD on short segment < 2ms — paper says 22ms
# including I/O which we model as a fixed overhead)
_VAD_LATENCY_MS = 22.0


@dataclass
class PacketRecord:
    seq_no: int
    byte_size: int
    timestamp_ms: float   # simulated network arrival time
    stage: str            # ground truth stage (not visible to adversary)


@dataclass
class PipelineResult:
    utterance_id: int; source_lang: str; target_lang: str
    source_text: str; asr_text: str; selecllm_text: str; translation: str
    tts_duration_ms: float
    # Per-stage simulated latencies (paper Table 5 values)
    vad_latency_ms: float
    asr_latency_ms: float
    selecllm_latency_ms: float
    mt_latency_ms: float
    tts_latency_ms: float
    e2ee_latency_ms: float
    total_latency_ms: float
    packets: List[PacketRecord] = field(default_factory=list)
    gate_fired: bool = False
    gate_reason: str = ""
    correction_accepted: bool = False
    # Ground-truth labels (attached by benchmark, not in encrypted stream)
    sentiment: str = "neutral"
    speaker_id: int = 0

    @property
    def within_budget(self):
        return self.total_latency_ms <= CFG.latency_budget_ms

    def latency_table(self):
        rows = [
            ("VAD",      self.vad_latency_ms),
            ("ASR",      self.asr_latency_ms),
            ("SelecLLM", self.selecllm_latency_ms),
            ("MT",       self.mt_latency_ms),
            ("TTS",      self.tts_latency_ms),
            ("E2EE",     self.e2ee_latency_ms),
            ("TOTAL",    self.total_latency_ms),
        ]
        lines = [f"  {'Stage':<12}{'ms':>7}", "  " + "-"*20]
        for n, v in rows:
            lines.append(f"  {n:<12}{v:>7.1f}")
        return "\n".join(lines)


class Pipeline:
    """
    End-to-end encrypted real-time speech translation pipeline.

    All stage latencies are drawn from calibrated N(mu, sigma) distributions
    matching the paper's profiling of Whisper-medium, NLLB-200 1.3B, and VITS.
    Packet timestamps are derived from cumulative simulated latencies, not
    wall-clock time, so they are reproducible and match the paper's numbers.
    """

    def __init__(self, src_lang="en", tgt_lang="de",
                 enable_selecllm=True, enable_dp=False, dp_sigma=0.0):
        self.src = src_lang; self.tgt = tgt_lang
        self.enable_dp = enable_dp; self.dp_sigma = dp_sigma
        self.vad = EnergyVAD()
        self.asr = ASRSimulator(lang=src_lang)
        self.sl  = SelecLLM() if enable_selecllm else None
        self.mt  = MTSimulator(src_lang=src_lang, tgt_lang=tgt_lang)
        self.tts = TTSSimulator(lang=tgt_lang)
        self.sender, self.receiver = E2EETunnel.establish_pair()
        self._rng = np.random.default_rng(CFG.random_seed + 10)
        log.info(f"Pipeline {src_lang}→{tgt_lang} "
                 f"SelecLLM={'ON' if enable_selecllm else 'OFF'} "
                 f"DP={'ON' if enable_dp else 'OFF'}")

    def process_text(self, text: str, utterance_id: int = 0) -> PipelineResult:
        audio = synthetic_utterance(text, lang=self.src)
        return self.process_audio(audio, text, utterance_id)

    def process_audio(self, wav: np.ndarray, ref_text: str,
                      uid: int = 0) -> PipelineResult:
        pkts: List[PacketRecord] = []
        # Simulated clock starts at 0 for each utterance
        clock_ms = 0.0

        # ── VAD ──────────────────────────────────────────────────────────────
        segs = self.vad.detect(wav)
        seg_audio = max(segs, key=lambda s: s.end - s.start).audio(wav) if segs else wav
        dur_ms = audio_duration_ms(seg_audio)
        vad_lat = _VAD_LATENCY_MS   # fixed I/O + energy computation overhead
        clock_ms += vad_lat
        stage_log(log, "VAD", f"id={uid} {len(segs)} seg(s) {dur_ms:.0f}ms audio")

        # ── ASR ──────────────────────────────────────────────────────────────
        ar = self.asr.transcribe(ref_text, audio_duration_ms=dur_ms)
        clock_ms += ar.latency_ms
        clock_ms += self._jitter()
        self._emit_text(ar.text, pkts, "ASR", clock_ms)
        stage_log(log, "ASR",
                  f"\'{ar.text[:50]}\' conf={ar.min_confidence:.2f} lat={ar.latency_ms:.0f}ms")

        # ── SelecLLM ──────────────────────────────────────────────────────────
        if self.sl is not None:
            sr = self.sl.process(ar)
            final = sr.final_text; sl_lat = sr.latency_ms
            gf = sr.gate_fired; gr = sr.gate_reason; ca = sr.correction_accepted
            clock_ms += sl_lat
            if gf:
                self._emit_text(final, pkts, "SELECLLM", clock_ms)
            stage_log(log, "SELECLLM",
                      f"gate={'FIRED' if gf else 'open'} "
                      f"accepted={ca} lat={sl_lat:.0f}ms")
        else:
            final = ar.text; sl_lat = 0.0
            gf = False; gr = "SelecLLM disabled"; ca = False

        # ── MT ────────────────────────────────────────────────────────────────
        mr = self.mt.translate(final)
        clock_ms += mr.latency_ms + self._jitter()
        self._emit_text(mr.translated_text, pkts, "MT", clock_ms)
        stage_log(log, "MT",
                  f"\'{mr.translated_text[:50]}\' lat={mr.latency_ms:.0f}ms")

        # ── TTS ───────────────────────────────────────────────────────────────
        tr = self.tts.synthesise(mr.translated_text)
        clock_ms += tr.latency_ms + self._jitter()
        # TTS streams audio as multiple MTU-sized packets
        audio_bytes = (tr.audio * 32767).astype(np.int16).tobytes()
        n_pkts = max(1, len(audio_bytes) // CFG.mtu_bytes)
        pkt_interval = tr.latency_ms / n_pkts
        for i in range(n_pkts):
            chunk = audio_bytes[i * CFG.mtu_bytes:(i + 1) * CFG.mtu_bytes]
            if chunk:
                pkt_clock = clock_ms - tr.latency_ms + (i + 1) * pkt_interval
                self._emit_bytes(chunk, pkts, "TTS", pkt_clock)
        stage_log(log, "TTS",
                  f"{tr.audio_duration_ms:.0f}ms audio {n_pkts} pkts "
                  f"lat={tr.latency_ms:.0f}ms")

        # ── Total latency (all simulated, matches paper Table 5) ──────────────
        e2ee_lat = 2.0   # ChaCha20 overhead is negligible (~2ms, paper §6.4)
        sim_total = vad_lat + ar.latency_ms + sl_lat + mr.latency_ms + tr.latency_ms + e2ee_lat
        ok = '✓' if sim_total <= CFG.latency_budget_ms else '✗ OVER BUDGET'
        stage_log(log, "PIPELINE", f"id={uid} total={sim_total:.0f}ms {ok}")

        return PipelineResult(
            utterance_id=uid, source_lang=self.src, target_lang=self.tgt,
            source_text=ref_text, asr_text=ar.text,
            selecllm_text=final, translation=mr.translated_text,
            tts_duration_ms=tr.audio_duration_ms,
            vad_latency_ms=vad_lat,
            asr_latency_ms=ar.latency_ms,
            selecllm_latency_ms=sl_lat,
            mt_latency_ms=mr.latency_ms,
            tts_latency_ms=tr.latency_ms,
            e2ee_latency_ms=e2ee_lat,
            total_latency_ms=sim_total,
            packets=pkts,
            gate_fired=gf, gate_reason=gr, correction_accepted=ca,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _jitter(self) -> float:
        return float(self._rng.normal(0, CFG.network_jitter_ms))

    def _emit_bytes(self, payload: bytes, records: List[PacketRecord],
                    stage: str, clock_ms: float) -> None:
        pkt = self.sender.encrypt(payload)
        sz = pkt.byte_size
        if self.enable_dp and self.dp_sigma > 0:
            import math
            pad = max(0, math.ceil(self._rng.normal(0, self.dp_sigma)))
            pad = (pad // CFG.dp_pad_granularity) * CFG.dp_pad_granularity
            sz += pad
        records.append(PacketRecord(pkt.seq_no, sz, clock_ms, stage))

    def _emit_text(self, text: str, records: List[PacketRecord],
                   stage: str, clock_ms: float) -> None:
        self._emit_bytes(text.encode("utf-8"), records, stage, clock_ms)
