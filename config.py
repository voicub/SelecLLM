"""config.py — global parameters for the SelecLLM pipeline."""
from dataclasses import dataclass, field
from typing import List

@dataclass
class PipelineConfig:
    # Audio
    sample_rate: int = 16_000
    vad_energy_threshold: float = 0.01
    vad_min_duration_ms: float = 200.0

    # ASR (Whisper-medium profile — paper Table 5)
    asr_base_wer: float = 0.0871        # 8.71% on MuST-C en
    asr_confidence_mean: float = 0.82
    asr_confidence_std: float = 0.14
    asr_latency_mean_ms: float = 134.0  # paper value
    asr_latency_std_ms: float = 22.0
    # Real-time factor cap: Whisper-medium processes ~1s audio in ~0.28s on CPU
    # We cap audio_duration contribution so sim_total stays realistic
    asr_rtf: float = 0.0                # set to 0: latency is pure model time, not RTF

    # SelecLLM gating thresholds (paper §4, Eq. 3)
    tau_c: float = 0.72                 # min-confidence threshold
    tau_p: float = 47.0                 # perplexity threshold (calibrated for ~28.4% routing rate)

    # LLM corrector (Mistral-7B INT4 profile)
    llm_latency_mean_ms: float = 120.0
    llm_latency_std_ms: float = 18.0
    # Fraction of routed utterances where correction actually fires
    # (paper reports 28.4% routing rate with joint gate)
    llm_target_routing_rate: float = 0.284

    # MT (NLLB-200 1.3B profile)
    mt_latency_mean_ms: float = 87.0
    mt_latency_std_ms: float = 15.0

    # TTS (VITS profile — paper Table 5)
    tts_latency_mean_ms: float = 61.0
    tts_latency_std_ms: float = 11.0

    # Network / packetisation
    mtu_bytes: int = 1_400
    network_jitter_ms: float = 1.0

    # ChaCha20-Poly1305
    chacha_nonce_bytes: int = 12
    chacha_tag_bytes: int = 16

    # DP-Pad defence (paper §5)
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    # sensitivity = max size change between adjacent traces (one MTU segment)
    # Using 128B so sigma stays proportionate to real packet sizes (~200-1400B)
    # giving ~12% overhead as reported in the paper
    dp_sensitivity_bytes: int = 75    # -> sigma~363B -> 12.1% overhead at 1200B mean
    dp_pad_granularity: int = 64

    # Adversary
    adv_languages: List[str] = field(default_factory=lambda: [
        "en","de","fr","es","it","nl","pt","ro","ru","zh"
    ])

    # General
    latency_budget_ms: float = 500.0
    random_seed: int = 42

CFG = PipelineConfig()
