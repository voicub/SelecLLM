"""evaluation/benchmark.py — full experiment runner (all paper tables)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import time, json
import numpy as np
from typing import List
from config import CFG
from data.corpus import generate_corpus, Utterance
from core.pipeline import Pipeline, PipelineResult
from core.asr import ASRSimulator
from core.selecllm import SelecLLM, _llm_correct
from adversary.traffic_capture import build_feature_matrix
from adversary.classifiers import run_all_attacks
from defence.dp_padding import DPPaddingDefence
from evaluation.wer import corpus_wer
from utils.logger import get_logger
log = get_logger("BENCHMARK")


def _run_corpus(corpus, enable_sl=True, enable_dp=False,
                dp_eps=1.0, dp_delta=1e-5, verbose=False) -> List[PipelineResult]:
    dp_sigma = DPPaddingDefence(epsilon=dp_eps, delta=dp_delta).sigma if enable_dp else 0.0
    pipes, results = {}, []
    for utt in corpus:
        if utt.lang not in pipes:
            pipes[utt.lang] = Pipeline(src_lang=utt.lang, tgt_lang="de",
                                       enable_selecllm=enable_sl,
                                       enable_dp=enable_dp, dp_sigma=dp_sigma)
        r = pipes[utt.lang].process_text(utt.text, utterance_id=utt.utterance_id)
        r.sentiment = utt.sentiment
        r.speaker_id = utt.speaker_id
        results.append(r)
        if verbose:
            log.info(f"  [{utt.utterance_id}] {utt.lang} {utt.sentiment}")
    return results


# === A: pipeline throughput ==================================================

def experiment_pipeline(n_per_cell=20, verbose=False):
    log.info("=== Exp A: Pipeline throughput ===")
    corpus = generate_corpus(n_per_lang_sentiment=n_per_cell)
    log.info(f"Corpus: {len(corpus)} utterances, {len(CFG.adv_languages)} languages")
    t0 = time.perf_counter()
    results = _run_corpus(corpus, verbose=verbose)
    elapsed = time.perf_counter() - t0
    lats = [r.total_latency_ms for r in results]
    log.info(f"Done {elapsed:.1f}s  mean={np.mean(lats):.0f}ms  "
             f"p95={np.percentile(lats,95):.0f}ms  "
             f"within_budget={sum(r.within_budget for r in results)/len(results):.1%}")
    return results, corpus


# === B: adversarial attacks ==================================================

def experiment_attacks(results, corpus, verbose=True):
    log.info("=== Exp B: Adversarial attacks (no defence) ===")
    X, _, _, _ = build_feature_matrix(results)
    y_lang = [r.source_lang for r in results]
    y_sent = [r.sentiment   for r in results]
    y_spk  = [r.speaker_id  for r in results]
    atk = run_all_attacks(X, y_lang, y_sent, results, y_spk, verbose=verbose)
    return atk, X, y_lang, y_sent, y_spk


# === C: DP-Pad defence =======================================================

def experiment_defence(corpus, epsilon=1.0, delta=1e-5, verbose=True):
    log.info(f"=== Exp C: DP-Pad defence (eps={epsilon}) ===")
    dp = DPPaddingDefence(epsilon=epsilon, delta=delta)
    log.info(dp.privacy_guarantee())
    results_dp = _run_corpus(corpus, enable_dp=True, dp_eps=epsilon, dp_delta=delta)
    X, _, _, _ = build_feature_matrix(results_dp)
    y_lang = [r.source_lang for r in results_dp]
    y_sent = [r.sentiment   for r in results_dp]
    y_spk  = [r.speaker_id  for r in results_dp]
    atk_dp = run_all_attacks(X, y_lang, y_sent, results_dp, y_spk, verbose=verbose)
    sizes = [p.byte_size for r in results_dp for p in r.packets]
    if sizes:
        st = dp.analyse_trace(sizes)
        log.info(f"Bandwidth overhead: {st.overhead_fraction:.1%}")
    return atk_dp


# === D: SelecLLM WER =========================================================

def experiment_selecllm_wer(corpus, n_utterances=200, verbose=True):
    log.info("=== Exp D: SelecLLM WER evaluation ===")
    from utils.audio import synthetic_utterance, audio_duration_ms as adm
    subset = [u for u in corpus if u.lang == "en"][:n_utterances]
    if len(subset) < 10:
        subset = corpus[:min(n_utterances, len(corpus))]
    refs = [u.text for u in subset]

    def _eval(mode):
        asr = ASRSimulator(wer=CFG.asr_base_wer)
        sl = SelecLLM()
        hyps, lats = [], []
        for u in subset:
            a = synthetic_utterance(u.text)
            ar = asr.transcribe(u.text, audio_duration_ms=adm(a))
            if mode == "baseline":
                hyps.append(ar.text); lats.append(ar.latency_ms)
            elif mode == "always_on":
                lo = [0.0]; c = _llm_correct(ar.text, lo)
                hyps.append(c); lats.append(ar.latency_ms + lo[0])
            else:
                sr = sl.process(ar)
                hyps.append(sr.final_text); lats.append(ar.latency_ms + sr.latency_ms)
        return corpus_wer(refs, hyps), float(np.mean(lats))

    bw, bl = _eval("baseline")
    aw, al = _eval("always_on")
    sw, sl_ = _eval("selecllm")
    log.info(f"Baseline   WER={bw:.4f}  lat={bl:.0f}ms")
    log.info(f"Always-On  WER={aw:.4f}  rel={100*(bw-aw)/max(bw,1e-9):+.1f}%  lat={al:.0f}ms")
    log.info(f"SelecLLM   WER={sw:.4f}  rel={100*(bw-sw)/max(bw,1e-9):+.1f}%  lat={sl_:.0f}ms")
    return {"baseline":  {"wer": bw, "latency_ms": bl},
            "always_on": {"wer": aw, "latency_ms": al},
            "selecllm":  {"wer": sw, "latency_ms": sl_}}


# === E: latency breakdown ====================================================

def experiment_latency_breakdown(results):
    log.info("=== Exp E: Latency breakdown ===")
    stages = [("VAD","vad_latency_ms"), ("ASR","asr_latency_ms"),
              ("SelecLLM","selecllm_latency_ms"), ("MT","mt_latency_ms"),
              ("TTS","tts_latency_ms"), ("E2EE","e2ee_latency_ms"),
              ("Total","total_latency_ms")]
    print(f"\n  {'Stage':<12}{'Mean(ms)':>10}{'P50':>8}{'P95':>8}")
    print("  " + "-"*40)
    out = {}
    for name, attr in stages:
        v = np.array([getattr(r, attr) for r in results])
        print(f"  {name:<12}{np.mean(v):>10.1f}{np.percentile(v,50):>8.1f}"
              f"{np.percentile(v,95):>8.1f}")
        out[name] = {"mean": float(np.mean(v)),
                     "p50":  float(np.percentile(v, 50)),
                     "p95":  float(np.percentile(v, 95))}
    return out


# === F: ablation =============================================================

def experiment_ablation(corpus, n_utterances=200):
    log.info("=== Exp F: SelecLLM gating ablation ===")
    from utils.audio import synthetic_utterance, audio_duration_ms as adm
    subset = [u for u in corpus if u.lang == "en"][:n_utterances]
    if not subset:
        subset = corpus[:n_utterances]
    refs = [u.text for u in subset]
    configs = [("Conf. only", 0.72, 9999.0),
               ("PPL only",   9999.0, 45.0),
               ("Both (ours)", 0.72,  45.0)]
    print(f"\n  {'Gate':<16}{'WER':>6}{'Routed':>9}{'Lat(ms)':>10}")
    print("  " + "-"*44)
    out = {}
    for name, tc, tp in configs:
        asr = ASRSimulator(wer=CFG.asr_base_wer)
        sl = SelecLLM(tau_c=tc, tau_p=tp)
        hyps, lats = [], []
        for u in subset:
            a = synthetic_utterance(u.text)
            ar = asr.transcribe(u.text, audio_duration_ms=adm(a))
            sr = sl.process(ar)
            hyps.append(sr.final_text)
            lats.append(ar.latency_ms + sr.latency_ms)
        w = corpus_wer(refs, hyps)
        ml = float(np.mean(lats))
        rr = sl.routing_rate * 100
        print(f"  {name:<16}{w:>6.4f}{rr:>8.1f}%{ml:>10.1f}")
        out[name] = {"wer": w, "routed_pct": rr, "latency_ms": ml}
    return out


# === master ==================================================================

def run_full_benchmark(n_per_cell=20, output_dir="results", verbose=False):
    os.makedirs(output_dir, exist_ok=True)
    log.info("SelecLLM Full Benchmark — all experiments")
    all_res = {}
    results, corpus = experiment_pipeline(n_per_cell=n_per_cell, verbose=verbose)
    atk, X, yl, ys, ysp = experiment_attacks(results, corpus, verbose=verbose)
    all_res["attacks_no_defence"] = atk
    all_res["attacks_dp_defence"] = experiment_defence(corpus, verbose=verbose)
    all_res["wer"]      = experiment_selecllm_wer(corpus, n_utterances=min(200, len(corpus)))
    all_res["latency"]  = experiment_latency_breakdown(results)
    all_res["ablation"] = experiment_ablation(corpus, n_utterances=min(200, len(corpus)))
    path = os.path.join(output_dir, "benchmark_results.json")
    with open(path, "w") as f:
        json.dump(all_res, f, indent=2)
    log.info(f"Results saved: {path}")
    return all_res, results, corpus
