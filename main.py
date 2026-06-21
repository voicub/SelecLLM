#!/usr/bin/env python3
"""
main.py — SelecLLM CLI

Commands:
  run        Process one utterance through the full E2EE pipeline
  attack     Run all three adversarial traffic-analysis attacks
  evaluate   Evaluate SelecLLM vs baselines (WER + latency)
  defend     Apply DP-Pad defence and compare attack accuracy
  benchmark  Full paper reproduction (all experiments + plots)

Examples:
  python main.py run --lang en --target-lang de --text "Hello, how are you?"
  python main.py attack --n-samples 300
  python main.py evaluate --utterances 100
  python main.py defend --epsilon 1.0
  python main.py benchmark --n-per-cell 15 --output-dir results/
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import click
from colorama import Fore, Style, init as _init
_init(autoreset=True)

from config import CFG
from utils.logger import get_logger
log = get_logger("MAIN")


def _banner():
    print(Fore.CYAN + """
+------------------------------------------------------+
|  SelecLLM - Encrypted Speech Translation Research   |
|  Privacy Risks + LLM Error Correction in E2EE STT   |
+------------------------------------------------------+""" + Style.RESET_ALL)


@click.group()
def cli():
    """SelecLLM: Encrypted Speech Translation Research Platform."""
    _banner()


# ── run ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--lang",         "-l", default="en",  show_default=True, help="Source language")
@click.option("--target-lang",  "-t", default="de",  show_default=True, help="Target language")
@click.option("--text",         "-x", required=True,                    help="Input utterance")
@click.option("--no-selecllm",        is_flag=True,                     help="Disable SelecLLM")
@click.option("--dp",                 is_flag=True,                     help="Enable DP-Pad")
@click.option("--epsilon",      "-e", default=1.0,   show_default=True)
def run(lang, target_lang, text, no_selecllm, dp, epsilon):
    """Process one utterance through the full pipeline."""
    from core.pipeline import Pipeline
    dp_sigma = 0.0
    if dp:
        from defence.dp_padding import DPPaddingDefence
        d = DPPaddingDefence(epsilon=epsilon)
        dp_sigma = d.sigma
        log.info(d.privacy_guarantee())
    pipe = Pipeline(src_lang=lang, tgt_lang=target_lang,
                    enable_selecllm=not no_selecllm,
                    enable_dp=dp, dp_sigma=dp_sigma)
    r = pipe.process_text(text, utterance_id=0)

    sep = Fore.CYAN + "-"*56 + Style.RESET_ALL
    print(f"\n{sep}")
    print(f"{Fore.GREEN}  Source      :{Style.RESET_ALL} {r.source_text}")
    print(f"{Fore.YELLOW}  ASR         :{Style.RESET_ALL} {r.asr_text}")
    print(f"{Fore.MAGENTA}  SelecLLM    :{Style.RESET_ALL} {r.selecllm_text}")
    status = "FIRED" if r.gate_fired else "open"
    print(f"  Gate        : {status}  accepted={r.correction_accepted}")
    print(f"  Gate reason : {r.gate_reason}")
    print(f"{Fore.GREEN}  Translation :{Style.RESET_ALL} {r.translation}")
    print(f"\n  Latency breakdown:\n{r.latency_table()}")
    if r.within_budget:
        budget_str = Fore.GREEN + f"PASS (within {CFG.latency_budget_ms:.0f}ms)" + Style.RESET_ALL
    else:
        budget_str = Fore.RED + "FAIL (over budget)" + Style.RESET_ALL
    print(f"\n  Budget: {budget_str}  |  Encrypted packets: {len(r.packets)}")
    print(f"{sep}\n")


# ── attack ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--n-samples",  "-n", default=300, show_default=True,
              help="Total utterances across all languages")
@click.option("--languages",  "-l", default=None,
              help="Comma-separated language codes (default: all 10)")
@click.option("--verbose/--quiet", default=True)
def attack(n_samples, languages, verbose):
    """Three adversarial traffic-analysis attacks (no defence)."""
    from data.corpus import generate_corpus
    from evaluation.benchmark import _run_corpus, experiment_attacks
    langs = languages.split(",") if languages else CFG.adv_languages
    npc = max(5, n_samples // (len(langs) * 3))
    log.info(f"{npc} utt/(lang x sent) x {len(langs)} langs x 3 = {npc*len(langs)*3} total")
    corpus = generate_corpus(n_per_lang_sentiment=npc, languages=langs)
    results = _run_corpus(corpus)
    atk, *_ = experiment_attacks(results, corpus, verbose=verbose)
    li = atk.get("language_id", {}); st = atk.get("speaker_turn", {}); si = atk.get("sentiment", {})
    print(f"\n{Fore.CYAN}RESULTS{Style.RESET_ALL}")
    print(f"  Language ID   accuracy : {li.get('accuracy',0):.1%}  "
          f"(chance {li.get('chance',0):.1%})")
    print(f"  Speaker-Turn  F1       : {st.get('f1_binary',0):.3f}")
    print(f"  Sentiment     accuracy : {si.get('accuracy',0):.1%}  "
          f"(chance {si.get('chance',0):.1%})")


# ── evaluate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--utterances", "-n", default=150, show_default=True)
def evaluate(utterances):
    """Evaluate SelecLLM vs baselines on WER and latency."""
    from data.corpus import generate_corpus
    from evaluation.benchmark import experiment_selecllm_wer
    npc = max(5, utterances // (len(CFG.adv_languages) * 3))
    corpus = generate_corpus(n_per_lang_sentiment=npc)
    wr = experiment_selecllm_wer(corpus, n_utterances=utterances)
    bwer = wr.get("baseline", {}).get("wer", 0)
    print(f"\n{Fore.CYAN}WER RESULTS{Style.RESET_ALL}")
    for k, lbl in [("baseline",  "Baseline (Whisper)"),
                   ("always_on", "Always-On LLM"),
                   ("selecllm",  "SelecLLM (ours)")]:
        r = wr.get(k, {}); w = r.get("wer", 0); lat = r.get("latency_ms", 0)
        rel = 100*(bwer-w)/max(bwer, 1e-9) if k != "baseline" else 0
        print(f"  {lbl:<24}  WER={w:.4f} ({rel:+.1f}%)  lat={lat:.0f}ms")


# ── defend ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--epsilon",   "-e", default=1.0,   show_default=True)
@click.option("--delta",     "-d", default=1e-5,  show_default=True)
@click.option("--n-samples", "-n", default=200,   show_default=True)
@click.option("--verbose/--quiet", default=True)
def defend(epsilon, delta, n_samples, verbose):
    """DP-Pad defence: compare attack accuracy before and after."""
    from data.corpus import generate_corpus
    from evaluation.benchmark import _run_corpus, experiment_attacks, experiment_defence
    from defence.dp_padding import DPPaddingDefence
    npc = max(5, n_samples // (len(CFG.adv_languages) * 3))
    corpus = generate_corpus(n_per_lang_sentiment=npc)
    log.info("Running WITHOUT defence...")
    results = _run_corpus(corpus)
    atk_nd, *_ = experiment_attacks(results, corpus, verbose=False)
    log.info(f"Running WITH DP-Pad (eps={epsilon})...")
    atk_dp = experiment_defence(corpus, epsilon=epsilon, delta=delta, verbose=verbose)
    dp = DPPaddingDefence(epsilon=epsilon, delta=delta)
    oh = dp.expected_overhead_fraction()

    print(f"\n{Fore.CYAN}DEFENCE COMPARISON{Style.RESET_ALL}")
    print(f"  {'Metric':<26}{'No Defence':>12}{'DP-Pad':>12}")
    print("  " + "-"*52)

    def row(lbl, key1, key2, fmt):
        nv = atk_nd.get(key1, {}).get(key2, 0)
        dv = atk_dp.get(key1, {}).get(key2, 0)
        print(f"  {lbl:<26}{format(nv, fmt):>12}{format(dv, fmt):>12}")

    row("LI accuracy",        "language_id",  "accuracy",   ".1%")
    row("Speaker-Turn F1",    "speaker_turn", "f1_binary",  ".3f")
    row("Sentiment accuracy", "sentiment",    "accuracy",   ".1%")
    print(f"  {'BW overhead':<26}{'0.0%':>12}{oh:.1%}")
    print(f"  {'sigma (Gaussian noise)':<26}{'---':>12}{dp.sigma:.0f}B")


# ── benchmark ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--n-per-cell",  "-n", default=15,       show_default=True,
              help="Utterances per (lang x sentiment) cell")
@click.option("--output-dir",  "-o", default="results", show_default=True)
@click.option("--plots/--no-plots",  default=True)
@click.option("--verbose/--quiet",   default=False)
def benchmark(n_per_cell, output_dir, plots, verbose):
    """Full paper reproduction: all experiments + figures."""
    from evaluation.benchmark import run_full_benchmark
    t0 = time.perf_counter()
    all_res, results, corpus = run_full_benchmark(
        n_per_cell=n_per_cell, output_dir=output_dir, verbose=verbose)
    if plots:
        from evaluation.plots import generate_all_plots
        generate_all_plots(
            results,
            all_res.get("wer", {}),
            all_res.get("attacks_no_defence", {}),
            all_res.get("attacks_dp_defence", {}),
            output_dir=output_dir,
        )
    elapsed = time.perf_counter() - t0
    print(f"\n{Fore.GREEN}Done in {elapsed:.1f}s{Style.RESET_ALL}")
    print(f"  JSON : {output_dir}/benchmark_results.json")
    if plots:
        print(f"  Figs : {output_dir}/fig*.png")


if __name__ == "__main__":
    cli()
