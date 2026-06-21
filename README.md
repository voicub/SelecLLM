# SelecLLM

## Requirements
```sbah
pip install numpy scipy scikit-learn cryptography matplotlib colorama click
```
## Quick start
```bash
# Single utterance through the full E2EE pipeline
python main.py run --lang en --target-lang de --text "Hello, how are you today?"

# All three adversarial attacks on 300 utterances
python main.py attack --n-samples 300

# WER comparison: baseline vs always-on LLM vs SelecLLM
python main.py evaluate --utterances 150

# DP-Pad defence + attack comparison
python main.py defend --epsilon 1.0

# Full paper benchmark + 5 figures
python main.py benchmark --n-per-cell 15 --output-dir results/
```
