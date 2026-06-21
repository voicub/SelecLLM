"""utils/logger.py — coloured console logger."""
import logging, sys
from colorama import Fore, Style, init as _init
_init(autoreset=True)

_LCOLOUR = {"DEBUG":Fore.CYAN,"INFO":Fore.GREEN,"WARNING":Fore.YELLOW,
             "ERROR":Fore.RED,"CRITICAL":Fore.MAGENTA}
_STAGECOL = {"VAD":Fore.BLUE,"ASR":Fore.CYAN,"SELECLLM":Fore.MAGENTA,
              "MT":Fore.YELLOW,"TTS":Fore.GREEN,"E2EE":Fore.RED,
              "ATTACK":Fore.RED,"DEFENCE":Fore.GREEN,"EVAL":Fore.CYAN}

class _Fmt(logging.Formatter):
    def format(self, r):
        c = _LCOLOUR.get(r.levelname,"")
        ts = self.formatTime(r, "%H:%M:%S")
        return f"{ts} {c}[{r.levelname[0]}]{Style.RESET_ALL} {Fore.WHITE}{r.name:<14}{Style.RESET_ALL} {r.getMessage()}"

def get_logger(name, level=logging.INFO):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(_Fmt())
        lg.addHandler(h)
    lg.setLevel(level); lg.propagate = False
    return lg

def stage_log(lg, stage, msg):
    c = _STAGECOL.get(stage.upper(), Fore.WHITE)
    lg.info(f"{c}[{stage}]{Style.RESET_ALL} {msg}")
