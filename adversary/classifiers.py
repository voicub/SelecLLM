"""
adversary/classifiers.py — Three traffic-analysis attacks (paper §5.2).

Attack 1: Language Identification  (GradientBoosting, 300 trees)
Attack 2: Speaker-Turn Detection   (RandomForest, 150 trees, pair features)
Attack 3: Sentiment Inference      (GradientBoosting, 200 trees)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from typing import List, Dict, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from config import CFG
from utils.logger import get_logger
log = get_logger("ATTACK")

class TrafficAdversary:
    """Base adversary.  clf is passed explicitly — never uses truthiness on sklearn objects."""
    def __init__(self, name: str, clf):
        self.name = name
        self.clf  = clf          # <-- stored directly, no `clf or default`
        self.le   = LabelEncoder()
        self._trained = False

    def fit(self, X, y):
        self.le.fit(y); self.clf.fit(X, self.le.transform(y))
        self._trained = True
        log.info(f"[{self.name}] trained {len(X)} samples  classes={list(self.le.classes_)}")
        return self

    def predict(self, X) -> List[str]:
        return list(self.le.inverse_transform(self.clf.predict(X)))

    def evaluate(self, X_test, y_test, verbose=True) -> Dict[str,float]:
        yp = self.predict(X_test)
        acc = accuracy_score(y_test, yp)
        f1  = f1_score(y_test, yp, average="macro", zero_division=0)
        if verbose:
            log.info(f"[{self.name}] accuracy={acc:.3f}  macro-F1={f1:.3f}")
            print(classification_report(y_test, yp, zero_division=0))
        return {"accuracy": acc, "f1_macro": f1}

class LanguageIDAdversary(TrafficAdversary):
    """Attack 1: identify source language from encrypted packet trace."""
    def __init__(self):
        super().__init__("LangID",
            GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                       learning_rate=0.05, random_state=CFG.random_seed))
    def run_experiment(self, X, y_lang, test_size=0.30, verbose=True):
        X_tr,X_te,y_tr,y_te = train_test_split(X, y_lang, test_size=test_size,
                                                 random_state=CFG.random_seed, stratify=y_lang)
        self.fit(X_tr, y_tr)
        res = self.evaluate(X_te, y_te, verbose=verbose)
        res["chance"] = 1.0/len(set(y_lang)); return res

class SpeakerTurnAdversary(TrafficAdversary):
    """Attack 2: detect speaker turns from timing gaps between consecutive utterances."""
    def __init__(self):
        super().__init__("SpeakerTurn",
            RandomForestClassifier(n_estimators=150, max_depth=8,
                                   random_state=CFG.random_seed, n_jobs=-1))
    @staticmethod
    def build_turn_features(results, speaker_ids):
        from adversary.traffic_capture import extract_features
        X,y=[],[]
        for i in range(1,len(results)):
            fp=extract_features(results[i-1].packets); fc=extract_features(results[i].packets)
            X.append(np.concatenate([fp, fc, fc-fp]))   # 126-dim
            y.append("1" if speaker_ids[i]!=speaker_ids[i-1] else "0")
        return np.array(X,dtype=np.float32), y
    def run_experiment(self, results, speaker_ids, test_size=0.30, verbose=True):
        X,y = self.build_turn_features(results, speaker_ids)
        if len(set(y))<2:
            log.warning("[SpeakerTurn] only one class present — skipping"); return {"accuracy":0,"f1_binary":0}
        X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=test_size,
                                                 random_state=CFG.random_seed, stratify=y)
        self.fit(X_tr, y_tr); yp=self.predict(X_te)
        f1  = f1_score([int(v) for v in y_te],[int(v) for v in yp],average="binary",zero_division=0)
        acc = accuracy_score(y_te, yp)
        if verbose:
            log.info(f"[SpeakerTurn] accuracy={acc:.3f}  binary-F1={f1:.3f}")
            print(classification_report(y_te, yp, zero_division=0))
        return {"accuracy": acc, "f1_binary": f1}

class SentimentAdversary(TrafficAdversary):
    """Attack 3: infer sentiment from packet size/timing patterns."""
    def __init__(self):
        super().__init__("Sentiment",
            GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                       learning_rate=0.08, random_state=CFG.random_seed))
    def run_experiment(self, X, y_sent, test_size=0.30, verbose=True):
        X_tr,X_te,y_tr,y_te = train_test_split(X, y_sent, test_size=test_size,
                                                 random_state=CFG.random_seed, stratify=y_sent)
        self.fit(X_tr, y_tr)
        res = self.evaluate(X_te, y_te, verbose=verbose)
        res["chance"] = 1.0/3.0; return res

def run_all_attacks(X, y_lang, y_sent, results, speaker_ids, verbose=True):
    log.info("="*58+"\nADVERSARIAL ATTACK SUITE")
    log.info("--- Attack 1: Language Identification ---")
    li = LanguageIDAdversary().run_experiment(X, y_lang, verbose=verbose)
    log.info("--- Attack 2: Speaker-Turn Detection ---")
    st = SpeakerTurnAdversary().run_experiment(results, speaker_ids, verbose=verbose)
    log.info("--- Attack 3: Sentiment Inference ---")
    si = SentimentAdversary().run_experiment(X, y_sent, verbose=verbose)
    log.info(f"SUMMARY  LI={li['accuracy']:.1%}(chance={li['chance']:.1%})  "
             f"ST-F1={st['f1_binary']:.3f}  SI={si['accuracy']:.1%}(chance={si['chance']:.1%})")
    return {"language_id": li, "speaker_turn": st, "sentiment": si}
