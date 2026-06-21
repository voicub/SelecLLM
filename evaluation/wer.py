"""evaluation/wer.py — Word Error Rate via dynamic-programming edit distance."""
from typing import List

def _edits(ref: List[str], hyp: List[str]):
    n,m=len(ref),len(hyp)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            if ref[i-1]==hyp[j-1]: dp[i][j]=dp[i-1][j-1]
            else: dp[i][j]=1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    i,j=n,m; S=D=I=0
    while i>0 or j>0:
        if i>0 and j>0 and ref[i-1]==hyp[j-1]: i-=1;j-=1
        elif i>0 and j>0 and dp[i][j]==dp[i-1][j-1]+1: S+=1;i-=1;j-=1
        elif i>0 and dp[i][j]==dp[i-1][j]+1: D+=1;i-=1
        else: I+=1;j-=1
    return S,D,I

def wer(reference: str, hypothesis: str) -> float:
    ref=reference.lower().split(); hyp=hypothesis.lower().split()
    if not ref: return 0.0 if not hyp else 1.0
    S,D,I=_edits(ref,hyp); return (S+D+I)/len(ref)

def corpus_wer(references: List[str], hypotheses: List[str]) -> float:
    te=0; tr=0
    for r,h in zip(references,hypotheses):
        rt=r.lower().split(); ht=h.lower().split()
        if not rt: continue
        S,D,I=_edits(rt,ht); te+=S+D+I; tr+=len(rt)
    return te/max(1,tr)
