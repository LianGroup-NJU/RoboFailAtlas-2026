"""Shared validation-only stopping and exact best-checkpoint selection."""
import math
class Convergence:
    def __init__(self, min_steps, patience=4, min_delta=0.0005, state=None):
        self.min_steps=min_steps;self.patience=patience;self.min_delta=min_delta
        self.state=state or {"anchor_A":None,"bad_checks":0,"best_A":None,"best_T":None,"best_step":None,"checks":0,"converged":False}
    def update(self, report, step):
        a=report.get("A_mae");t=report.get("T_auc")
        if a is None or not math.isfinite(a):raise ValueError("invalid_full_validation_A")
        if t is not None and not math.isfinite(t):raise ValueError("invalid_full_validation_T")
        s=self.state;s["checks"]+=1
        prior_t=s["best_T"] if s["best_T"] is not None else -math.inf
        if s["best_A"] is None or a<s["best_A"] or a==s["best_A"] and (t if t is not None else -math.inf)>prior_t:
            s.update(best_A=a,best_T=t,best_step=step)
        if s["anchor_A"] is None or s["anchor_A"]-a>=self.min_delta:
            s["anchor_A"]=a;s["bad_checks"]=0
        else:s["bad_checks"]+=1
        s["converged"]=step>=self.min_steps and s["bad_checks"]>=self.patience
        return dict(s)
