"""Quick one-off: regenerate just the reuse_at_p and necessity_gap plots from
the fresh results_relp_new/cross_task_relp/ metrics, using the existing
summarize.py helpers but pointed at the new data dir. Bypasses the cross-task
ablation outputs (which are still stale).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "results" / "cross_task_ablation_relp"))

import summarize as S

# Override METHODS["relp"] paths to point at fresh data
S.METHODS["relp"]["main_dir"] = REPO / "results_relp_new" / "cross_task_relp"
S.METHODS["relp"]["ct_dir"] = REPO / "results_relp_new" / "cross_task_ablation_relp_dummy"  # only used by plots we won't run
S.FIG_DIR = REPO / "results_relp_new" / "figs"
S.FIG_DIR.mkdir(parents=True, exist_ok=True)

# `available_models` checks ct_path for each K — that fails on our dummy. So we
# stub it for the plots that don't need ct outputs.
def available_models_main_only(method="relp"):
    present = []
    for key, display in S.MODELS:
        for task in S.TASKS:
            p = S.main_dir(method, key, task)
            if p.exists():
                present.append((key, display))
                break
    seen = set()
    out = []
    for k, d in present:
        if k in seen:
            continue
        seen.add(k)
        out.append((k, d))
    return out

models_here = available_models_main_only("relp")
print(f"Models with fresh metrics: {[m[1] for m in models_here]}")

for k in (10, 20, 30):
    out = S.plot_reuse_at_p("relp", models_here, k=k)
    print(f"[reuse_at_p] {out}")

for p in (100, 95):
    out = S.plot_necessity_gap("relp", models_here, p=p)
    print(f"[necessity_gap] {out}")

for k in (10, 20, 30):
    for p in (100, 95):
        out = S.plot_within_task_bars("relp", models_here, k=k, p=p)
        print(f"[within_task_bars] {out}")
