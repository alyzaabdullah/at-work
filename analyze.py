"""Analysis pipeline for SYNC session data (run here on SIMULATED data).

1. Smoothness per trial: log dimensionless jerk (LDLJ) from the raw tilt trace.
2. Per-person summaries for each session and hand.
3. Test-retest reliability (ICC(2,1), absolute agreement) between sessions 1 and 2.
4. Known-groups check: dominant vs non-dominant hand (paired t-test, Cohen's dz).
"""
import sys, json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy import stats

IN, OUT = sys.argv[1], sys.argv[2]
trials = pd.read_csv(f"{IN}/sync_simulated_trials.csv", comment="#")
raw = pd.read_csv(f"{IN}/sync_simulated_raw_tilt.csv", comment="#")
HZ = 60
b, a = butter(4, 8 / (HZ / 2))

def ldlj(g, rt_ms, mt_ms):
    g = g[(g.t_ms >= rt_ms) & (g.t_ms <= rt_ms + mt_ms + 200)]
    if len(g) < 20:
        return np.nan
    x = filtfilt(b, a, g.tilt_x_deg.values)
    y = filtfilt(b, a, g.tilt_y_deg.values)
    dt = 1 / HZ
    jx = np.gradient(np.gradient(np.gradient(x, dt), dt), dt)
    jy = np.gradient(np.gradient(np.gradient(y, dt), dt), dt)
    T = len(g) * dt
    A = np.hypot(x[-1] - x[0], y[-1] - y[0])
    if A < 1e-6:
        return np.nan
    dlj = (T**5 / A**2) * np.sum(jx**2 + jy**2) * dt
    return -np.log(dlj)

keyed = {k: g for k, g in raw.groupby(["participant", "session", "hand", "trial"])}
vals = []
for r in trials.itertuples():
    g = keyed.get((r.participant, r.session, r.hand, r.trial))
    vals.append(np.nan if g is None or not r.completed else ldlj(g, r.rt_ms, r.mt_ms))
trials["ldlj"] = vals

summ = (trials.groupby(["participant", "session", "hand"])
        .agg(day=("day", "first"), hit_rate=("hit", "mean"), mean_error=("endpoint_error", "mean"),
             mean_rt_ms=("rt_ms", "mean"), mean_mt_ms=("mt_ms", "mean"),
             beat_offset_sd_ms=("beat_offset_ms", "std"), smoothness_ldlj=("ldlj", "mean"))
        .reset_index())
summ.to_csv(f"{OUT}/sync_simulated_summary.csv", index=False)

def icc21(m):
    """m: n subjects x k sessions."""
    n, k = m.shape
    gm = m.mean()
    msr = k * ((m.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((m.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

metrics = {"smoothness_ldlj": "Smoothness (LDLJ)", "mean_error": "Endpoint error",
           "hit_rate": "Hit rate", "mean_rt_ms": "Reaction time",
           "mean_mt_ms": "Movement time", "beat_offset_sd_ms": "Beat timing variability"}
dom = summ[(summ.hand == "dominant") & (summ.session.isin([1, 2]))]
rel = []
for col, label in metrics.items():
    w = dom.pivot(index="participant", columns="session", values=col)
    rel.append(dict(metric=label, icc=round(float(icc21(w.values)), 2)))

s1 = summ[summ.session == 1].pivot(index="participant", columns="hand", values="smoothness_ldlj")
d = s1["dominant"] - s1["non-dominant"]
t, p = stats.ttest_rel(s1["dominant"], s1["non-dominant"])
known = dict(mean_dom=round(float(s1["dominant"].mean()), 2),
             mean_nondom=round(float(s1["non-dominant"].mean()), 2),
             t=round(float(t), 2), p=float(p), dz=round(float(d.mean() / d.std(ddof=1)), 2),
             n=int(len(d)), all_higher=int((d > 0).sum()))

w = dom.pivot(index="participant", columns="session", values="smoothness_ldlj")
pairs = {
    "retest": [dict(id=i, s1=round(float(r[1]), 3), s2=round(float(r[2]), 3)) for i, r in w.iterrows()],
    "hands": [dict(id=i, dom=round(float(r["dominant"]), 3), nondom=round(float(r["non-dominant"]), 3))
              for i, r in s1.iterrows()],
}
# ---- dashboard data: per session cards, progression, and a light trajectory sample
def longest_streak(h):
    best = cur = 0
    for v in h:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best
dash = {}
for (pid, ses, hand), g in trials.groupby(["participant", "session", "hand"]):
    g = g.sort_values("trial")
    rg = raw[(raw.participant == pid) & (raw.session == ses) & (raw.hand == hand)]
    traj = [[[round(a / 30, 3), round(b_ / 30, 3)] for a, b_ in zip(t.tilt_x_deg.values[::4], t.tilt_y_deg.values[::4])]
            for _, t in rg.groupby("trial")]
    dash.setdefault(pid, {}).setdefault(hand, []).append(dict(
        session=int(ses), day=int(g.day.iloc[0]), trials=int(len(g)),
        completed=round(float(g.completed.mean()) * 100), streak=longest_streak(g.hit.values),
        hit_rate=round(float(g.hit.mean()) * 100, 1),
        mean_error=round(float(g.endpoint_error.mean()), 3), mean_rt=round(float(g.rt_ms.mean())),
        mean_mt=round(float(g.mt_ms.mean())), beat_sd=round(float(g.beat_offset_ms.std())),
        samples=int(len(rg)), ldlj=round(float(g.ldlj.mean()), 2), traj=traj))
json.dump(dash, open(f"{OUT}/dashboard.json", "w"))

json.dump(dict(reliability=rel, known=known, pairs=pairs, n_sessions=int(trials.session.max()),
               n_trials=int((trials.session <= 2).sum()), n_samples=int((raw.session <= 2).sum())),
          open(f"{OUT}/results.json", "w"), indent=1)
print(json.dumps(dict(reliability=rel, known=known), indent=1))
