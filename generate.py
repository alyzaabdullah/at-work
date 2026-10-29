"""Generate a SIMULATED SYNC dataset for demonstrating the analysis pipeline.

Nothing here is real participant data. Values are drawn from made-up
distributions chosen to look plausible for adults playing a tilt-to-target game.
"""
import sys
import numpy as np
import pandas as pd

OUT = sys.argv[1]
rng = np.random.default_rng(2026)

N = 12                 # simulated adults
TRIALS = 30            # per hand per session
HZ = 60                # tilt sampling rate
MAX_TILT = 30.0        # degrees = 1.0 normalized
TARGET_DIST = 0.6      # normalized
TARGET_W = 0.15        # target diameter, normalized
BEAT_MS = 1000         # beat interval
SESSIONS = 8           # weekly sessions (day 1, 8, 15, ...)

trials_rows, raw_rows = [], []

def min_jerk(t, T):
    s = np.clip(t / T, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5

for p in range(1, N + 1):
    pid = f"S{p:02d}"
    skill = rng.normal(0, 1)          # stable between-person trait
    for session in range(1, SESSIONS + 1):
        # practice effect that levels off over weeks
        learn = 0.24 * (1 - np.exp(-(session - 1) / 2.5))
        for hand in ("dominant", "non-dominant"):
            hand_pen = 0.0 if hand == "dominant" else 1.0
            # person x hand quality (lower = smoother/more accurate)
            q = 1.0 - 0.26 * skill + 0.22 * hand_pen - learn + rng.normal(0, 0.04)
            q = max(q, 0.35)
            for tr in range(1, TRIALS + 1):
                ang = rng.choice(np.arange(8)) * np.pi / 4
                tx, ty = TARGET_DIST * np.cos(ang), TARGET_DIST * np.sin(ang)
                completed = rng.random() > 0.02
                rt = float(np.exp(rng.normal(np.log(310 * (0.85 + 0.15 * q)), 0.18)))
                mt = float(np.exp(rng.normal(np.log(620 * (0.8 + 0.25 * q)), 0.15)))
                err_sd = 0.045 * q
                ex, ey = tx + rng.normal(0, err_sd), ty + rng.normal(0, err_sd)
                # primary movement
                t = np.arange(0, (rt + mt + 400) / 1000, 1 / HZ)
                tm = t - rt / 1000
                prim_T = mt / 1000 * 0.8
                frac = min_jerk(tm, prim_T)
                # overshoot/undershoot then corrective submovement
                over = 1 + rng.normal(0.0, 0.10 * q)
                x = frac * ex * over
                y = frac * ey * over
                p_corr = min(0.25 + 0.35 * q, 0.95)
                n_corr = rng.binomial(2, p_corr / 2)
                for c in range(n_corr):
                    start = prim_T + c * 0.12
                    cfrac = min_jerk(tm - start, 0.18)
                    x += cfrac * (ex - ex * over) / max(n_corr, 1)
                    y += cfrac * (ey - ey * over) / max(n_corr, 1)
                if n_corr == 0:
                    x += min_jerk(tm - prim_T, 0.2) * (ex - ex * over)
                    y += min_jerk(tm - prim_T, 0.2) * (ey - ey * over)
                # physiological tremor + sensor noise
                trem = 0.004 * q
                x += trem * np.sin(2 * np.pi * 9 * t + rng.uniform(0, 6)) + rng.normal(0, 0.0015, len(t))
                y += trem * np.cos(2 * np.pi * 9 * t + rng.uniform(0, 6)) + rng.normal(0, 0.0015, len(t))
                x[tm < 0] = rng.normal(0, 0.0015, (tm < 0).sum())
                y[tm < 0] = rng.normal(0, 0.0015, (tm < 0).sum())
                end_err = float(np.hypot(x[-1] - tx, y[-1] - ty))
                hit = completed and end_err <= TARGET_W / 2
                beat_off = float(rng.normal(0, 90 + 130 * q))
                trials_rows.append(dict(
                    participant=pid, session=session, day=1 + 7 * (session - 1),
                    hand=hand, trial=tr, target_angle_deg=round(np.degrees(ang)),
                    completed=completed, hit=bool(hit),
                    endpoint_error=round(end_err if completed else np.nan, 4),
                    rt_ms=round(rt, 1) if completed else np.nan,
                    mt_ms=round(mt, 1) if completed else np.nan,
                    beat_offset_ms=round(beat_off, 1) if completed else np.nan,
                ))
                if completed:
                    for ti, xi, yi in zip(t, x, y):
                        raw_rows.append((pid, session, hand, tr, round(ti * 1000, 1),
                                         round(xi * MAX_TILT, 2), round(yi * MAX_TILT, 2)))

trials = pd.DataFrame(trials_rows)
raw = pd.DataFrame(raw_rows, columns=["participant", "session", "hand", "trial",
                                      "t_ms", "tilt_x_deg", "tilt_y_deg"])
note = "# SIMULATED DATA - not real participants. Generated for testing and demonstration only.\n"
for df, name in ((trials, "sync_simulated_trials.csv"), (raw, "sync_simulated_raw_tilt.csv")):
    with open(f"{OUT}/{name}", "w") as f:
        f.write(note)
        df.to_csv(f, index=False)
print(len(trials), "trials;", len(raw), "raw samples")
