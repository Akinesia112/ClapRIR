#!/usr/bin/env python3
"""The E3 table and its verdict. CPU only.

Aggregation order matters and is registered in publication/EXPERIMENTS.md:

    1. median the Flow's 5 draws within (seed, room)   <- sampling uncertainty
    2. median over seeds within room                    <- training stochasticity
    3. cluster bootstrap over ROOMS                     <- the statistical unit

Step 1 is what stops the Flow being credited with five times the sample size for
having a sampler. The regression has one deterministic prediction per (seed,
room) and passes through step 1 unchanged.

Decision order is frozen: EDC -> C50/EDT -> EDP -> log-spectrum. NRMSE is printed
and takes no part.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

PROVIDERS = ("shoebox", "mit", "but", "ace", "openair")
JUDGEABLE = ("mit", "openair")
BOOTSTRAP = 10_000
METRICS = (("edc_rmse_db", "EDC RMSE (dB)", True),
           ("abs_c50_error_db", "|C50 error| (dB)", True),
           ("abs_edt_error_s", "|EDT error| (s)", True),
           ("echo_density_rmse", "EDP RMSE", True),
           ("stft_logmag_mse", "STFT log-mag MSE", True),
           ("nrmse", "NRMSE (companion)", False))


def load(p: Path):
    out = []
    for r in csv.DictReader(p.open()):
        d = {}
        for k, v in r.items():
            if k in ("arm", "provider", "room_id", "sample_id"):
                d[k] = v
            else:
                try:
                    d[k] = float(v) if v not in ("", "nan") else np.nan
                except ValueError:
                    d[k] = v
        out.append(d)
    return out


def collapse(rows, metric):
    """(arm, provider, room) -> one value per room, draws then seeds medianed."""
    by = {}
    for r in rows:
        v = r.get(metric, np.nan)
        if np.isfinite(v):
            by.setdefault((r["arm"], r["provider"], r["room_id"], r["seed"]), []).append(v)
    per_seed = {k: float(np.median(v)) for k, v in by.items()}       # step 1
    by_room = {}
    for (arm, prov, room, _), v in per_seed.items():
        by_room.setdefault((arm, prov, room), []).append(v)
    return {k: float(np.median(v)) for k, v in by_room.items()}      # step 2


def contrast(room_vals, provider, rng):
    """flow minus regression, cluster bootstrap over rooms."""
    rooms = sorted({r for (a, p, r) in room_vals if p == provider})
    d = [room_vals[("flow", provider, r)] - room_vals[("regression", provider, r)]
         for r in rooms
         if ("flow", provider, r) in room_vals and ("regression", provider, r) in room_vals]
    if not d:
        return None
    d = np.array(d)
    boot = np.median(d[rng.integers(0, len(d), size=(BOOTSTRAP, len(d)))], axis=1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {"provider": provider, "n_rooms": len(d), "median_delta": float(np.median(d)),
            "ci95_low": lo, "ci95_high": hi,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "judgeable": provider in JUDGEABLE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-root", type=Path, default=Path("reports/matched_estimator_comparison"))
    a = ap.parse_args()
    rows = load(a.report_root / "results/per_example.csv")
    rng = np.random.default_rng(20260907)
    seeds = sorted({r["seed"] for r in rows})
    print(f"{len(rows)} rows, seeds {seeds}, "
          f"{len({r['room_id'] for r in rows})} rooms\n")

    print("=" * 96)
    print("E3  1 s matched Flow  MINUS  1 s matched Regression")
    print("    negative = the Flow is better.  * = CI excludes zero AND the provider")
    print("    has rooms to resample (mit 42, openair 9). BUT 2 / ACE 1 are descriptive;")
    print("    shoebox is a synthetic diagnostic.")
    print("=" * 96)
    print(f"  {'metric':<24}" + "".join(f"{p:>13}" for p in PROVIDERS))
    out, verdict_terms = [], {}
    for metric, label, gating in METRICS:
        rv = collapse(rows, metric)
        cells, row = [], {}
        for p in PROVIDERS:
            c = contrast(rv, p, rng)
            row[p] = c
            if c:
                out.append({**c, "metric": metric, "gating": gating})
                star = "*" if c["excludes_zero"] and c["judgeable"] else " "
                cells.append(f"{c['median_delta']:>12.4f}{star}")
            else:
                cells.append(f"{'-':>13}")
        if gating:
            verdict_terms[metric] = row
        print(f"  {label:<24}" + "".join(cells))
    print("\n  absolute medians, MIT (the one provider carrying full inference):")
    for metric, label, _ in METRICS:
        rv = collapse(rows, metric)
        f = np.median([v for (arm, p, _), v in rv.items() if arm == "flow" and p == "mit"])
        g = np.median([v for (arm, p, _), v in rv.items() if arm == "regression" and p == "mit"])
        print(f"    {label:<24} regression {g:>10.4f}    flow {f:>10.4f}")

    # ---- the frozen decision order ----------------------------------------
    def read(metric):
        wins = losses = 0
        for p in JUDGEABLE:
            c = verdict_terms[metric].get(p)
            if not c or not c["excludes_zero"]:
                continue
            wins += c["median_delta"] < 0
            losses += c["median_delta"] > 0
        return wins, losses

    print("\n" + "=" * 96)
    print("VERDICT, in the frozen order EDC -> C50/EDT -> EDP -> log-spectrum")
    print("=" * 96)
    decided, reason = None, []
    for metric, label, gating in METRICS:
        if not gating:
            continue
        w, l = read(metric)
        state = ("flow ahead" if w and not l else "regression ahead" if l and not w
                 else "split" if w and l else "no separation")
        reason.append(f"{label}: {state} ({w} judgeable provider(s) favour flow, {l} favour regression)")
        print(f"  {label:<24} {state}")
        if decided is None and state in ("flow ahead", "regression ahead"):
            decided = "flow" if state == "flow ahead" else "regression"
    if decided is None:
        verdict, line = "INCONCLUSIVE", (
            "no gating metric separates the two arms on a provider that can carry an "
            "interval; on the meeting's rule the simpler estimator wins, so regression "
            "is the mainline and the Flow work becomes development history.")
    elif decided == "regression":
        verdict, line = "REGRESSION", (
            "the matched regression is ahead on the first gating metric that separates "
            "them, so it is the paper's estimator and E4 goes to supplementary.")
    else:
        verdict, line = "FLOW", (
            "the Flow is ahead on the first gating metric that separates them; it earns "
            "the mainline only if this survives the remaining gating metrics below it.")
    print(f"\nVERDICT E3 {verdict} {line}")
    print("\n  cost, reported not matched:  regression 1 forward pass, 26.53 M active")
    print("                               flow       20 NFE,        29.31 M active")
    (a.report_root / "results/contrasts.json").write_text(json.dumps(out, indent=2) + "\n")
    (a.report_root / "result.json").write_text(json.dumps(
        {"id": "E3-MATCHED-COMPARISON", "verdict": verdict, "one_line": line,
         "decision_order": [m for m, _, g in METRICS if g],
         "reading": reason, "n_seeds": len(seeds),
         "nrmse_excluded_from_verdict": True,
         "flow_draws_aggregated_within_seed_room": True}, indent=2) + "\n")
    print(f"\nwrote {a.report_root/'result.json'}")


if __name__ == "__main__":
    main()
