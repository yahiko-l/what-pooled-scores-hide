"""R1 — matched-controls intervention ladder (M1, EXPERIMENT_PLAN run card R1).

Separates the "composition carrier" from generic relation-graph corruption:
each rung rewires the relation graph while preserving progressively more
low-order statistics; frozen ultra_3g is evaluated zero-shot on each variant.

Rungs (cumulative preservation):
  L0 original
  L5 fully random endpoints (destroys degrees)             — floor
  L1 degree-preserving rewire + reciprocal-pair count restored
  L2 L1 + 2/3-motif census (2-paths exact via degree preservation;
     triangle census matched by simulated annealing)
  L3 L2 + swaps restricted to connected components (post-checked)
  L4 identity permutation within type-wise degree-profile classes
     (preserves ALL structural statistics; may be near-trivial — reported)
  L4loose arbitrary identity permutation (isomorphic relabel; preserves the
     entire unlabeled graph, breaks only the relation<->structure binding)

Datasets: FB v1-v4 (in-family diagnostic) + exploration set NL-25/WK-25
(out-of-family, exploratory only — R1 data deviation clause, Our_Workflow §8).
WN v1-v4: permutation-null boundary layer only (small-|R|).

Claim mode (PLAN_AMENDMENT_1, post prereg-v1 tag): --claim-graphs runs the
one-shot C1-OOF confirmation on NL-50/75/100 + WK-75 with exact-construction
rungs only (C-3: L1/L4/L4loose/L5), the C-5 primary line, the C-6 mechanism
line, and the C-7 small-|R| permutation-null power boundary; --ckpt swaps in
ultra_4g / ultra_50g for the §D cross-ckpt replication.

Pass line (mechanical): some rung that
preserves all low-order statistics (L2/L3/L4/L4loose) loses >= 50% MRR.
"""
import os, json, time, argparse
import numpy as np
import torch

import pilot_harness as PH
import tunit_common as TC

RUNGS = ["L1", "L2", "L3", "L4", "L4loose", "L5"]

# PLAN_AMENDMENT_1 §C frozen invocation (production claim mode). Any request
# touching a claim graph must match this EXACTLY (amendment review:
# the one-shot evidence must not be producible under a
# weakened invocation).
CLAIM_FROZEN = {
    "graphs": {"nl-50", "nl-75", "nl-100", "wk-75"},
    "rungs": {"L1", "L4", "L4loose", "L5"},
    "seeds": 5,
    "claim_null": 200,
    "claim_null_r_threshold": 30,
    "ckpts": {"ultra_3g", "ultra_4g", "ultra_50g"},
}


def validate_claim_invocation(args, rungs):
    """Enforce the frozen §C invocation whenever a claim graph is requested.
    Returns 'production' | 'smoke' | None (no claim graphs requested)."""
    req = {g for g in args.claim_graphs.split(",") if g}
    if not req:
        return None
    claim_names = {f"{n}-{v}" for n, v in TC.CLAIM_INGRAM}
    if not (req & claim_names):
        # exploration-graph smoke of the claim code path — never evidence
        return "smoke"
    problems = []
    if req != CLAIM_FROZEN["graphs"]:
        problems.append(f"graphs {sorted(req)} != frozen "
                        f"{sorted(CLAIM_FROZEN['graphs'])} (all four, one shot)")
    if set(rungs) != CLAIM_FROZEN["rungs"]:
        problems.append(f"rungs {sorted(set(rungs))} != frozen "
                        f"{sorted(CLAIM_FROZEN['rungs'])}")
    if args.seeds != CLAIM_FROZEN["seeds"]:
        problems.append(f"seeds {args.seeds} != {CLAIM_FROZEN['seeds']}")
    if args.claim_null != CLAIM_FROZEN["claim_null"]:
        problems.append(f"claim_null {args.claim_null} != "
                        f"{CLAIM_FROZEN['claim_null']}")
    if args.claim_null_r_threshold != CLAIM_FROZEN["claim_null_r_threshold"]:
        problems.append(f"claim_null_r_threshold "
                        f"{args.claim_null_r_threshold} != "
                        f"{CLAIM_FROZEN['claim_null_r_threshold']}")
    if args.ckpt not in CLAIM_FROZEN["ckpts"]:
        problems.append(f"ckpt {args.ckpt} not in frozen official set")
    if args.limit:
        problems.append("--limit forbidden in production claim mode")
    if problems:
        raise SystemExit("claim-mode invocation violates PLAN_AMENDMENT_1 §C "
                         "(frozen): " + "; ".join(problems))
    return "production"


def eval_variant(model, data, filtered, limit=0):
    d = data
    if limit:
        d = PH.subsample_targets(data, limit)
    m = PH.eval_mrr(model, d, filtered)
    return {"mrr": m["mrr"], "hits@10": m.get("hits@10", float("nan"))}


def run_ladder(model, name, version, loader, rungs, seeds, out, anneal_steps, limit):
    data, filtered = loader()
    base = eval_variant(model, data, filtered, limit)
    TC.append_jsonl(out, {"run": "R1", "dataset": f"{name}-{version}",
                          "rung": "L0", "seed": -1, **base})
    curves = {"L0": [base["mrr"]]}
    warns = {"L0": [False]}
    laststats = {}
    for rung in rungs:
        curves[rung], warns[rung] = [], []
        for s in seeds:
            t0 = time.time()
            variant, stats = TC.ladder_variant(data, rung, seed=s,
                                               anneal_steps=anneal_steps, log=True)
            if rung == "L3":  # post-check component preservation
                before = _n_components(data.relation_graph)
                after = _n_components(variant.relation_graph)
                stats["components_before"] = before
                stats["components_after"] = after
                if before != after:
                    stats["warn_components"] = "component count changed"
            m = eval_variant(model, variant, filtered, limit)
            rec = {"run": "R1", "dataset": f"{name}-{version}", "rung": rung,
                   "seed": s, "build_sec": round(time.time() - t0, 1),
                   **m, "recovery": m["mrr"] / (base["mrr"] + 1e-12), **stats}
            TC.append_jsonl(out, rec)
            curves[rung].append(m["mrr"])
            warns[rung].append(any(k.startswith("warn") for k in stats))
            laststats[rung] = stats
            del variant
            torch.cuda.empty_cache()
    return base["mrr"], curves, warns, laststats


def _n_components(rg):
    ed = TC.RGEditor(rg, seed=0)
    comp = ed._components()
    return int(len(set(comp.tolist())))


def run_permutation_null(model, name, version, loader, n_null, out, limit, seed0=0):
    data, filtered = loader()
    base = eval_variant(model, data, filtered, limit)
    TC.append_jsonl(out, {"run": "R1", "dataset": f"{name}-{version}",
                          "rung": "L0", "seed": -1, **base})
    # boundary-layer report: how permutable is the strict (within-class) rung?
    _, strict_stats = TC.ladder_variant(data, "L4", seed=0)
    null_mrrs = []
    for s in range(seed0, seed0 + n_null):
        variant, _ = TC.ladder_variant(data, "L4loose", seed=s)
        m = eval_variant(model, variant, filtered, limit)
        null_mrrs.append(m["mrr"])
        if (s - seed0 + 1) % 25 == 0:
            TC.append_jsonl(out, {"run": "R1", "dataset": f"{name}-{version}",
                                  "rung": "null-progress", "done": s - seed0 + 1,
                                  "null_mean": float(np.mean(null_mrrs))})
        del variant
        torch.cuda.empty_cache()
    rec = {"run": "R1", "dataset": f"{name}-{version}", "rung": "perm-null",
           "n_null": n_null, "base_mrr": base["mrr"],
           "null_mean": float(np.mean(null_mrrs)),
           "null_std": float(np.std(null_mrrs)),
           "null_q": {q: float(np.percentile(null_mrrs, q)) for q in (2.5, 50, 97.5)},
           "strict_class_permutable_nodes": strict_stats.get("permutable_nodes"),
           "n_nodes": strict_stats.get("n_nodes")}
    TC.append_jsonl(out, rec)
    return base["mrr"], null_mrrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="all",
                    help="'all' or comma list from " + ",".join(RUNGS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fb-versions", default="v1,v2,v3,v4")
    ap.add_argument("--wn-null", type=int, default=1000,
                    help="permutation-null draws per WN version (0 = skip)")
    ap.add_argument("--wn-versions", default="v1,v2,v3,v4")
    ap.add_argument("--explore", default="nl-25,wk-25",
                    help="exploration-set graphs for the exploratory "
                         "out-of-family replication ('' = skip)")
    ap.add_argument("--claim-graphs", default="",
                    help="amendment-1 §C claim-set graphs for the one-shot "
                         "C1-OOF confirmation, e.g. 'nl-50,nl-75,nl-100,"
                         "wk-75' ('' = skip; requires the prereg-v1 tag)")
    ap.add_argument("--claim-null", type=int, default=200,
                    help="permutation-null draws for small-|R| claim graphs "
                         "(amendment-1 C-7)")
    ap.add_argument("--claim-null-r-threshold", type=int, default=30,
                    help="base-relation count below which the C-7 null "
                         "boundary layer runs")
    ap.add_argument("--ckpt", default="ultra_3g",
                    help="frozen official ckpt: ultra_3g (primary) or "
                         "ultra_4g / ultra_50g (amendment-1 §D replication)")
    ap.add_argument("--anneal-steps", type=int, default=2000000)
    ap.add_argument("--limit", type=int, default=0,
                    help="subsample test targets (sanity dry-run only)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    TC.set_all_seeds(0)
    rungs = RUNGS if args.ladder == "all" else args.ladder.split(",")
    claim_mode = validate_claim_invocation(args, rungs)
    run_id = args.run_id or ("R1" if not args.limit else "R1_dryrun")
    d = args.out_dir or TC.run_dir(run_id)
    out = os.path.join(d, "r1_ladder_results.jsonl")
    ckpt_path = args.ckpt if os.path.sep in args.ckpt else os.path.join(
        PH.ULTRA_DIR, "ckpts", args.ckpt + ".pth")
    TC.write_run_meta(run_id, {"run": "R1", "rungs": rungs, "seeds": args.seeds,
                               "fb_versions": args.fb_versions,
                               "wn_null": args.wn_null,
                               "explore": args.explore,
                               "claim_graphs": args.claim_graphs,
                               "claim_mode": claim_mode,
                               "claim_null": args.claim_null,
                               "claim_null_r_threshold":
                                   args.claim_null_r_threshold,
                               "anneal_steps": args.anneal_steps,
                               "limit": args.limit,
                               "ckpt": args.ckpt + " (frozen)",
                               "ckpt_path": ckpt_path,
                               "protocol": "ULTRA official filtered ranking; "
                                           "claim graphs per "
                                           "PLAN_AMENDMENT_1 §C/§D"},
                      seed=0)

    model = PH.load_model(ckpt_path)
    seeds = list(range(args.seeds))
    summary = {"datasets": {}, "boundary": {}, "explore": {}, "claim": {}}

    with TC.GpuHours(run_id):
        for v in [x for x in args.fb_versions.split(",") if x]:
            base, curves, warns, _ = run_ladder(
                model, "fb", v, lambda v=v: PH.load_inductive("fb", v),
                rungs, seeds, out, args.anneal_steps, args.limit)
            summary["datasets"][f"fb-{v}"] = {
                "base_mrr": base,
                "recovery": {r: [m / (base + 1e-12) for m in ms]
                             for r, ms in curves.items()},
                "warned": warns}
        for g in [x for x in args.explore.split(",") if x]:
            name, ver = g.split("-")
            base, curves, warns, _ = run_ladder(
                model, name, ver, lambda n=name, vv=ver: TC.load_ingram(n, vv),
                rungs, seeds, out, args.anneal_steps, args.limit)
            summary["explore"][g] = {
                "base_mrr": base, "label": "exploratory-not-claim-evidence",
                "recovery": {r: [m / (base + 1e-12) for m in ms]
                             for r, ms in curves.items()},
                "warned": warns}
        for g in [x for x in args.claim_graphs.split(",") if x]:
            name, ver = g.split("-")
            loader = lambda n=name, vv=ver: TC.load_ingram(
                n, vv, allow_claim=(claim_mode == "production"))
            data, _f = loader()
            n_rel_doubled = int(data.num_relations)
            census = {"n_rel_doubled": n_rel_doubled,
                      "n_rel_base": n_rel_doubled // 2,
                      "n_entities": int(data.num_nodes),
                      "n_test_triples": int(data.target_edge_index.shape[1])}
            del data, _f
            base, curves, warns, lstats = run_ladder(
                model, name, ver, loader,
                rungs, seeds, out, args.anneal_steps, args.limit)
            rec = {r: [m / (base + 1e-12) for m in ms]
                   for r, ms in curves.items()}
            med = {r: float(np.median(v)) for r, v in rec.items()}
            l4s = lstats.get("L4", {})
            perm_frac = (None if "L4" not in lstats else
                         l4s.get("permutable_nodes", 0) /
                         max(l4s.get("n_nodes", 1), 1))
            null_block, underpowered, below_anom = None, False, False
            if census["n_rel_base"] < args.claim_null_r_threshold \
                    and args.claim_null:
                nb, nulls = run_permutation_null(
                    model, name, ver, loader, args.claim_null, out,
                    args.limit, seed0=1000)
                lo, hi = (float(np.percentile(nulls, 2.5)),
                          float(np.percentile(nulls, 97.5)))
                # two-sided (amendment review C-7 fix): underpowered only when
                # L0 sits INSIDE the null band (indistinguishable from
                # relabel); L0 below the band is adverse-direction evidence
                # and stays powered, flagged as an anomaly.
                underpowered = lo <= nb <= hi
                below_anom = nb < lo
                null_block = {"n_null": args.claim_null, "base_mrr": nb,
                              "null_q2.5": lo, "null_q97.5": hi,
                              "l0_within_null_band": underpowered,
                              "below_null_anomaly": below_anom}
            low_info = None if perm_frac is None else bool(perm_frac < 0.10)
            summary["claim"][g] = {
                "label": ("claim-set-C1-OOF (PLAN_AMENDMENT_1, prereg-v1)"
                          if claim_mode == "production" else
                          "claim-code-path-SMOKE on exploration graph — "
                          "NOT claim evidence"),
                "ckpt": args.ckpt,
                "base_mrr": base, **census,
                "recovery": rec, "warned": warns,
                "median_recovery": med,
                "mean_recovery": {r: float(np.mean(v))
                                  for r, v in rec.items()},
                "l4_permutable_nodes": l4s.get("permutable_nodes"),
                "l4_permutable_frac": (None if perm_frac is None else
                                       round(float(perm_frac), 4)),
                "primary_pass_C5": bool(med.get("L4loose", 1.0) <= 0.5),
                "mechanism_C6": {
                    "l4_median": med.get("L4"),
                    "l4_minus_l4loose": (None if "L4" not in med or
                                         "L4loose" not in med else
                                         med["L4"] - med["L4loose"]),
                    "low_information": low_info,
                    "pass": (None if low_info is not False or
                             "L4" not in med or "L4loose" not in med
                             else bool(med["L4"] >= 0.75 and
                                       med["L4"] - med["L4loose"] >= 0.3)),
                },
                "null_boundary_C7": null_block,
                "underpowered": bool(underpowered)}
        if args.wn_null:
            for v in [x for x in args.wn_versions.split(",") if x]:
                base, nulls = run_permutation_null(
                    model, "wn", v, lambda v=v: PH.load_inductive("wn", v),
                    args.wn_null, out, args.limit)
                summary["boundary"][f"wn-{v}"] = {
                    "base_mrr": base,
                    "null_mean_recovery": float(np.mean(nulls) / (base + 1e-12))}

    # mechanical pass line. L2/L3 variants
    # whose anneal emitted ANY warn (census residual / component change) are
    # EXCLUDED from the structure-preserving evidence (review round-2
    # condition); L4/L4loose are exact by construction.
    verdicts = {}
    for ds, info in summary["datasets"].items():
        drops, excluded = {}, {}
        for r in ("L2", "L3", "L4", "L4loose"):
            if r not in info["recovery"]:
                continue
            vals = [m for m, w in zip(info["recovery"][r],
                                      info["warned"].get(r, []))
                    if not (w and r in ("L2", "L3"))]
            excluded[r] = len(info["recovery"][r]) - len(vals)
            if vals:
                drops[r] = 1.0 - float(np.mean(vals))
        best_rung = max(drops, key=drops.get) if drops else None
        verdicts[ds] = {"max_drop_structure_preserving": drops,
                        "excluded_warned_variants": excluded,
                        "passes_C1_line": bool(drops and drops[best_rung] >= 0.5),
                        "at_rung": best_rung}
    summary["mechanical_passline"] = verdicts
    if summary["claim"]:
        powered = [g for g, i in summary["claim"].items()
                   if not i["underpowered"]]
        n_pass = sum(1 for g in powered
                     if summary["claim"][g]["primary_pass_C5"])
        blocked = len(powered) <= 1
        fb_pass = [ds for ds, v in verdicts.items() if v["passes_C1_line"]]
        summary["claim_verdict_C1_OOF"] = {
            "protocol": "PLAN_AMENDMENT_1 C-5/C-7 (mechanical)",
            "claim_mode": claim_mode,
            "smoke": claim_mode != "production",
            "ckpt": args.ckpt,
            "graphs_run": sorted(summary["claim"]),
            "powered": sorted(powered),
            "underpowered": sorted(set(summary["claim"]) - set(powered)),
            "n_pass_primary": int(n_pass),
            "n_powered": len(powered),
            "mechanical_status": (
                "blocked" if blocked else
                "confirmed" if n_pass / len(powered) >= 0.75 else
                "partial" if n_pass / len(powered) >= 0.5 else
                "not_confirmed"),
            # D-4 within-run replication read (only when FB in-family ladders
            # ran in the same invocation; cross-run aggregation is the
            # collector/audit's job)
            "replicates_D4": (None if not verdicts else {
                "ckpt": args.ckpt,
                "fb_pass": sorted(fb_pass),
                "fb_n": len(verdicts),
                "status": ("yes" if len(fb_pass) >= 3 and not blocked and
                           n_pass / max(len(powered), 1) >= 0.75 else
                           "partial" if (fb_pass or n_pass) else "no")})}
    summary["note"] = ("positive/negative here is a mechanical threshold "
                       "comparison. Exploration-set rows are "
                       "exploratory and never claim evidence.")
    with open(os.path.join(d, "r1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("R1 done ->", d, flush=True)


if __name__ == "__main__":
    main()
