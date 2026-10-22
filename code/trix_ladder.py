"""TRIX cross-backbone matched-control ladder (PLAN_AMENDMENT_2 §C).

Production claim runner mirroring `r1_ladder.py`, ported to the TRIX backbone.
The instrument is unchanged in spirit; only the relation-graph representation
differs (TRIX's `data.relation_adj` = dict of 4 sub-graphs hh/ht/th/tt over
relation ids, vs ULTRA's single 4-typed relation graph). The matched-control
intervention (a single global relation-id permutation pi applied across all 4
sub-graphs, queries/entities/GT/filter fixed) lives in `trix_harness.py`.

Rungs (claim mode, exact-construction only): L1 / L4 / L4loose / L5.
  L4      within-degree-profile-class relabel (role binding preserved)
  L4loose arbitrary relabel (role binding broken; label-free stats preserved)
  L5      per-sub-graph random endpoints (degree floor)
  L1      per-sub-graph degree-preserving configuration shuffle, mediator fixed
          (context rung; see _l1_degree_preserving for the exact invariant set,
           incl. the documented divergence from ULTRA's reciprocal-restoring L1).

Frozen invocation (§C-9): any request touching a claim (OOF) graph must match
TRIX_CLAIM_FROZEN EXACTLY, or the run is refused. FB in-family runs via --fb-versions
and carries no one-shot constraint. The four OOF graphs are REUSE-disclosed
(PLAN_AMENDMENT_2 §B-3): already consumed once by ULTRA under prereg-v1; here
they are paired replication-on-new-backbone under the same frozen thresholds,
NOT a fresh blind one-shot.

Pass line is mechanical.
"""
import os, json, time, argparse, copy
import numpy as np
import torch

import trix_harness as TH

RUNGS = ["L1", "L4", "L4loose", "L5"]

# PLAN_AMENDMENT_2 §C-9 frozen invocation (production claim mode). Mirrors
# r1_ladder.CLAIM_FROZEN. The claim GRAPHS are the four OOF graphs only; FB
# in-family is unconstrained (--fb-versions), exactly as in r1_ladder.
TRIX_CLAIM_FROZEN = {
    "graphs": {"nl-50", "nl-75", "nl-100", "wk-75"},
    "rungs": {"L1", "L4", "L4loose", "L5"},
    "seeds": 5,
    "claim_null": 200,
    "claim_null_r_threshold": 30,
    "ckpt": "entity_prediction.pth",
}
CLAIM_GRAPH_NAMES = {"nl-50", "nl-75", "nl-100", "wk-75"}


def validate_trix_claim_invocation(args, rungs):
    """Enforce the frozen §C invocation whenever a claim (OOF) graph is
    requested. Returns 'production' | 'smoke' | None. Mirrors
    r1_ladder.validate_claim_invocation."""
    req = {g for g in args.claim_graphs.split(",") if g}
    if not req:
        return None
    if not (req & CLAIM_GRAPH_NAMES):
        # exploration-graph smoke of the claim code path — never evidence
        return "smoke"
    problems = []
    if req != TRIX_CLAIM_FROZEN["graphs"]:
        problems.append(f"graphs {sorted(req)} != frozen "
                        f"{sorted(TRIX_CLAIM_FROZEN['graphs'])} (all four, one shot)")
    if set(rungs) != TRIX_CLAIM_FROZEN["rungs"]:
        problems.append(f"rungs {sorted(set(rungs))} != frozen "
                        f"{sorted(TRIX_CLAIM_FROZEN['rungs'])}")
    if args.seeds != TRIX_CLAIM_FROZEN["seeds"]:
        problems.append(f"seeds {args.seeds} != {TRIX_CLAIM_FROZEN['seeds']}")
    if args.claim_null != TRIX_CLAIM_FROZEN["claim_null"]:
        problems.append(f"claim_null {args.claim_null} != "
                        f"{TRIX_CLAIM_FROZEN['claim_null']}")
    if args.claim_null_r_threshold != TRIX_CLAIM_FROZEN["claim_null_r_threshold"]:
        problems.append(f"claim_null_r_threshold {args.claim_null_r_threshold} "
                        f"!= {TRIX_CLAIM_FROZEN['claim_null_r_threshold']}")
    if os.path.basename(args.ckpt) != TRIX_CLAIM_FROZEN["ckpt"]:
        problems.append(f"ckpt {args.ckpt} != frozen {TRIX_CLAIM_FROZEN['ckpt']}")
    if args.limit:
        problems.append("--limit forbidden in production claim mode")
    if problems:
        raise SystemExit("claim-mode invocation violates PLAN_AMENDMENT_2 §C "
                         "(frozen): " + "; ".join(problems))
    return "production"


# ---------------------------------------------------------------- L1 rung ----

def _l1_degree_preserving(data, seed):
    """L1 context rung: per-sub-graph degree-preserving configuration shuffle.

    For each sub-graph, permute the DST endpoints among edges while keeping SRC
    and the mediator edge_type (shared entity id) fixed per edge position.

    PRESERVED EXACTLY (per sub-graph): each relation's out-degree AND in-degree
    (dst is a reordering of the same multiset, so per-node in-degree is
    unchanged), edge count, and every edge's mediator edge_type + order.
    NOT PRESERVED (documented divergence from ULTRA's claim-mode L1, which uses
    the RGEditor to also restore reciprocal-pair counts): the ht<->th
    reciprocal-pair structure, and self-loop/multi-edge avoidance. L1 is a
    CONTEXT rung (mild-rewiring reference), not the matched pair; the claim-
    bearing contrast is L4 vs L4loose. This divergence is disclosed, not hidden.
    """
    d = copy.copy(data)
    adj = data.relation_adj
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    new_adj = {}
    for key in TH.KEYS:
        sub = adj[key]
        new_sub = copy.copy(sub)
        ei = sub.edge_index
        if ei.shape[1] > 1:
            src = ei[0].cpu()
            dst = ei[1].cpu()
            perm = torch.randperm(dst.numel(), generator=g)
            new_sub.edge_index = torch.stack([src, dst[perm]]).to(ei.device)
        else:
            new_sub.edge_index = ei.clone()
        new_adj[key] = new_sub  # edge_type (mediator) + num_nodes unchanged
    d.relation_adj = new_adj
    stats = {"rung": "L1", "seed": int(seed),
             "n_nodes": int(adj["hh"].num_nodes),
             "edge_counts_base": TH._edge_counts(adj),
             "edge_counts_variant": TH._edge_counts(new_adj),
             "invariants_preserved": ["per_relation_out_degree",
                                      "per_relation_in_degree", "edge_count",
                                      "mediator_edge_type_and_order"],
             "invariants_not_preserved": ["reciprocal_ht_th_pair_counts",
                                          "self_loop_multi_edge_avoidance"]}
    stats["edge_counts_preserved"] = (
        stats["edge_counts_variant"] == stats["edge_counts_base"])
    return d, stats


def ladder_variant_trix(data, rung, seed):
    """Dispatch: L1 -> local degree-preserving shuffle; L4/L4loose/L5 ->
    trix_harness.trix_relation_perm (the validated global-pi intervention)."""
    if rung == "L1":
        return _l1_degree_preserving(data, seed)
    return TH.trix_relation_perm(data, rung, seed)


# ------------------------------------------------------------- data loaders ----

def load_graph(name, version, allow_claim=False):
    """FB in-family via trix_harness; NL/WK OOF via TRIX's IngramInductive.
    Claim (OOF) graphs require allow_claim=True (only production claim mode,
    after passing the frozen validator) — mirrors the r1_ladder allow_claim
    guard, so a claim graph can never be silently loaded off the claim path."""
    if name == "fb":
        return TH.load_inductive_fb(version)
    if name in ("nl", "wk"):
        gkey = f"{name}-{version}"
        if gkey in CLAIM_GRAPH_NAMES and not allow_claim:
            raise PermissionError(
                f"{gkey} is a frozen claim (OOF) graph; load only via the "
                "production claim path (PLAN_AMENDMENT_2 §C). Exploration "
                "graphs nl-25/wk-25 are unconstrained.")
        return TH.load_inductive_ingram(name, version)
    raise ValueError(f"unknown graph family {name!r}")


# ------------------------------------------------------------- ladder runner ----

def eval_variant(model, data, filtered, limit=0):
    m = TH.eval_mrr(model, data, filtered, limit=limit)
    return {"mrr": m["mrr"], "hits@10": m.get("hits@10", float("nan"))}


def run_ladder(model, name, version, loader, rungs, seeds, out, limit):
    data, filtered = loader()
    snap = TH.snapshot_relation_adj(data)
    base = eval_variant(model, data, filtered, limit)
    TH_append(out, {"run": "TRIX-R1", "dataset": f"{name}-{version}",
                    "rung": "L0", "seed": -1, **base})
    curves = {"L0": [base["mrr"]]}
    laststats = {}
    for rung in rungs:
        curves[rung] = []
        for s in seeds:
            t0 = time.time()
            variant, stats = ladder_variant_trix(data, rung, seed=s)
            m = eval_variant(model, variant, filtered, limit)
            rec = {"run": "TRIX-R1", "dataset": f"{name}-{version}", "rung": rung,
                   "seed": s, "build_sec": round(time.time() - t0, 2), **m,
                   "recovery": m["mrr"] / (base["mrr"] + 1e-12), **stats}
            TH_append(out, rec)
            curves[rung].append(m["mrr"])
            laststats[rung] = stats
            del variant
            torch.cuda.empty_cache()
    # audit field: the interventions never mutated the original relation graph
    reladj_untouched = TH.relation_adj_unchanged(data, snap)
    invol = TH.perm_involution_check(data, seed=0)
    return (base["mrr"], curves, laststats,
            {"relation_adj_untouched": bool(reladj_untouched),
             "perm_involution_ok": bool(invol["involution_ok"])})


def run_permutation_null(model, name, version, loader, n_null, out, limit,
                         seed0=1000):
    data, filtered = loader()
    base = eval_variant(model, data, filtered, limit)
    null_mrrs = []
    for s in range(seed0, seed0 + n_null):
        variant, _ = TH.trix_relation_perm(data, "L4loose", seed=s)
        m = eval_variant(model, variant, filtered, limit)
        null_mrrs.append(m["mrr"])
        del variant
        torch.cuda.empty_cache()
    lo, hi = (float(np.percentile(null_mrrs, 2.5)),
              float(np.percentile(null_mrrs, 97.5)))
    rec = {"run": "TRIX-R1", "dataset": f"{name}-{version}", "rung": "perm-null",
           "n_null": n_null, "base_mrr": base["mrr"],
           "null_q2.5": lo, "null_q50": float(np.percentile(null_mrrs, 50)),
           "null_q97.5": hi}
    TH_append(out, rec)
    return base["mrr"], null_mrrs


def TH_append(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="L1,L4,L4loose,L5")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fb-versions", default="v1,v2,v3,v4",
                    help="FB in-family versions (unconstrained; '' = skip)")
    ap.add_argument("--claim-graphs", default="",
                    help="OOF claim graphs, e.g. 'nl-50,nl-75,nl-100,wk-75' "
                         "('' = skip; frozen invocation enforced)")
    ap.add_argument("--claim-null", type=int, default=200)
    ap.add_argument("--claim-null-r-threshold", type=int, default=30)
    ap.add_argument("--ckpt", default=os.path.join(TH._TRIX, "entity_prediction.pth"))
    ap.add_argument("--limit", type=int, default=0,
                    help="subsample test targets (sanity/pre-freeze only; "
                         "forbidden in claim mode)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-id", default="TRIX_R1")
    args = ap.parse_args()

    rungs = args.ladder.split(",")
    claim_mode = validate_trix_claim_invocation(args, rungs)
    d = args.out_dir or os.path.join(TH._PILOT, "runs", args.run_id)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "trix_ladder_results.jsonl")
    meta = {"run": args.run_id, "backbone": "TRIX", "ckpt": args.ckpt,
            "rungs": rungs, "seeds": args.seeds, "fb_versions": args.fb_versions,
            "claim_graphs": args.claim_graphs, "claim_mode": claim_mode,
            "limit": args.limit, "amendment": "PLAN_AMENDMENT_2 §C",
            "protocol": "TRIX official run_entity filtered ranking; matched "
                        "control = global relation-id perm over relation_adj"}
    with open(os.path.join(d, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    model = TH.load_model()
    seeds = list(range(args.seeds))
    summary = {"meta": meta, "fb": {}, "claim": {}}

    # ---- FB in-family (unconstrained) ----
    for v in [x for x in args.fb_versions.split(",") if x]:
        base, curves, lstats, audit = run_ladder(
            model, "fb", v, lambda v=v: load_graph("fb", v),
            rungs, seeds, out, args.limit)
        rec = {r: [m / (base + 1e-12) for m in ms] for r, ms in curves.items()}
        summary["fb"][f"fb-{v}"] = {
            "base_mrr": base,
            "median_recovery": {r: float(np.median(v_)) for r, v_ in rec.items()},
            "mean_recovery": {r: float(np.mean(v_)) for r, v_ in rec.items()},
            "audit": audit,
            "passes_C5_line": bool(np.median(rec.get("L4loose", [1.0])) <= 0.5)}

    # ---- OOF claim graphs (frozen; only in production claim mode) ----
    for g in [x for x in args.claim_graphs.split(",") if x]:
        name, ver = g.split("-")
        loader = lambda n=name, vv=ver: load_graph(
            n, vv, allow_claim=(claim_mode == "production"))
        data, _f = loader()
        n_rel_doubled = int(data.num_relations)
        census = {"n_rel_doubled": n_rel_doubled, "n_rel_base": n_rel_doubled // 2,
                  "n_entities": int(data.num_nodes),
                  "n_test_triples": int(data.target_edge_index.shape[1])}
        del data, _f
        base, curves, lstats, audit = run_ladder(
            model, name, ver, loader, rungs, seeds, out, args.limit)
        rec = {r: [m / (base + 1e-12) for m in ms] for r, ms in curves.items()}
        med = {r: float(np.median(v_)) for r, v_ in rec.items()}
        l4s = lstats.get("L4", {})
        perm_frac = l4s.get("permutable_fraction")
        # low-information guard (frozen §C-5 / r1_ladder semantics): C-6 is
        # report-only when <10% of relation nodes are L4-permutable.
        low_info = None if perm_frac is None else bool(perm_frac < 0.10)
        null_block, underpowered = None, False
        if census["n_rel_base"] < args.claim_null_r_threshold and args.claim_null:
            nb, nulls = run_permutation_null(
                model, name, ver, loader, args.claim_null, out, args.limit)
            lo, hi = float(np.percentile(nulls, 2.5)), float(np.percentile(nulls, 97.5))
            underpowered = lo <= nb <= hi
            null_block = {"n_null": args.claim_null, "base_mrr": nb,
                          "null_q2.5": lo, "null_q97.5": hi,
                          "l0_within_null_band": bool(underpowered)}
        summary["claim"][g] = {
            "label": ("claim-set cross-backbone (PLAN_AMENDMENT_2, prereg-v2)"
                      if claim_mode == "production" else
                      "claim-code-path SMOKE — NOT claim evidence"),
            "backbone": "TRIX", "ckpt": os.path.basename(args.ckpt),
            "base_mrr": base, **census, "audit": audit,
            "median_recovery": med,
            "mean_recovery": {r: float(np.mean(v_)) for r, v_ in rec.items()},
            "l4_permutable_fraction": perm_frac,
            "primary_pass_C5": bool(med.get("L4loose", 1.0) <= 0.5),
            "mechanism_C6": {
                "l4_median": med.get("L4"),
                "l4_minus_l4loose": (None if "L4" not in med or "L4loose" not in med
                                     else med["L4"] - med["L4loose"]),
                "low_information": low_info,
                # DISCLOSED POST-FREEZE FIX (2026-07-09): the guard
                # `low_info is not False` was dropped in the original port
                # (r1_ladder.py has it), so the prereg-v2 production summary
                # (runs/TRIX_CLAIM_prereg_v2, nl-100 perm_frac 0.057) labeled
                # C-6 pass=True where frozen §C-5 requires low-information
                # (report-only). Raw evals and primary C-5 are unaffected;
                # collection recomputes labels per frozen semantics.
                "pass": (None if low_info is not False or
                         "L4" not in med or "L4loose" not in med
                         else bool(med["L4"] >= 0.75 and med["L4"] - med["L4loose"] >= 0.3)),
            },
            "null_boundary_C7": null_block, "underpowered": bool(underpowered)}

    # ---- mechanical verdict (pending audit) ----
    if summary["claim"]:
        powered = [g for g, i in summary["claim"].items() if not i["underpowered"]]
        n_pass = sum(1 for g in powered if summary["claim"][g]["primary_pass_C5"])
        fb_pass = [k for k, v in summary["fb"].items() if v["passes_C5_line"]]
        summary["cross_backbone_verdict"] = {
            "protocol": "PLAN_AMENDMENT_2 C-5 (mechanical)",
            "claim_mode": claim_mode, "smoke": claim_mode != "production",
            "backbone": "TRIX", "fb_pass": sorted(fb_pass),
            "powered_oof": sorted(powered),
            "n_pass_oof": int(n_pass), "n_powered_oof": len(powered),
            "mechanical_status": (
                "blocked" if len(powered) <= 1 else
                "confirmed" if (len(fb_pass) >= 3 and n_pass / len(powered) >= 0.75) else
                "partial" if n_pass / max(len(powered), 1) >= 0.5 else "not_confirmed")}
    summary["note"] = ("mechanical thresholds only. Exploration "
                       "and --limit rows are never claim evidence.")
    with open(os.path.join(d, "trix_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("TRIX_R1 done ->", d, flush=True)


if __name__ == "__main__":
    main()
