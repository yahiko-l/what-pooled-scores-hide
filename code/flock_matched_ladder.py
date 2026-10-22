"""flock_matched_ladder.py — PLAN_AMENDMENT_3 production runner (FLOCK, 4 cells).

Wm/Xm in FLOCK's D-ii semantics (message edge_type relabel, query/filter fixed),
base-side Pi, atomic lift, CRN per draw (crn_seed == draw; L0/Wm/Xm each after
full RNG reset). §E-7 strata are NOT computed for FLOCK (black-box rank access
would require source modification — frozen scope, disclosed in §E-7); coverage
and msg_mass are reported instead. Post-C1: transactional rows, fail-closed
certificates, full snapshots, budget ledger, resume, smoke isolation (nl-25).

Production: CUDA_VISIBLE_DEVICES=<id> python flock_matched_ladder.py
Smoke (§C-9): python flock_matched_ladder.py --smoke --draws 3
Selftest: python flock_matched_ladder.py --selftest
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import flock_ladder as FL
import flock_harness as FH
import matched_pairs as MP

BACKBONE = "flock"
DEVICE = FL.DEVICE


def query_coverage(test, msg_et_after):
    q = torch.unique(test.target_edge_type)
    present = set(int(x) for x in torch.unique(msg_et_after).tolist())
    covered = sum(1 for r in q.tolist() if int(r) in present)
    return round(covered / max(int(q.numel()), 1), 6)


def run_cell(cfg, model, gkey, test, filt, draws, mode, judgeable, flags):
    label = MP.PROD_LABEL[BACKBONE] if mode == "production" else MP.SMOKE_LABEL
    ckpt = MP.CKPTS[BACKBONE][0]
    common = {"run": MP.RUN_TAG, "backbone": BACKBONE, "ckpt": ckpt,
              "graph": gkey, "mode": mode, "label": label,
              "judgeable": judgeable, "flags": flags}
    rows = []
    logger = FH.util.get_root_logger(file=False)
    snap = {"ei": test.edge_index.clone(), "et": test.edge_type.clone(),
            "tei": test.target_edge_index.clone(),
            "tet": test.target_edge_type.clone(),
            "fei": filt.edge_index.clone(), "fet": filt.edge_type.clone()}
    n_msg0 = int(test.edge_type.shape[0])
    R_base = int(test.num_relations) // 2
    num_rels = int(test.num_relations)
    rg_ei, rg_et = FL.build_relation_graph_edges(
        test.edge_index, test.edge_type, int(test.num_nodes), num_rels)
    prof = FL.rg_degree_profile(rg_ei, rg_et, num_rels)
    for d in draws:
        t0 = time.time()
        A = MP.make_arms(prof, R_base, d, BACKBONE)
        m0 = FH.eval_metrics(cfg, model, test, filt, DEVICE, logger, d)
        base = MP.check_mrr(m0["mrr"], f"{gkey} d{d} L0")
        if not (base > 0):
            raise MP.CertificateError(f"{gkey} d{d}: nonpositive L0 MRR {base}")
        rows.append({**common, "arm": "L0", "draw": int(d), "crn_seed": int(d),
                     "mrr": base, "R_base": R_base})
        mask = MP.lifted_mask(A["M"], R_base).to(test.edge_type.device)
        msg_mass = float(mask[test.edge_type].float().mean())  # §E-7 FLOCK 替代
        for arm in MP.ARMS[BACKBONE]:
            pi = MP.pi_for_arm(A, arm)
            pi_t = torch.as_tensor(np.asarray(pi), dtype=torch.long,
                                   device=test.edge_type.device)
            new_et = pi_t[test.edge_type]
            if int(new_et.shape[0]) != n_msg0:
                raise MP.CertificateError(
                    f"{gkey} {arm} d{d}: message edge count changed (§C-6(vii))")
            variant = FH.clone_data_with(test, edge_type=new_et)
            cov = query_coverage(test, new_et)
            m = FH.eval_metrics(cfg, model, variant, filt, DEVICE, logger, d)
            mrr = MP.check_mrr(m["mrr"], f"{gkey} d{d} {arm}")
            rows.append({**common, "arm": arm, "draw": int(d),
                         "crn_seed": int(d), "mrr": mrr,
                         "recovery": mrr / base, "pi_sha": A["sha"][arm],
                         "edge_count_preserved": True,
                         "query_coverage_after_perm": cov,
                         "msg_mass": msg_mass, **A["stats"],
                         "build_eval_sec": round(time.time() - t0, 1)})
            del variant
        for tensor, key in ((test.edge_index, "ei"), (test.edge_type, "et"),
                            (test.target_edge_index, "tei"),
                            (test.target_edge_type, "tet"),
                            (filt.edge_index, "fei"), (filt.edge_type, "fet")):
            if not torch.equal(tensor, snap[key]):
                raise MP.CertificateError(
                    f"{gkey} d{d}: original tensor {key} mutated (§C-6(vii))")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows, {"n_draws": len(draws)}


def done_cells(out_path, draws):
    """Exact-key completeness for resume (C2/C3): done only if L0/Wm/Xm each
    carry the CURRENT invocation's full draw set."""
    cells = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("arm") == "ERROR":
                    continue
                cells.setdefault(r.get("graph"), {}).setdefault(
                    r.get("arm"), set()).add(r.get("draw"))
    want = set(draws)
    done = set()
    for g, arms in cells.items():
        if all(arms.get(a, set()) == want for a in ("L0",) + MP.ARMS[BACKBONE]):
            done.add(g)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default=None)
    ap.add_argument("--draws", type=int, default=len(MP.DRAW_SEEDS[BACKBONE]))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print(MP.validator_selftest(BACKBONE))
        return

    run_id = args.run_id or (MP.SMOKE_RUN_PREFIX + BACKBONE if args.smoke
                             else MP.PROD_RUN_IDS[BACKBONE])
    # explicit --graphs passed to validator UNMODIFIED (C2 fix)
    graphs = (args.graphs.split(",") if args.graphs is not None
              else (MP.SMOKE_GRAPHS[BACKBONE] if args.smoke
                    else MP.GRAPHS[BACKBONE]))
    draws = list(range(args.draws))
    mode = MP.validate_matched_invocation(
        BACKBONE, graphs, draws, MP.CKPTS[BACKBONE], MP.ARMS[BACKBONE],
        args.limit, args.smoke, run_id)

    out_dir = os.path.join(MP.PILOT_DIR, "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "flock_matched_results.jsonl")
    MP.refuse_or_resume(out_path, args.resume)   # both modes (C3)
    skip = done_cells(out_path, draws) if args.resume else set()
    summary = {"status": "running", "mode": mode, "run_id": run_id,
               "graphs": graphs, "draws": draws,
               "numpy": np.__version__, "torch": torch.__version__, "cells": {}}

    model_holder = {}
    for gkey in graphs:
        cell_key = f"{BACKBONE}:{MP.CKPTS[BACKBONE][0]}:{gkey}"
        if gkey in skip:
            summary["cells"][cell_key] = {"ok": True, "resumed_skip": True}
            continue
        MP.budget_check()
        name, ver = gkey.split("-")
        cfg, test, filt = FL.build_graph(name, ver,
                                         allow_claim=(mode == "production"))
        if "model" not in model_holder:
            model_holder["model"] = FL.load_model(cfg)
        judgeable = (mode == "production") and (gkey in MP.JUDGEABLE[BACKBONE])
        err_base = {"run": MP.RUN_TAG, "backbone": BACKBONE,
                    "ckpt": MP.CKPTS[BACKBONE][0], "graph": gkey,
                    "mode": mode, "judgeable": judgeable,
                    "label": (MP.PROD_LABEL[BACKBONE]
                              if mode == "production" else MP.SMOKE_LABEL),
                    "flags": []}
        t0 = time.time()
        ok, res = MP.run_cell_with_retries(
            out_dir, cell_key,
            lambda: run_cell(cfg, model_holder["model"], gkey, test, filt,
                             draws, mode, judgeable, []),
            out_path, err_base)
        MP.log_gpu(cell_key, time.time() - t0, mode)
        summary["cells"][cell_key] = (
            {"ok": True, **res} if ok else
            {"ok": False, "error": res, "irrecoverable_ERROR": True,
             "outcome_effect": "FLOCK tier UNAVAILABLE if judgeable (§A)"})
        del test, filt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary["status"] = "done"
    summary["gpu_hours_ledger_total"] = MP.budget_check()
    with open(os.path.join(out_dir, "flock_matched_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("done ->", out_path)


if __name__ == "__main__":
    main()
