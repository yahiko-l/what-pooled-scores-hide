#!/usr/bin/env python3
"""
FLOCK matched-control smoke harness (EXPLORATORY, NON-CLAIM).

Question: does breaking RELATION role-binding collapse FLOCK's zero-shot entity
prediction, or is FLOCK invariant to it?

This harness lives OUTSIDE FLOCK/src and does NOT modify any FLOCK source or
checkpoint. It:
  (a) loads a graph exactly the way FLOCK's official runner (src_entity/run_many.py)
      does -- same config, same dataset builder, same Ingram test filter;
  (b) evaluates zero-shot entity-prediction filtered MRR by REUSING FLOCK's OWN
      ranking/eval code -- it calls run.test() verbatim, no reimplemented ranking;
  (c) applies two relation-relabel interventions on the *test* Data and re-evals.

Interventions (relation permutation, inverse-pairing preserving):
  sigma : a permutation of the BASE relation ids {0..R_base-1}
  pi    : the induced permutation on base+inverse space {0..2*R_base-1}:
            pi(r) = sigma(r)            for r <  R_base   (base)
            pi(r) = sigma(r-R_base)+R_base for r >= R_base (inverse)

  D-ii (role-binding break, ULTRA/TRIX "L4loose" analog):
        inference/message edge_type <- pi(edge_type)
        query target_edge_type      = ORIGINAL
        filter                      = ORIGINAL
     -> desyncs relation identity from structural role.
  D-i (equivariance positive control, consistent GLOBAL relabel):
        inference edge_type    <- pi
        query target_edge_type <- pi   (base-only, = sigma on that subrange)
        filter edge_type       <- pi
     -> EXPECT recovery ~ 1 (FLOCK is relation-equivariant by construction).

CRN seeding: before EACH eval we reset random/np/torch(+cuda) to the SAME fixed S
so base and variant see the same walk-randomness stream (paired / common random
numbers). FLOCK's walks are numpy-unseeded, so this is required for a fair paired
comparison. (We also set cudnn deterministic, exactly as FLOCK's own set_seed does.)

L0-vs-L0 null band: base is evaluated with 3 different walk seeds S=0,1,2; the
min/median/max base MRR is the invariance jitter band. A variant "collapses" only
if its recovery is well OUTSIDE this band.
"""
import os
import sys
import copy
import json
import time
import random
import argparse
import statistics

import numpy as np
import torch

REPO = os.environ.get(
    "FLOCK_REPO",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "FLOCK"),
)
CKPT = os.path.join(REPO, "checkpoints", "flock_entity.pth")
CONFIG = os.path.join(REPO, "src_entity", "config", "zeroshot_inductive", "n16_ensemble16.yaml")
DATASET = "NLIngram"
VERSION = 25  # NL-25 -- EXPLORATION-tier graph only (NL-50/75/100, WK-75 are OFF-LIMITS)

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flock_smoke_nl25.json")
SEEDS = [0, 1, 2]

# make FLOCK's own package + runner importable (src_entity on path -> `from flock ...` + `import run`)
sys.path.insert(0, os.path.join(REPO, "src_entity"))
from torch_geometric.data import Data  # noqa: E402
from flock import util  # noqa: E402
from flock.models import Flock  # noqa: E402
import run as flock_run  # noqa: E402  # FLOCK's OWN test() -- ranking/eval reused verbatim


def set_all_seeds(S: int):
    """Reset every RNG FLOCK's eval can touch to the SAME S (common random numbers)."""
    random.seed(S)
    np.random.seed(S)
    torch.manual_seed(S)
    torch.cuda.manual_seed_all(S)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_perms(R_base: int, device, draw_seed: int):
    """Draw sigma (base perm) with a DEDICATED rng so it is independent of walk seeding,
    then build the inverse-pairing-preserving pi over base+inverse space."""
    rng = np.random.RandomState(100000 + draw_seed)
    sigma_np = rng.permutation(R_base)
    sigma = torch.as_tensor(sigma_np, dtype=torch.long, device=device)
    pi = torch.empty(2 * R_base, dtype=torch.long, device=device)
    pi[:R_base] = sigma
    pi[R_base:] = sigma + R_base
    return sigma, pi


def clone_data_with(data: Data, **overrides) -> Data:
    """Deep-clone a PyG Data then override tensor fields, leaving the original untouched."""
    d = data.clone()
    for k, v in overrides.items():
        setattr(d, k, v)
    return d


def eval_metrics(cfg, model, test_data, filtered_data, device, logger, seed: int):
    """Seed (CRN), then call FLOCK's OWN test() and return its metrics verbatim."""
    set_all_seeds(seed)
    metrics = flock_run.test(
        cfg,
        model,
        test_data,
        device=device,
        logger=logger,
        filtered_data=filtered_data,
        return_metrics=True,
    )
    return {k: float(v.item()) for k, v in metrics.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    # --- load config/model/data exactly like run_many.py (zeroshot: epochs=0, bpe=null) ---
    ctx = {"gpus": [args.gpu], "ckpt": CKPT, "epochs": 0, "bpe": "null",
           "dataset": DATASET, "version": VERSION}
    cfg = util.load_config(CONFIG, context=ctx)
    logger = util.get_root_logger(file=False)

    assert cfg.task["name"] == "InductiveInference", cfg.task["name"]
    dataset = util.build_dataset(cfg, device=device)
    train_data, valid_data, test_data = (
        dataset[0].to(device),
        dataset[1].to(device),
        dataset[2].to(device),
    )

    model = Flock(model_cfg=cfg.model)
    state = torch.load(cfg.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    # --- Ingram test filter, mirroring run_many.py (Ingram branch) verbatim ---
    assert "Ingram" in cfg.dataset["class"], cfg.dataset["class"]
    full_inference_edges = torch.cat(
        [valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index],
        dim=1,
    )
    full_inference_etypes = torch.cat(
        [valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type]
    )
    filt = Data(
        edge_index=full_inference_edges,
        edge_type=full_inference_etypes,
        num_nodes=test_data.num_nodes,
    ).to(device)

    # --- inverse-convention diagnostics ---
    num_relations = int(test_data.num_relations)
    R_base = num_relations // 2
    max_msg = int(test_data.edge_type.max())
    min_msg = int(test_data.edge_type.min())
    max_q = int(test_data.target_edge_type.max())
    min_q = int(test_data.target_edge_type.min())
    msg_convention = "base+inverse" if max_msg >= R_base else "base-only"
    query_convention = "base-only" if max_q < R_base else "base+inverse"
    diagnostics = {
        "num_nodes_test_graph": int(test_data.num_nodes),
        "num_msg_edges(base+inv)": int(test_data.edge_index.shape[1]),
        "num_test_queries": int(test_data.target_edge_index.shape[1]),
        "Data.num_relations": num_relations,
        "R_base": R_base,
        "message_edge_type_range": [min_msg, max_msg],
        "query_target_edge_type_range": [min_q, max_q],
        "filter_edge_type_max": int(filt.edge_type.max()),
        "message_relation_convention": msg_convention,
        "query_relation_convention": query_convention,
        "inverse_rule": "inverse(r) = r + R_base (num_relations//2), per models.py:82/642",
        "permutation_applied": "pi (base+inverse, pairing-preserving) on message; "
        "D-i also applies pi to query(base->sigma) and filter",
    }
    assert max_msg < 2 * R_base, (max_msg, R_base)
    print("[diag]", json.dumps(diagnostics, indent=2), flush=True)

    keep = {  # sanity: originals must be untouched after we build variants
        "orig_msg_etype": test_data.edge_type.clone(),
        "orig_q_etype": test_data.target_edge_type.clone(),
        "orig_filt_etype": filt.edge_type.clone(),
    }

    results = {
        "status": "EXPLORATORY_SMOKE_NONCLAIM",
        "note": "Pure exploration. Not a research claim. FLOCK zero-shot entity "
        "prediction with relation-role-binding matched controls on NL-25.",
        "graph": f"{DATASET}:{VERSION} (InGram NL-25, EXPLORATION tier)",
        "config": CONFIG,
        "checkpoint": CKPT,
        "eval_code": "FLOCK src_entity/run.py::test (reused verbatim, return_metrics=True)",
        "seeds": SEEDS,
        "crn_seeding": "random/np/torch/torch.cuda all reset to same S before each eval; "
        "cudnn deterministic (as FLOCK set_seed)",
        "diagnostics": diagnostics,
        "base_L0": {},
        "D_ii_L4loose_role_break": {},
        "D_i_equivariance_control": {},
    }

    # ---- resume support: reuse any completed (section, seed) evals from a prior run ----
    _SECS = ("base_L0", "D_ii_L4loose_role_break", "D_i_equivariance_control")
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON) as _f:
                _prev = json.load(_f)
            for _sec in _SECS:
                for _s, _m in _prev.get(_sec, {}).items():
                    if isinstance(_m, dict) and _m.get("mrr") is not None:
                        results[_sec][_s] = _m
            print(f"[resume] loaded {sum(len(results[s]) for s in _SECS)} prior evals", flush=True)
        except Exception as _e:
            print(f"[resume] could not load partial ({_e}); starting fresh", flush=True)

    def _save_partial():
        with open(OUT_JSON, "w") as _f:
            json.dump(results, _f, indent=2)

    # --- L0 base: 3 walk seeds -> jitter band ---
    for S in SEEDS:
        if str(S) in results["base_L0"]:
            m = results["base_L0"][str(S)]
            print(f"[base S={S}] (cached) mrr={m['mrr']:.6f} hits@10={m['hits@10']:.6f}", flush=True)
            continue
        m = eval_metrics(cfg, model, test_data, filt, device, logger, S)
        results["base_L0"][str(S)] = m
        _save_partial()
        print(f"[base S={S}] mrr={m['mrr']:.6f} hits@10={m['hits@10']:.6f}", flush=True)

    # --- D-ii and D-i: sigma redrawn per seed ---
    for S in SEEDS:
        sigma, pi = make_perms(R_base, device, draw_seed=S)

        # D-ii: message relabeled by pi; query + filter ORIGINAL
        if str(S) in results["D_ii_L4loose_role_break"]:
            m_ii = results["D_ii_L4loose_role_break"][str(S)]
            print(f"[D-ii S={S}] (cached) mrr={m_ii['mrr']:.6f} hits@10={m_ii['hits@10']:.6f}", flush=True)
        else:
            td_ii = clone_data_with(test_data, edge_type=pi[test_data.edge_type])
            m_ii = eval_metrics(cfg, model, td_ii, filt, device, logger, S)
            results["D_ii_L4loose_role_break"][str(S)] = m_ii
            _save_partial()
            print(f"[D-ii S={S}] mrr={m_ii['mrr']:.6f} hits@10={m_ii['hits@10']:.6f}", flush=True)

        # D-i: consistent global relabel (message + query + filter all by pi)
        if str(S) in results["D_i_equivariance_control"]:
            m_i = results["D_i_equivariance_control"][str(S)]
            print(f"[D-i  S={S}] (cached) mrr={m_i['mrr']:.6f} hits@10={m_i['hits@10']:.6f}", flush=True)
        else:
            td_i = clone_data_with(
                test_data,
                edge_type=pi[test_data.edge_type],
                target_edge_type=pi[test_data.target_edge_type],
            )
            filt_i = clone_data_with(filt, edge_type=pi[filt.edge_type])
            m_i = eval_metrics(cfg, model, td_i, filt_i, device, logger, S)
            results["D_i_equivariance_control"][str(S)] = m_i
            _save_partial()
            print(f"[D-i  S={S}] mrr={m_i['mrr']:.6f} hits@10={m_i['hits@10']:.6f}", flush=True)

    # sanity: originals untouched
    assert torch.equal(test_data.edge_type, keep["orig_msg_etype"])
    assert torch.equal(test_data.target_edge_type, keep["orig_q_etype"])
    assert torch.equal(filt.edge_type, keep["orig_filt_etype"])

    # --- summaries ---
    base_mrr = [results["base_L0"][str(S)]["mrr"] for S in SEEDS]
    band = {
        "min": min(base_mrr),
        "median": statistics.median(base_mrr),
        "max": max(base_mrr),
        "spread": max(base_mrr) - min(base_mrr),
    }
    # paired per-seed recovery = variant_mrr / base_mrr(same seed)
    rec_ii = [results["D_ii_L4loose_role_break"][str(S)]["mrr"] / results["base_L0"][str(S)]["mrr"] for S in SEEDS]
    rec_i = [results["D_i_equivariance_control"][str(S)]["mrr"] / results["base_L0"][str(S)]["mrr"] for S in SEEDS]
    # the null band expressed as a recovery interval a truly-invariant variant would occupy
    band_rec = {"low": band["min"] / band["median"], "high": band["max"] / band["median"]}

    results["summary"] = {
        "base_MRR_L0_L0_jitter_band": band,
        "D_ii_recovery_per_seed": {str(S): rec_ii[i] for i, S in enumerate(SEEDS)},
        "D_ii_recovery_median": statistics.median(rec_ii),
        "D_i_recovery_per_seed": {str(S): rec_i[i] for i, S in enumerate(SEEDS)},
        "D_i_recovery_median": statistics.median(rec_i),
        "null_band_recovery_interval": band_rec,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print("\n[SUMMARY]", json.dumps(results["summary"], indent=2), flush=True)
    print(f"[written] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
