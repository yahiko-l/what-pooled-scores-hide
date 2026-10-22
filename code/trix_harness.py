"""TRIX entity-prediction harness for the matched-control ladder instrument.

Mirrors pilot_harness.py (the ULTRA harness) for the TRIX KG-foundation-model
backbone, so the same "matched-control ladder" (tunit_common.ladder_variant)
can be ported cross-backbone. TRIX differs from ULTRA structurally:

  * ULTRA relation graph:  ONE typed graph `data.relation_graph` with
    edge_type in {0,1,2,3} (the 4 head/tail interaction types).
  * TRIX relation graph:   `data.relation_adj` = dict of FOUR PyG `Data`
    sub-graphs keyed hh/ht/th/tt.  Each sub-graph's `edge_index` is over
    relation ids (num_nodes = num_relations); its `edge_type` is the shared
    ENTITY node id (NOT an interaction type).  The 4 keys ARE the 4
    interaction types.  (TRIX/src/trix/tasks.py::build_relation_graph.)

The entity model (TRIX/src/trix/models_entity.py:86) reads `data.relation_adj`
and queries relation representations by the query relation id
(`h_index = batch[:,0,2]`).  So permuting relation_adj node labels while the
query relation ids stay fixed produces exactly the intended
relation-identity <-> structural-role desync -- the same instrument the
ULTRA harness realises via rg_clone_into on the single typed relation_graph.

INTEGRITY: nothing under TRIX/src or the .pth checkpoints is modified. This
file only READS them.  The eval MRR is TRIX's own ranking code
(run_entity.test) verbatim.  Exploratory, non-claim.
"""
import os
import sys
import copy

# --- env for TRIX's JIT-compiled rspmm CUDA extension (defensive: the caller
# is expected to export these, but set sane defaults so a bare invocation
# still finds the pre-built extension). Must be set BEFORE importing trix.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
_PILOT = os.path.dirname(os.path.abspath(__file__))
_TRIX = os.path.join(_PILOT, "TRIX")
os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR",
    os.path.join(_TRIX, "trix_output", "torch_extensions"))
_cuda_bin = "/usr/local/cuda/bin"
if _cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _cuda_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
from torch_geometric.data import Data

# TRIX source (read-only reference) on the path.
_TRIX_SRC = os.path.join(_TRIX, "src")
if _TRIX_SRC not in sys.path:
    sys.path.insert(0, _TRIX_SRC)

from trix import util as trix_util          # noqa: E402
from trix.models_entity import TRIX          # noqa: E402
from trix.datasets import FB15k237Inductive, NLIngram, WKIngram  # noqa: E402
import run_entity                            # noqa: E402  (TRIX's own eval loop)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONFIG = os.path.join(_TRIX, "config", "run_entity_inductive_local.yaml")
CKPT = os.path.join(_TRIX, "entity_prediction.pth")
DATA_ROOT = os.path.join(_TRIX, "trix-kg-datasets") + os.sep

# The 4 interaction sub-graphs in a FIXED order (sorted), so degree-profile
# columns and every permutation are deterministic and reproducible.
KEYS = ("hh", "ht", "th", "tt")

# Metric list mirrors run_entity_inductive_local.yaml exactly, so the numbers
# match TRIX's own reproduction run.
_METRICS = ["mr", "mrr", "hits@1", "hits@3", "hits@10", "hits@10_50"]


class _Quiet:
    """A logger stub: run_entity.test only ever calls .warning()."""
    def warning(self, *a, **k):
        pass


def _model_cfg():
    """Read the model block from the official local config (single source of
    truth, mirrors run_entity.py). The YAML is jinja-templated on
    dataset/version/gpus/epochs/bpe/ckpt; we render with a dummy context --
    only cfg.model is consumed here."""
    ctx = {"dataset": "FB15k237Inductive", "version": "v1", "gpus": "[0]",
           "epochs": "0", "bpe": "null", "ckpt": CKPT}
    return trix_util.load_config(CONFIG, context=ctx)


# ------------------------------------------------------------------ TASK 1 ----

def load_model():
    """Build TRIX (relation_model + entity_model_1 + entity_model_2) from the
    local config's model block, load entity_prediction.pth, -> cuda:0.eval().
    Constructed identically to run_entity.py (TRIX/src/run_entity.py:297)."""
    cfg = _model_cfg()
    model = TRIX(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_1_cfg=cfg.model.entity_model_1,
        entity_model_2_cfg=cfg.model.entity_model_2,
    )
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state["model"])
    return model.to(DEVICE).eval()


def load_inductive_fb(version="v1"):
    """Load FB15k237Inductive via TRIX's own dataset (auto-builds relation_adj
    through the build_relation_graph pre_transform). Returns
    (test_data, test_filtered_data), both on cuda:0.

    test_filtered_data is built EXACTLY as run_entity.py's non-Ingram
    inductive branch (TRIX/src/run_entity.py:321-324): inference edges + test
    target edges."""
    ds = FB15k237Inductive(root=DATA_ROOT, version=version)
    test_data = ds[2]
    fe = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
    ft = torch.cat([test_data.edge_type, test_data.target_edge_type])
    filtered = Data(edge_index=fe, edge_type=ft, num_nodes=test_data.num_nodes)
    return test_data.to(DEVICE), filtered.to(DEVICE)


def load_inductive_ingram(name, version):
    """Load an InGram NL/WK inductive graph via TRIX's own IngramInductive
    dataset (auto-builds relation_adj via the build_relation_graph
    pre_transform). Returns (test_data, test_filtered_data) on cuda:0.

    test_filtered_data is built EXACTLY as run_entity.py's Ingram branch
    (TRIX/src/run_entity.py:314-317): inference edges + valid targets + test
    targets (Ingram valid/test share the inference graph, valid_on_inf=True)."""
    cls = {"nl": NLIngram, "wk": WKIngram}[name]
    ds = cls(root=DATA_ROOT, version=str(version))
    valid_data, test_data = ds[1], ds[2]
    fe = torch.cat([valid_data.edge_index, valid_data.target_edge_index,
                    test_data.target_edge_index], dim=1)
    ft = torch.cat([valid_data.edge_type, valid_data.target_edge_type,
                    test_data.target_edge_type])
    filtered = Data(edge_index=fe, edge_type=ft, num_nodes=test_data.num_nodes)
    return test_data.to(DEVICE), filtered.to(DEVICE)


def _subsample_targets(data, n, seed=0):
    """Seeded subsample of the target (query) triples for a speed-limited
    smoke. relation_adj / entity graph / filter are untouched."""
    m = data.target_edge_index.shape[1]
    if m <= n:
        return data
    d = copy.copy(data)
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(m, generator=g)[:n].to(data.target_edge_index.device)
    d.target_edge_index = data.target_edge_index[:, idx]
    d.target_edge_type = data.target_edge_type[idx]
    return d


def eval_mrr(model, data, filtered, limit=0, batch_size=32, seed=0):
    """Filtered entity-prediction MRR via TRIX's OWN ranking (run_entity.test):
    batches target triples, tail then head prediction with all-negative
    candidates, tasks.compute_ranking with the filtered mask, aggregates
    mrr = (1/ranking).mean() and hits@k over both directions.

    limit>0 seed-subsamples target triples for speed (limit=0 -> full test).
    Returns the full metric dict as floats (incl. 'mrr', 'hits@10')."""
    d = _subsample_targets(data, limit, seed) if limit and limit > 0 else data
    from easydict import EasyDict
    cfg = EasyDict({
        "train": {"batch_size": int(batch_size), "gpus": [0]},
        "task": {"metric": list(_METRICS)},
    })
    metrics = run_entity.test(cfg, model, d, device=DEVICE, logger=_Quiet(),
                              filtered_data=filtered, return_metrics=True)
    return {k: float(v) for k, v in metrics.items()}


# ------------------------------------------------------------------ TASK 2 ----

def relgraph_degree_profile(relation_adj):
    """(n, 8) degree profile per relation id from the dict-of-4 relation graph,
    mirroring tunit_common.rg_degree_profile applied to TRIX's 4 sub-graphs.

    For each key k in FIXED order KEYS: out-degree = #edges with the relation
    as edge_index[0]; in-degree = #edges as edge_index[1]. Column layout:
    (out_hh,out_ht,out_th,out_tt, in_hh,in_ht,in_th,in_tt) -- the same
    out-first / in-second layout as rg_degree_profile."""
    n = int(relation_adj["hh"].num_nodes)
    prof = torch.zeros(n, 8, dtype=torch.long)
    for k, key in enumerate(KEYS):
        assert int(relation_adj[key].num_nodes) == n, "sub-graph n mismatch"
        ei = relation_adj[key].edge_index.cpu()
        if ei.numel() == 0:
            continue
        ones = torch.ones(ei.shape[1], dtype=torch.long)
        prof[:, k].scatter_add_(0, ei[0], ones)       # out-degree
        prof[:, 4 + k].scatter_add_(0, ei[1], ones)   # in-degree
    return prof


def _edge_counts(relation_adj):
    return {key: int(relation_adj[key].edge_index.shape[1]) for key in KEYS}


def _apply_perm(data, perm):
    """Deep-enough copy: fresh entity-Data store (copy.copy) + a NEW
    relation_adj dict whose 4 sub-graphs each get a NEW permuted edge_index
    ([perm[ei[0]], perm[ei[1]]]); edge_type (entity node) + num_nodes +
    num_relations kept UNCHANGED. The ORIGINAL data.relation_adj is untouched
    (PyG Data.__copy__ isolates the store; new tensors, no in-place writes).
    Entity triples / queries / ground truth kept fixed."""
    d = copy.copy(data)
    adj = data.relation_adj
    dev = adj["hh"].edge_index.device
    p = torch.as_tensor(np.asarray(perm), dtype=torch.long, device=dev)
    new_adj = {}
    for key in KEYS:
        sub = adj[key]
        new_sub = copy.copy(sub)
        ei = sub.edge_index
        if ei.numel() == 0:
            new_sub.edge_index = ei.clone()
        else:
            new_sub.edge_index = torch.stack([p[ei[0]], p[ei[1]]])
        new_adj[key] = new_sub          # edge_type / num_nodes / num_relations unchanged
    d.relation_adj = new_adj
    return d


def _apply_random_endpoints(data, seed):
    """L5 floor control: each sub-graph independently gets uniform-random
    relation-id endpoints (same edge count per sub-graph). Destroys degrees.
    Mirrors pilot_harness.random_relgraph, applied per sub-graph in fixed key
    order (torch.Generator seeded)."""
    d = copy.copy(data)
    adj = data.relation_adj
    n = int(adj["hh"].num_nodes)
    dev = adj["hh"].edge_index.device
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    new_adj = {}
    for key in KEYS:
        sub = adj[key]
        new_sub = copy.copy(sub)
        E = int(sub.edge_index.shape[1])
        if E == 0:
            new_sub.edge_index = sub.edge_index.clone()
        else:
            src = torch.randint(n, (E,), generator=g)
            dst = torch.randint(n, (E,), generator=g)
            new_sub.edge_index = torch.stack([src, dst]).to(dev)
        new_adj[key] = new_sub
    d.relation_adj = new_adj
    return d


def trix_relation_perm(data, rung, seed):
    """The matched-control intervention on data.relation_adj (dict-of-4).

    Mirrors tunit_common.ladder_variant exactly, but on TRIX's 4 sub-graphs:
      L4      : within-equal-(8-dim-degree-profile)-class shuffle; singleton
                classes fixed. A SINGLE global perm over relation ids is then
                applied to all 4 sub-graphs (a relabel that preserves every
                structural statistic; breaks only identity<->role binding).
      L4loose : perm = default_rng(seed).permutation(n) over ALL relation ids.
      L5      : per-sub-graph uniform-random endpoints (floor control;
                destroys degrees).
    L1 is not required for this smoke.

    Returns (data_variant, stats). Does NOT rebuild relation_adj from a
    modified entity graph; does NOT permute edge_type; keeps entity triples,
    queries and ground truth fixed.
    """
    adj = data.relation_adj
    n = int(adj["hh"].num_nodes)
    stats = {"rung": rung, "seed": int(seed), "n_nodes": n,
             "edge_counts_base": _edge_counts(adj)}

    if rung in ("L4", "L4loose"):
        rng = np.random.default_rng(seed)
        perm = np.arange(n)
        if rung == "L4":
            prof = relgraph_degree_profile(adj)
            classes = {}
            for i in range(n):
                classes.setdefault(tuple(prof[i].tolist()), []).append(i)
            n_permutable = 0
            for members in classes.values():
                if len(members) > 1:
                    n_permutable += len(members)
                    shuffled = rng.permutation(members)
                    for a, b in zip(members, shuffled):
                        perm[a] = b
            stats["permutable_nodes"] = int(n_permutable)
            stats["n_profile_classes"] = int(len(classes))
            stats["permutable_fraction"] = float(n_permutable) / max(n, 1)
        else:  # L4loose
            perm = rng.permutation(n)
            stats["permutable_nodes"] = int(n)
            stats["permutable_fraction"] = 1.0
        d = _apply_perm(data, perm)
        stats["edge_counts_variant"] = _edge_counts(d.relation_adj)
        stats["edge_counts_preserved"] = (
            stats["edge_counts_variant"] == stats["edge_counts_base"])
        return d, stats

    if rung == "L5":
        d = _apply_random_endpoints(data, seed)
        stats["permutable_nodes"] = int(n)   # not meaningful for the floor control
        stats["permutable_fraction"] = 1.0
        stats["edge_counts_variant"] = _edge_counts(d.relation_adj)
        stats["edge_counts_preserved"] = (
            stats["edge_counts_variant"] == stats["edge_counts_base"])
        return d, stats

    raise ValueError(f"unsupported rung {rung!r} (this smoke supports L4/L4loose/L5)")


# ----------------------------------------------------------- sanity checks ----

def perm_involution_check(data, seed):
    """SANITY CHECK (TASK 2): a single global permutation applied across all 4
    sub-graphs, then relabelled back by perm^{-1}, must recover the original
    edge_index EXACTLY (an involution/identity check on the relabel op)."""
    adj = data.relation_adj
    n = int(adj["hh"].num_nodes)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    inv = np.empty(n, dtype=perm.dtype)
    inv[perm] = np.arange(n)
    dev = adj["hh"].edge_index.device
    p = torch.as_tensor(perm, dtype=torch.long, device=dev)
    pinv = torch.as_tensor(inv, dtype=torch.long, device=dev)
    per_key, ok = {}, True
    for key in KEYS:
        ei = adj[key].edge_index
        if ei.numel() == 0:
            per_key[key] = True
            continue
        permuted = torch.stack([p[ei[0]], p[ei[1]]])
        recovered = torch.stack([pinv[permuted[0]], pinv[permuted[1]]])
        match = bool(torch.equal(recovered, ei))
        per_key[key] = match
        ok = ok and match
    return {"seed": int(seed), "involution_ok": bool(ok), "per_key": per_key}


def snapshot_relation_adj(data):
    """CPU clone of every sub-graph edge_index, to prove afterwards that the
    interventions never mutate the original relation graph."""
    return {key: data.relation_adj[key].edge_index.detach().cpu().clone()
            for key in KEYS}


def relation_adj_unchanged(data, snap):
    return all(torch.equal(data.relation_adj[key].edge_index.detach().cpu(), snap[key])
               for key in KEYS)


# --------------------------------------------------------------- self-test ----

if __name__ == "__main__":
    print("[selftest] loading TRIX + fb-v1 ...", flush=True)
    model = load_model()
    test_data, filtered = load_inductive_fb("v1")
    n = int(test_data.relation_adj["hh"].num_nodes)
    print(f"[selftest] relation-graph nodes n = {n}; "
          f"edge counts = {_edge_counts(test_data.relation_adj)}", flush=True)
    base = eval_mrr(model, test_data, filtered, limit=0)
    print(f"[selftest] base MRR (limit=0) = {base['mrr']:.6f}  "
          f"hits@10 = {base['hits@10']:.6f}  (expect ~0.5148)", flush=True)
    inv = perm_involution_check(test_data, 0)
    print(f"[selftest] involution check seed0: {inv}", flush=True)
    _, s4 = trix_relation_perm(test_data, "L4", 0)
    print(f"[selftest] L4 permutable {s4['permutable_nodes']}/{s4['n_nodes']} "
          f"({s4['permutable_fraction']:.3f}); edge_counts_preserved="
          f"{s4['edge_counts_preserved']}", flush=True)
