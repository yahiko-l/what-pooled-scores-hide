"""Shared infrastructure of the exact-control and matched-contrast runners: run directories
and ledgers, InGram loading, filtered-rank collection, relation-graph degree profiles and
the relabeling ladder."""
import os, sys, json, time, copy, math, random
import numpy as np
import torch

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
ULTRA_DIR = os.path.join(PILOT_DIR, "ULTRA")
if ULTRA_DIR not in sys.path:
    sys.path.insert(0, ULTRA_DIR)

from easydict import EasyDict
from torch_geometric.data import Data
from ultra import datasets as uds
from ultra import tasks

import pilot_harness as PH  # loaders and eval_mrr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS_DIR = os.path.join(PILOT_DIR, "runs")
LEDGER = os.path.join(RUNS_DIR, "gpu_hours_ledger.jsonl")

# InGram graphs load_ingram may open; the four graphs of the preregistered one-shot
# confirmation also need allow_claim=True.
ALLOWED_INGRAM = {("nl", "25"), ("wk", "25"),
                  ("nl", "50"), ("nl", "75"), ("nl", "100"), ("wk", "75")}
CLAIM_INGRAM = {("nl", "50"), ("nl", "75"), ("nl", "100"), ("wk", "75")}


# ---------------------------------------------------------------- run io ----

def run_dir(run_id):
    d = os.path.join(RUNS_DIR, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def write_run_meta(run_id, config, seed):
    d = run_dir(run_id)
    with open(os.path.join(d, "config.yaml"), "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {json.dumps(v)}\n")
    with open(os.path.join(d, "seed"), "w") as f:
        f.write(str(seed) + "\n")


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print("RESULT", json.dumps(record)[:400], flush=True)


class GpuHours:
    """Wall-clock * n_gpus ledger (budget accounting, M1 <= 40 GPU-h)."""

    def __init__(self, run_id, n_gpus=1):
        self.run_id, self.n = run_id, n_gpus

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        os.makedirs(RUNS_DIR, exist_ok=True)
        hours = (time.time() - self.t0) / 3600.0 * self.n
        append_jsonl(LEDGER, {"run": self.run_id, "gpu_hours": round(hours, 4),
                              "n_gpus": self.n, "ts": time.strftime("%F %T")})


def set_all_seeds(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


# ------------------------------------------------------------- datasets ----

def load_ingram(name, version, allow_claim=False):
    """InGram loader. Exploration set (nl-25/wk-25) loads freely; the
    amendment-1 claim set (nl-50/75/100, wk-75) additionally requires
    allow_claim=True — passed only by the r1_ladder claim mode after it
    validates the frozen PLAN_AMENDMENT_1 §C invocation. Everything else
    is forbidden."""
    if (name, str(version)) not in ALLOWED_INGRAM:
        raise PermissionError(
            f"InGram {name}-{version} is outside the allowed sets (plan §0/§2"
            f" + PLAN_AMENDMENT_1 §B); contact is forbidden.")
    if (name, str(version)) in CLAIM_INGRAM and not allow_claim:
        raise PermissionError(
            f"InGram {name}-{version} is claim-set: one-shot C1-OOF access "
            f"only via the r1_ladder claim mode (PLAN_AMENDMENT_1 §C, "
            f"prereg-v1), which passes allow_claim=True after validating "
            f"the frozen invocation.")
    cls = {"nl": uds.NLIngram, "wk": uds.WKIngram}[name]
    ds = cls(root=PH.DATA_ROOT, version=str(version))
    valid_data, test_data = ds[1], ds[2]
    # official ULTRA filtering for Ingram (script/run.py L40-45): inference
    # graph + valid targets + test targets
    fe = torch.cat([valid_data.edge_index, valid_data.target_edge_index,
                    test_data.target_edge_index], dim=1)
    ft = torch.cat([valid_data.edge_type, valid_data.target_edge_type,
                    test_data.target_edge_type])
    filtered = Data(edge_index=fe, edge_type=ft, num_nodes=test_data.num_nodes)
    return test_data.to(DEVICE), filtered.to(DEVICE)


# ------------------------------------------------- per-query rank collector ----

def collect_ranks(model, data, filtered, bs=64):
    """Official ULTRA filtered protocol (script/run.py test), but returns
    per-query ranks for both directions. GT = dataset target triples only."""
    triplets = torch.cat([data.target_edge_index,
                          data.target_edge_type.unsqueeze(0)]).t()
    t_ranks, h_ranks = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, triplets.shape[0], bs):
            batch = triplets[i:i + bs].to(DEVICE)
            t_batch, h_batch = tasks.all_negative(data, batch)
            t_pred = model(data, t_batch)
            h_pred = model(data, h_batch)
            t_mask, h_mask = tasks.strict_negative_mask(filtered, batch)
            pos_h, pos_t, _ = batch.t()
            t_ranks.append(tasks.compute_ranking(t_pred, pos_t, t_mask).cpu())
            h_ranks.append(tasks.compute_ranking(h_pred, pos_h, h_mask).cpu())
    t = torch.cat(t_ranks).float()
    h = torch.cat(h_ranks).float()
    ranks = torch.cat([t, h])
    return {"mrr": float((1 / ranks).mean()),
            "hits@10": float((ranks <= 10).float().mean()),
            "t_ranks": t.tolist(), "h_ranks": h.tolist()}


# --------------------------------------------------- structural features ----

def rg_degree_profile(rg):
    """(num_nodes, 8) in/out degree per edge type."""
    n = int(rg.num_nodes)
    prof = torch.zeros(n, 8, dtype=torch.long)
    ei, et = rg.edge_index.cpu(), rg.edge_type.cpu()
    for t in range(4):
        m = et == t
        prof[:, t].scatter_add_(0, ei[0][m], torch.ones(int(m.sum()), dtype=torch.long))
        prof[:, 4 + t].scatter_add_(0, ei[1][m], torch.ones(int(m.sum()), dtype=torch.long))
    return prof


def undirected_masks(rg):
    """Per-node undirected neighbour bitmask (numpy packed uint64)."""
    n = int(rg.num_nodes)
    words = (n + 63) // 64
    masks = np.zeros((n, words), dtype=np.uint64)
    ei = rg.edge_index.cpu().numpy()
    for u, v in zip(ei[0], ei[1]):
        if u == v:
            continue
        masks[u, v // 64] |= np.uint64(1) << np.uint64(v % 64)
        masks[v, u // 64] |= np.uint64(1) << np.uint64(u % 64)
    return masks


def popcount_rows(x):
    return np.unpackbits(x.view(np.uint8), axis=-1).sum(axis=-1).astype(np.int64)


# reverse-type map for the 4 relation-graph interaction types
# hh(0)<->hh, tt(1)<->tt, ht(2)<->th(3): the reverse of an ht edge is th
REV_T = {0: 0, 1: 1, 2: 3, 3: 2}


def triangle_stats(rg):
    """Per-edge triangle participation (undirected simple graph) totals."""
    masks = undirected_masks(rg)
    ei = rg.edge_index.cpu().numpy()
    et = rg.edge_type.cpu().numpy()
    tri_edge = popcount_rows(masks[ei[0]] & masks[ei[1]])  # per edge
    per_type = np.zeros(4, dtype=np.int64)
    for t in range(4):
        per_type[t] = int(tri_edge[et == t].sum())
    return {"participation_total": int(tri_edge.sum()),
            "per_type": per_type.tolist()}


def recip_count(rg):
    """#edges whose reverse edge (with the REVERSE type) also exists."""
    ei, et = rg.edge_index.cpu(), rg.edge_type.cpu()
    fwd = set(zip(ei[0].tolist(), ei[1].tolist(), et.tolist()))
    return sum((v, u, REV_T[t]) in fwd for (u, v, t) in fwd)


# ------------------------------------------------ rewiring ladder helpers ----

class RGEditor:
    """Double-edge-swap machinery on a relation graph with incremental
    reciprocal-count and triangle-census deltas (bitmask popcounts)."""

    def __init__(self, rg, seed=0, component_locked=False):
        self.n = int(rg.num_nodes)
        self.src = rg.edge_index[0].cpu().numpy().copy()
        self.dst = rg.edge_index[1].cpu().numpy().copy()
        self.et = rg.edge_type.cpu().numpy().copy()
        self.rng = np.random.default_rng(seed)
        self.words = (self.n + 63) // 64
        self.masks = undirected_masks(rg)
        self.edge_multi = {}
        for u, v in zip(self.src, self.dst):
            a, b = (int(u), int(v)) if u <= v else (int(v), int(u))
            self.edge_multi[(a, b)] = self.edge_multi.get((a, b), 0) + 1
        self.typed = set(zip(self.src.tolist(), self.dst.tolist(), self.et.tolist()))
        self.component_locked = component_locked
        if component_locked:
            self.comp = self._components()

    def _components(self):
        parent = list(range(self.n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v in zip(self.src, self.dst):
            ru, rv = find(int(u)), find(int(v))
            if ru != rv:
                parent[ru] = rv
        return np.array([find(i) for i in range(self.n)])

    def _mask_add(self, u, v):
        a, b = (u, v) if u <= v else (v, u)
        c = self.edge_multi.get((a, b), 0)
        self.edge_multi[(a, b)] = c + 1
        if c == 0 and u != v:
            self.masks[u, v // 64] |= np.uint64(1) << np.uint64(v % 64)
            self.masks[v, u // 64] |= np.uint64(1) << np.uint64(u % 64)

    def _mask_del(self, u, v):
        a, b = (u, v) if u <= v else (v, u)
        c = self.edge_multi.get((a, b), 0)
        if c <= 1:
            self.edge_multi.pop((a, b), None)
            if u != v:
                self.masks[u, v // 64] &= ~(np.uint64(1) << np.uint64(v % 64))
                self.masks[v, u // 64] &= ~(np.uint64(1) << np.uint64(u % 64))
        else:
            self.edge_multi[(a, b)] = c - 1

    def tri_through(self, u, v):
        return int(popcount_rows((self.masks[u] & self.masks[v])[None])[0])

    def total_triangles(self):
        """Exact triangle count of the CURRENT simple undirected graph."""
        edges = [e for e in self.edge_multi.keys() if e[0] != e[1]]
        if not edges:
            return 0
        us = np.array([e[0] for e in edges]); vs = np.array([e[1] for e in edges])
        return int(popcount_rows(self.masks[us] & self.masks[vs]).sum()) // 3

    def _local_recip(self, affected):
        """Exact reciprocal contribution of the affected edge set under the
        CURRENT typed-edge set (reverse edge carries the REVERSE type)."""
        return sum(1 for e in affected
                   if e in self.typed and (e[1], e[0], REV_T[e[2]]) in self.typed)

    def _mult(self, u, v):
        a, b = (u, v) if u <= v else (v, u)
        return self.edge_multi.get((a, b), 0)

    def propose_swap(self, i, j):
        """Swap dst of edges i,j (same type). Returns (d_recip, d_tri) or None,
        where d_tri is the EXACT change in total simple-graph triangle count
        (sequential removal/addition with multiplicity guards). The swap is
        APPLIED on return; caller keeps it or calls undo(i, j)."""
        if self.et[i] != self.et[j]:
            return None
        u1, v1, u2, v2 = int(self.src[i]), int(self.dst[i]), int(self.src[j]), int(self.dst[j])
        t = int(self.et[i])
        if v1 == v2 or u1 == v2 or u2 == v1:
            return None
        if self.component_locked and not (self.comp[u1] == self.comp[v2] == self.comp[u2] == self.comp[v1]):
            return None
        if (u1, v2, t) in self.typed or (u2, v1, t) in self.typed:
            return None  # avoid typed multi-edges
        r = REV_T[t]
        affected = {(u1, v1, t), (u2, v2, t), (u1, v2, t), (u2, v1, t),
                    (v1, u1, r), (v2, u2, r), (v2, u1, r), (v1, u2, r)}
        recip_before = self._local_recip(affected)
        # sequential exact triangle delta on the simple undirected graph
        d_tri = 0
        self.typed.discard((u1, v1, t))
        if u1 != v1 and self._mult(u1, v1) == 1:
            d_tri -= self.tri_through(u1, v1)
        self._mask_del(u1, v1)
        self.typed.discard((u2, v2, t))
        if u2 != v2 and self._mult(u2, v2) == 1:
            d_tri -= self.tri_through(u2, v2)
        self._mask_del(u2, v2)
        if u1 != v2 and self._mult(u1, v2) == 0:
            d_tri += self.tri_through(u1, v2)
        self._mask_add(u1, v2); self.typed.add((u1, v2, t))
        if u2 != v1 and self._mult(u2, v1) == 0:
            d_tri += self.tri_through(u2, v1)
        self._mask_add(u2, v1); self.typed.add((u2, v1, t))
        self.dst[i], self.dst[j] = v2, v1
        recip_after = self._local_recip(affected)
        return recip_after - recip_before, d_tri

    def undo(self, i, j):
        """Unconditional inverse swap (no validity checks — restoring a state
        that was valid by construction; propose_swap's guards could wrongly
        reject the inverse, e.g. with self-loops)."""
        u1, v1 = int(self.src[i]), int(self.dst[i])
        u2, v2 = int(self.src[j]), int(self.dst[j])
        t = int(self.et[i])
        self.typed.discard((u1, v1, t)); self._mask_del(u1, v1)
        self.typed.discard((u2, v2, t)); self._mask_del(u2, v2)
        self._mask_add(u1, v2); self.typed.add((u1, v2, t))
        self._mask_add(u2, v1); self.typed.add((u2, v1, t))
        self.dst[i], self.dst[j] = v2, v1

    # ---- reciprocity-preserving move set (v2 construction) ----
    # The destroy-then-anneal paradigm cannot restore the reciprocal-pair
    # count by random same-type dst-swaps (diagnostic 2026-07-02: triangles
    # converge, recip stuck at ~40%). v2 moves keep recip EXACT by
    # construction: (i) joint swap of two reciprocal PAIRS (both directions
    # swapped atomically), (ii) swap of two unpaired same-type edges,
    # rejected if it would create/destroy any reciprocal pair.

    def build_pair_index(self):
        """edge idx -> reverse edge idx for reciprocal pairs (REV type).
        Self-reciprocal edges (their reverse is themselves: self-loops with
        hh/tt types) are LOCKED — moving them would silently destroy their
        reciprocity, so they are excluded from both move pools."""
        loc = {}
        for k, (u, v, t) in enumerate(zip(self.src, self.dst, self.et)):
            loc[(int(u), int(v), int(t))] = k
        self.pair_of = np.full(len(self.src), -1, dtype=np.int64)
        self_recip = np.zeros(len(self.src), dtype=bool)
        for k, (u, v, t) in enumerate(zip(self.src, self.dst, self.et)):
            r = loc.get((int(v), int(u), REV_T[int(t)]))
            if r is None:
                continue
            if r == k:
                self_recip[k] = True
            else:
                self.pair_of[k] = r
        self.paired_idx = np.nonzero(self.pair_of >= 0)[0]
        self.unpaired_idx = np.nonzero((self.pair_of < 0) & ~self_recip)[0]
        self.n_locked_self_recip = int(self_recip.sum())

    def _would_pair(self, u, v, t):
        return (v, u, REV_T[t]) in self.typed

    def propose_swap_recip_preserving(self, i, j):
        """Move (ii): swap dst of two UNPAIRED same-type edges; reject any
        move that would create or break a reciprocal pair. Returns exact
        triangle delta or None. Applied on return (undo with undo())."""
        if self.et[i] != self.et[j] or self.pair_of[i] >= 0 or self.pair_of[j] >= 0:
            return None
        u1, v1, u2, v2 = int(self.src[i]), int(self.dst[i]), int(self.src[j]), int(self.dst[j])
        t = int(self.et[i])
        if v1 == v2 or u1 == v2 or u2 == v1:
            return None
        if self.component_locked and not (self.comp[u1] == self.comp[v2] == self.comp[u2] == self.comp[v1]):
            return None
        if (u1, v2, t) in self.typed or (u2, v1, t) in self.typed:
            return None
        if self._would_pair(u1, v2, t) or self._would_pair(u2, v1, t):
            return None  # would create a reciprocal pair
        if self._would_pair(u1, v1, t) or self._would_pair(u2, v2, t):
            return None  # defensive: removed edge is currently reciprocal
        d_tri = 0
        self.typed.discard((u1, v1, t))
        if u1 != v1 and self._mult(u1, v1) == 1:
            d_tri -= self.tri_through(u1, v1)
        self._mask_del(u1, v1)
        self.typed.discard((u2, v2, t))
        if u2 != v2 and self._mult(u2, v2) == 1:
            d_tri -= self.tri_through(u2, v2)
        self._mask_del(u2, v2)
        if u1 != v2 and self._mult(u1, v2) == 0:
            d_tri += self.tri_through(u1, v2)
        self._mask_add(u1, v2); self.typed.add((u1, v2, t))
        if u2 != v1 and self._mult(u2, v1) == 0:
            d_tri += self.tri_through(u2, v1)
        self._mask_add(u2, v1); self.typed.add((u2, v1, t))
        self.dst[i], self.dst[j] = v2, v1
        return d_tri

    def propose_pairswap(self, i, j):
        """Move (i): jointly swap two reciprocal PAIRS (i, pair_of[i]) and
        (j, pair_of[j]): (u1<->v1, u2<->v2) -> (u1<->v2, u2<->v1) in both
        directions. Preserves per-type degrees and recip count exactly.
        Returns exact triangle delta or None. Undo with undo_pairswap()."""
        pi, pj = int(self.pair_of[i]), int(self.pair_of[j])
        if pi < 0 or pj < 0 or j in (i, pi) or pj == i:
            return None
        if self.et[i] != self.et[j]:
            return None
        u1, v1, u2, v2 = int(self.src[i]), int(self.dst[i]), int(self.src[j]), int(self.dst[j])
        t = int(self.et[i])
        rt = REV_T[t]
        if len({u1, v1, u2, v2}) < 4:
            return None
        if self.component_locked and not (self.comp[u1] == self.comp[v2] == self.comp[u2] == self.comp[v1]):
            return None
        for (a, b, tt) in ((u1, v2, t), (u2, v1, t), (v2, u1, rt), (v1, u2, rt)):
            if (a, b, tt) in self.typed:
                return None
        d_tri = 0
        for (a, b, tt) in ((u1, v1, t), (v1, u1, rt), (u2, v2, t), (v2, u2, rt)):
            self.typed.discard((a, b, tt))
            if a != b and self._mult(a, b) == 1:
                d_tri -= self.tri_through(a, b)
            self._mask_del(a, b)
        for (a, b, tt) in ((u1, v2, t), (v2, u1, rt), (u2, v1, t), (v1, u2, rt)):
            if a != b and self._mult(a, b) == 0:
                d_tri += self.tri_through(a, b)
            self._mask_add(a, b)
            self.typed.add((a, b, tt))
        self.dst[i], self.dst[pi] = v2, u1  # i: u1->v2 ; pi (was v1->u1): v2->u1
        self.src[pi] = v2
        self.dst[j], self.dst[pj] = v1, u2  # j: u2->v1 ; pj (was v2->u2): v1->u2
        self.src[pj] = v1
        # pairing partners unchanged (i<->pi, j<->pj)
        return d_tri

    def undo_pairswap(self, i, j):
        self.propose_pairswap(i, j)  # involution under the same (i, j)

    def snapshot(self):
        return self.src.copy(), self.dst.copy()

    def restore(self, snap):
        """Rebuild internal state from a saved (src, dst) snapshot."""
        self.src, self.dst = snap[0].copy(), snap[1].copy()
        self.masks = np.zeros_like(self.masks)
        self.edge_multi = {}
        self.typed = set()
        for u, v, t in zip(self.src, self.dst, self.et):
            u, v, t = int(u), int(v), int(t)
            self.typed.add((u, v, t))
            self._mask_add(u, v)

    def to_rg(self, rg_template):
        rg = copy.copy(rg_template)
        rg.edge_index = torch.tensor(np.stack([self.src, self.dst]), dtype=torch.long,
                                     device=rg_template.edge_index.device)
        rg.edge_type = torch.tensor(self.et, dtype=torch.long,
                                    device=rg_template.edge_index.device)
        return rg


def ladder_variant(data, rung, seed, anneal_steps=60000, tol=0.02, log=None):
    """Build one rung of the matched-controls ladder (R1).

    rung: 'L1' degree-preserving rewire + reciprocal-pair count restored
          'L2' L1 + 2/3-motif census matched (2-paths exact by construction;
               triangles annealed to within tol)
          'L3' L2 + swaps restricted to connected components
          'L4' identity permutation within (type-wise degree-profile) classes
               (preserves ALL structural statistics; breaks binding only)
          'L4loose' arbitrary identity permutation (isomorphic relabel)
          'L5' fully random endpoints per type (destroys degrees too)
    """
    rg0 = data.relation_graph
    if rung == "L5":
        return PH.random_relgraph(data, seed), {}
    if rung in ("L4", "L4loose"):
        prof = rg_degree_profile(rg0)
        n = prof.shape[0]
        rng = np.random.default_rng(seed)
        perm = np.arange(n)
        if rung == "L4":
            classes = {}
            for i in range(n):
                classes.setdefault(tuple(prof[i].tolist()), []).append(i)
            n_nontrivial = 0
            for members in classes.values():
                if len(members) > 1:
                    n_nontrivial += len(members)
                    shuffled = rng.permutation(members)
                    for a, b in zip(members, shuffled):
                        perm[a] = b
            stats = {"permutable_nodes": int(n_nontrivial), "n_nodes": int(n)}
        else:
            perm = rng.permutation(n)
            stats = {"permutable_nodes": int(n), "n_nodes": int(n)}
        p = torch.tensor(perm, dtype=torch.long, device=rg0.edge_index.device)
        new_ei = torch.stack([p[rg0.edge_index[0]], p[rg0.edge_index[1]]])
        return PH.rg_clone_into(data, new_ei, rg0.edge_type), stats

    # L1-L3 (v2 construction): reciprocity-preserving move set — reciprocal
    # pairs swap atomically, unpaired edges swap among themselves with
    # recip-status guards. Recip count and per-type degrees are preserved
    # EXACTLY by construction (the destroy-then-restore paradigm cannot
    # restore recip by random moves — diagnostic 2026-07-02); L2/L3 anneal
    # the exact total-triangle count back (converges), typed census tracked
    # by periodic recompute + best snapshot.
    ed = RGEditor(rg0, seed=seed, component_locked=(rung == "L3"))
    ed.build_pair_index()
    target_recip = recip_count(rg0)
    target_tri = ed.total_triangles()
    target_typed = np.array(triangle_stats(rg0)["per_type"], dtype=float)
    cur_tri = target_tri
    rng = ed.rng
    n_pair, n_unpair = len(ed.paired_idx), len(ed.unpaired_idx)
    p_pair = n_pair / max(n_pair + n_unpair, 1)

    def random_move():
        """One recip-preserving move; returns (kind, i, j, d_tri) or None."""
        if rng.random() < p_pair and n_pair >= 4:
            i, j = ed.paired_idx[rng.integers(0, n_pair, 2)]
            r = ed.propose_pairswap(int(i), int(j))
            return ("P", int(i), int(j), r) if r is not None else None
        if n_unpair >= 2:
            i, j = ed.unpaired_idx[rng.integers(0, n_unpair, 2)]
            r = ed.propose_swap_recip_preserving(int(i), int(j))
            return ("U", int(i), int(j), r) if r is not None else None
        return None

    def undo_move(kind, i, j):
        if kind == "P":
            ed.undo_pairswap(i, j)
        else:
            ed.undo(i, j)

    m = len(ed.src)
    # 1) burn-in randomisation (recip invariant throughout)
    burn = 0
    for _ in range(min(m * 6, 900000)):
        mv = random_move()
        if mv is None:
            continue
        cur_tri += mv[3]; burn += 1

    # 2) L2/L3: anneal exact total triangles back to target
    def objective(tc):
        return abs(tc - target_tri) / max(target_tri, 1)

    def typed_residual():
        cur = np.array(triangle_stats(ed.to_rg(rg0))["per_type"], dtype=float)
        return float(np.abs(cur - target_typed).sum() / max(target_typed.sum(), 1))

    obj = objective(cur_tri)
    best_snap, best_typed = None, float("inf")
    accepted = 0
    if rung in ("L2", "L3"):
        T0, T1 = 0.5, 1e-5
        for step in range(anneal_steps):
            T = T0 * (T1 / T0) ** (step / max(anneal_steps - 1, 1))
            mv = random_move()
            if mv is None:
                continue
            kind, i, j, d = mv
            new_obj = objective(cur_tri + d)
            if new_obj <= obj or rng.random() < math.exp(-(new_obj - obj) / T):
                cur_tri += d; obj = new_obj
                accepted += 1
                if obj < 5 * tol and accepted % 2000 == 0:
                    tr = typed_residual()
                    if tr < best_typed:
                        best_typed, best_snap = tr, ed.snapshot()
                    if tr < tol / 2:
                        break
            else:
                undo_move(kind, i, j)
        tr = typed_residual()
        if tr < best_typed:
            best_typed, best_snap = tr, ed.snapshot()
        if best_snap is not None:
            ed.restore(best_snap)

        # stage B: batch hill-climb the TYPED census on the total-exact
        # manifold — batches of zero-total-delta moves, kept only if the
        # (cheap, vectorised) full typed recompute improves; ~45 s cap
        t_hc = time.time()
        cur_typed = typed_residual()
        hc_kept = hc_tried = 0
        while time.time() - t_hc < 45 and cur_typed > tol / 2:
            snap = ed.snapshot()
            moved = 0
            trials = 0
            while moved < 150 and trials < 3000:
                trials += 1
                mv = random_move()
                if mv is None:
                    continue
                if mv[3] != 0:
                    undo_move(mv[0], mv[1], mv[2])
                    continue
                moved += 1
            if moved == 0:
                break
            new_typed = typed_residual()
            hc_tried += 1
            if new_typed < cur_typed:
                cur_typed = new_typed
                hc_kept += 1
            else:
                ed.restore(snap)

    # exact final verification (recompute from scratch — tracked deltas are
    # only annealing guidance)
    final_rg = ed.to_rg(rg0)
    final_recip = recip_count(final_rg)
    final_tri = ed.total_triangles()
    final_typed = np.array(triangle_stats(final_rg)["per_type"], dtype=float)
    typed_res = float(np.abs(final_typed - target_typed).sum()
                      / max(target_typed.sum(), 1))
    recip_dev = abs(final_recip - target_recip) / max(target_recip, 1)
    final_obj = recip_dev + (objective(final_tri) if rung in ("L2", "L3") else 0.0)
    stats = {"construction": "v2-recip-preserving-moves",
             "burnin_swaps": burn, "anneal_accepted": accepted,
             "n_paired_edges": int(n_pair), "n_unpaired_edges": int(n_unpair),
             "n_locked_self_recip": int(getattr(ed, "n_locked_self_recip", 0)),
             "recip_target": int(target_recip), "recip_final": int(final_recip),
             "tri_target": int(target_tri), "tri_final": int(final_tri),
             "typed_tri_residual": typed_res,
             "objective_residual": float(final_obj)}
    if log:
        print(f"[{rung} s{seed}] {stats}", flush=True)
    if final_obj > tol:
        stats["warn"] = f"residual {final_obj:.4f} > tol {tol}"
    if rung in ("L2", "L3") and typed_res > 3 * tol:
        stats["warn_typed"] = f"typed triangle residual {typed_res:.4f}"
    d = copy.copy(data)
    d.relation_graph = final_rg
    return d, stats
