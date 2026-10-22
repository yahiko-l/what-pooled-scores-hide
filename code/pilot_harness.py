"""ULTRA harness: the official checkpoints, the GraIL inductive splits, relation-graph
utilities and ULTRA's filtered-ranking evaluation, shared by the ULTRA runners."""
import os, sys, copy
import torch

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
ULTRA_DIR = os.path.join(PILOT_DIR, "ULTRA")
sys.path.insert(0, ULTRA_DIR)

from easydict import EasyDict
from torch_geometric.data import Data
from ultra import datasets as uds
from ultra import tasks, util  # noqa
from ultra.models import Ultra
from script.run import test as ultra_test

DATA_ROOT = os.path.join(PILOT_DIR, "kg-datasets")
CKPT = os.path.join(ULTRA_DIR, "ckpts", "ultra_3g.pth")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_CFG = dict(
    relation_model=dict(cls="RelNBFNet", input_dim=64, hidden_dims=[64] * 6,
                        message_func="distmult", aggregate_func="sum",
                        short_cut=True, layer_norm=True),
    entity_model=dict(cls="EntityNBFNet", input_dim=64, hidden_dims=[64] * 6,
                      message_func="distmult", aggregate_func="sum",
                      short_cut=True, layer_norm=True),
)


def make_cfg(bs=64):
    return EasyDict({
        "train": {"batch_size": bs},
        "task": {"name": "InductiveInference", "num_negative": 256,
                 "strict_negative": True, "adversarial_temperature": 1,
                 "metric": ["mr", "mrr", "hits@1", "hits@3", "hits@10"]},
    })


def load_model(ckpt_path=None):
    """Load a frozen official ULTRA ckpt (default ultra_3g; amendment-1 §D
    cross-ckpt replication passes ultra_4g / ultra_50g)."""
    rel = dict(MODEL_CFG["relation_model"]); rel["class"] = rel.pop("cls")
    ent = dict(MODEL_CFG["entity_model"]); ent["class"] = ent.pop("cls")
    model = Ultra(rel_model_cfg=EasyDict(rel), entity_model_cfg=EasyDict(ent))
    state = torch.load(ckpt_path or CKPT, map_location="cpu")
    model.load_state_dict(state["model"])
    return model.to(DEVICE).eval()


def load_inductive(name, version):
    cls = {"fb": uds.FB15k237Inductive, "wn": uds.WN18RRInductive}[name]
    ds = cls(root=DATA_ROOT, version=version)
    train_data, valid_data, test_data = ds[0], ds[1], ds[2]
    # Grail-style filtering (run.py else-branch): inference edges + test edges
    fe = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
    ft = torch.cat([test_data.edge_type, test_data.target_edge_type])
    filtered = Data(edge_index=fe, edge_type=ft, num_nodes=test_data.num_nodes)
    return test_data.to(DEVICE), filtered.to(DEVICE)


def subsample_targets(data, n, seed=0):
    d = copy.copy(data)
    m = data.target_edge_index.shape[1]
    if m <= n:
        return data
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(m, generator=g)[:n].to(data.target_edge_index.device)
    d.target_edge_index = data.target_edge_index[:, idx]
    d.target_edge_type = data.target_edge_type[idx]
    return d


class _Quiet:
    def warning(self, *a, **k): pass


def eval_mrr(model, data, filtered, bs=64):
    cfg = make_cfg(bs)
    with torch.no_grad():
        metrics = ultra_test(cfg, model, data, device=DEVICE, logger=_Quiet(),
                             filtered_data=filtered, return_metrics=True)
    return {k: float(v) for k, v in metrics.items()}


# ---------------- relation-graph interventions (Pilot B) ----------------

def rg_clone_into(data, new_ei, new_et):
    d = copy.copy(data)
    rg = copy.copy(data.relation_graph)
    rg.edge_index = new_ei
    rg.edge_type = new_et
    d.relation_graph = rg
    return d


def random_relgraph(data, seed):
    """Same #edges per type, uniform random endpoints (destroys degrees too)."""
    rg = data.relation_graph
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = int(rg.num_nodes)
    ei = rg.edge_index.clone()
    et = rg.edge_type
    for t in range(int(rg.num_relations)):
        m = (et == t).nonzero(as_tuple=True)[0]
        ei[0][m] = torch.randint(n, (m.numel(),), generator=g).to(ei.device)
        ei[1][m] = torch.randint(n, (m.numel(),), generator=g).to(ei.device)
    return rg_clone_into(data, ei, et)


# ---------------- Pilot A: micro-unit stability audit ----------------

def rel_degrees(rg):
    n = int(rg.num_nodes)
    deg = torch.zeros(n, dtype=torch.long, device=rg.edge_index.device)
    deg.scatter_add_(0, rg.edge_index[0], torch.ones_like(rg.edge_index[0]))
    deg.scatter_add_(0, rg.edge_index[1], torch.ones_like(rg.edge_index[1]))
    return deg
