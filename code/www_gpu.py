"""GPU admission for the WWW 2027 runners on a shared GPU server.

Sharing rule for these runs: a GPU is usable when memory.used <= 40960 MiB and
utilization.gpu <= 25 %, at most one of our processes runs on a GPU, and the
condition is checked before each launch and before each unit of work.

admit() runs at process start, before any torch.cuda call: it resolves the one
GPU named by CUDA_VISIBLE_DEVICES, waits until the GPU is usable, then takes an
exclusive lock keyed by the GPU uuid and holds it for the process lifetime.
between_units() repeats the usability check before each unit, not counting the
memory of this process's own context.
"""
import fcntl
import os
import statistics
import subprocess
import sys
import time

MAX_MEMORY_MIB = 40960
MAX_UTILIZATION = 25
POLL_SEC = 60
REPORT_EVERY_SEC = 600
PROJECT = os.path.basename(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_gpu = None
_lock = None


def _query(fields, kind="gpu"):
    out = subprocess.run(["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.strip()]


def resolve_visible_gpu():
    """{"index", "uuid"} of the single physical GPU named by CUDA_VISIBLE_DEVICES
    (an index or a GPU uuid)."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = [x.strip() for x in (raw or "").split(",") if x.strip()]
    if len(ids) != 1:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must name exactly one GPU, got {raw!r}")
    for index, uuid in _query("index,uuid"):
        if ids[0] in (index, uuid):
            return {"index": int(index), "uuid": uuid}
    raise SystemExit(f"CUDA_VISIBLE_DEVICES={ids[0]!r} matches no GPU listed by nvidia-smi")


def _sample(uuid):
    """(memory.used minus this process's own context, utilization.gpu)."""
    own = sum(int(mem) for u, pid, mem in _query("gpu_uuid,pid,used_memory", "compute-apps")
              if u == uuid and pid == str(os.getpid()))
    for u, used, util in _query("uuid,memory.used,utilization.gpu"):
        if u == uuid:
            return int(used) - own, int(util)
    raise SystemExit(f"GPU {uuid} is no longer listed by nvidia-smi")


def wait_until_usable(tag):
    """Block until the median of three samples meets the sharing rule."""
    waited = 0
    while True:
        samples = []
        for _ in range(3):
            samples.append(_sample(_gpu["uuid"]))
            time.sleep(1)
        memory = statistics.median(s[0] for s in samples)
        utilization = statistics.median(s[1] for s in samples)
        if memory <= MAX_MEMORY_MIB and utilization <= MAX_UTILIZATION:
            return
        if waited % REPORT_EVERY_SEC == 0:
            print(f"[www_gpu] {tag}: GPU {_gpu['index']} not usable yet (other processes "
                  f"{memory:.0f} MiB, utilization {utilization:.0f}%); waiting", flush=True)
        time.sleep(POLL_SEC)
        waited += POLL_SEC


def admit():
    """Resolve, wait until usable, then lock the GPU for this process's lifetime."""
    global _gpu, _lock
    if _lock is not None:
        return _gpu
    _gpu = resolve_visible_gpu()
    wait_until_usable("admission")
    path = f"/tmp/{PROJECT}-www-gpu-{_gpu['uuid']}.lock"
    fh = open(path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        holder = fh.read().strip() or "unknown"
        fh.close()
        raise SystemExit(f"GPU {_gpu['index']} ({_gpu['uuid']}) already runs one of our "
                         f"processes (lock {path}, holder: {holder})")
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid {os.getpid()}: {' '.join(sys.argv)}\n")
    fh.flush()
    _lock = fh
    return _gpu


def between_units(tag):
    if _lock is not None:
        wait_until_usable(tag)


def current():
    return None if _gpu is None else dict(_gpu)
