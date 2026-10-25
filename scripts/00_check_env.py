"""Check the runtime environment before experiments.

Run with `uv run python scripts/00_check_env.py` (on Colab: `python scripts/00_check_env.py`).
Prints library versions and the GPU, then trains one step of a 2-layer R-GCN with neighbour
sampling on a random graph, in fp32 and under bf16 autocast. Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
import traceback

NUM_NODES = 10_000
NUM_EDGES = 50_000
NUM_RELATIONS = 5
HIDDEN = 64
NUM_NEIGHBORS = [15, 10]
BATCH_SIZE = 512


def print_versions() -> None:
    import pykeen
    import torch
    import torch_geometric

    print(f"Python {sys.version.split()[0]}")
    print(f"torch {torch.__version__} | CUDA build {torch.version.cuda}")
    print(f"PyG {torch_geometric.__version__} | PyKEEN {pykeen.get_version()}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available: check `nvidia-smi` and the NVIDIA driver.")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM {props.total_memory / 1024**3:.1f} GiB")


def make_model_and_loader():
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import RGCNConv

    class RGCN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = RGCNConv(HIDDEN, HIDDEN, NUM_RELATIONS, num_bases=2)
            self.conv2 = RGCNConv(HIDDEN, HIDDEN, NUM_RELATIONS, num_bases=2)

        def forward(self, x, edge_index, edge_type):
            x = self.conv1(x, edge_index, edge_type).relu()
            return self.conv2(x, edge_index, edge_type)

    generator = torch.Generator().manual_seed(0)
    data = Data(
        x=torch.randn(NUM_NODES, HIDDEN, generator=generator),
        edge_index=torch.randint(0, NUM_NODES, (2, NUM_EDGES), generator=generator),
        edge_type=torch.randint(0, NUM_RELATIONS, (NUM_EDGES,), generator=generator),
    )
    loader = NeighborLoader(data, num_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE)
    print(
        f"Random graph: {NUM_NODES} nodes, {NUM_EDGES} edges, {NUM_RELATIONS} relations; "
        f"NeighborLoader({NUM_NEIGHBORS})"
    )
    return RGCN().cuda(), loader


def train_step(model, loader, bf16: bool) -> None:
    import torch

    batch = next(iter(loader)).to("cuda")
    model.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        out = model(batch.x, batch.edge_index, batch.edge_type)
        loss = out[: batch.batch_size].pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("missing or non-finite gradients")
    label = "bf16 autocast" if bf16 else "fp32"
    print(
        f"R-GCN forward+backward ({label}) ok: {batch.num_nodes} sampled nodes, "
        f"out dtype {out.dtype}, loss {loss.item():.4f}"
    )


def main() -> None:
    failures = []

    def check(name, fn, *args):
        try:
            return fn(*args)
        except Exception:
            print(f"FAILED: {name}")
            traceback.print_exc()
            failures.append(name)
            return None

    check("versions and GPU", print_versions)
    if not failures:
        built = check("random graph and model", make_model_and_loader)
        if built is not None:
            import torch

            print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
            check("fp32 train step", train_step, *built, False)
            check("bf16 train step", train_step, *built, True)
            print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")

    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
