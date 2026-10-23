"""Check the runtime environment before experiments."""

from __future__ import annotations

import sys


def main() -> None:
    print(f"Python {sys.version.split()[0]}")
    try:
        import torch

        print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    except ImportError as exc:
        raise SystemExit("PyTorch is not installed; run `uv sync`.") from exc


if __name__ == "__main__":
    main()
