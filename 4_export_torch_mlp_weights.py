
import argparse
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="models/torch_mlp.pth")
    ap.add_argument("--out", required=True, help="Output NPZ with weights")
    args = ap.parse_args()

    state = torch.load(args.pth, map_location="cpu")
    out = {}
    for k, v in state.items():
        out[k] = v.detach().cpu().numpy()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, **out)
    print(f"[OK] Exported weights to {outp}")


if __name__ == "__main__":
    main()
