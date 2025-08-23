import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to the checkpoint .pt/.pth file")
parser.add_argument("--show-tensors", action="store_true", help="Show tensor shapes under each key")
parser.add_argument("--full", action="store_true", help="Show full keys for nested dicts")
args = parser.parse_args()

path = args.path

# For PyTorch 2.6+ weights_only guard
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except Exception as e:
    print(f"[!] Standard load failed: {e}")
    print("[*] Retrying with safe_globals for NumPy scalars...")
    with torch.serialization.safe_globals([np._core.multiarray.scalar]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

print(f"\n=== Top-level type: {type(checkpoint)} ===")

# Case 1: Saved with torch.save(dict)
if isinstance(checkpoint, dict):
    print("\nTop-level keys:")
    for k in checkpoint.keys():
        v = checkpoint[k]
        if args.show_tensors:
            if isinstance(v, dict):
                tensor_keys = [x for x, y in v.items() if torch.is_tensor(y)]
                print(f"  - {k} (dict, {len(v)} items, {len(tensor_keys)} tensors)")
            elif torch.is_tensor(v):
                print(f"  - {k} (tensor {tuple(v.shape)})")
            else:
                print(f"  - {k} ({type(v)})")
        else:
            print(f"  - {k}")

    # Optional: show shapes of nested dicts (state_dicts)
    if args.full:
        print("\nNested dict shapes:")
        for k, v in checkpoint.items():
            if isinstance(v, dict):
                print(f"\n[{k}]")
                for subk, subv in v.items():
                    if torch.is_tensor(subv):
                        print(f"  {subk:40s} {tuple(subv.shape)}")
                    else:
                        print(f"  {subk:40s} ({type(subv)})")

# Case 2: Saved model object (rare)
else:
    print("\nCheckpoint is not a dict. Type:", type(checkpoint))
    try:
        sd = checkpoint.state_dict()
        print("It has a state_dict with keys:")
        for k in sd.keys():
            print("  -", k)
    except Exception as e:
        print("Cannot inspect state_dict:", e)