import runpy
import sys

import torch

with torch.no_grad():
    sample = torch.ones((256, 256), device="cuda")
    product = sample @ sample
    torch.cuda.synchronize()
    del sample, product

print("[runtime] CUDA BLAS warmup completed before loading training data", flush=True)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
