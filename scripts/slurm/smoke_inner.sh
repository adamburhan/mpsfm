#!/bin/bash
# Runs INSIDE the container (via smoke.sh). No quoting gymnastics needed here.
set -e

echo "=== gpu ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "=== sfm stack ==="
python - <<'EOF'
import pyceres, pycolmap
print("maxmix binding:", hasattr(pycolmap, "create_maxmix_depth_bundle_adjuster"))
import torch
a = torch.randn(2048, 2048, device="cuda")
print("torch:", torch.__version__, torch.version.cuda, "matmul finite:", (a @ a).isfinite().all().item())
import mpsfm
print("mpsfm import: ok")
EOF

echo "=== gs stack (kernels actually executing on this arch) ==="
/opt/gs/bin/python - <<'EOF'
import torch
print("gs venv:", torch.__version__, "capability:", torch.cuda.get_device_capability())
from simple_knn._C import distCUDA2
d = distCUDA2(torch.rand(10000, 3, device="cuda"))
print("simple_knn kernel: finite =", torch.isfinite(d).all().item(), ", positive =", (d.max() > 0).item())
from diff_gaussian_rasterization import GaussianRasterizer
import fused_ssim
print("rasterizer/fused_ssim import: ok")
EOF

echo "=== symlink chain (repo -> scratch -> datasets) ==="
ls /mpsfm/local/weights/
ls /mpsfm/local/benchmarks/scannetpp/data/0d2ee665be/images/ | head -3

echo "=== ALL SMOKE CHECKS PASSED ==="
