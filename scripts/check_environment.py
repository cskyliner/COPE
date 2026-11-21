"""Check the COPE runtime without downloading weights or loading datasets."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.chdir(root)
    for module in (
        "numpy", "PIL", "torch", "torchvision", "torchaudio", "yacs", "timm",
        "skimage", "tqdm", "ftfy", "regex", "cv2", "einops", "matplotlib",
    ):
        importlib.import_module(module)

    import torch
    import torchvision
    import clip
    from cope.loss.triplet_loss import TripletLoss

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and PyTorch build")

    torch.manual_seed(0)
    conv = torch.nn.Conv2d(3, 8, 3).cuda()
    conv(torch.randn(2, 3, 16, 16, device="cuda")).square().mean().backward()
    if not torch.isfinite(conv.weight.grad).all():
        raise RuntimeError("CUDA convolution produced non-finite gradients")

    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
    scores = torch.tensor([0.9, 0.8], device="cuda")
    if torchvision.ops.nms(boxes, scores, 0.5).tolist() != [0]:
        raise RuntimeError("torchvision CUDA NMS check failed")

    features = torch.randn(8, 32, device="cuda", requires_grad=True)
    labels = torch.arange(4, device="cuda").repeat_interleave(2)
    loss = TripletLoss(margin=0.3)(features, labels)[0].mean()
    loss.backward()
    if not torch.isfinite(features.grad).all():
        raise RuntimeError("Triplet loss produced non-finite gradients")
    if tuple(clip.tokenize(["a pedestrian"]).shape) != (1, 77):
        raise RuntimeError("CLIP tokenizer check failed")
    torch.cuda.synchronize()

    for entry in ("train_cope.py", "test_cope.py"):
        subprocess.run([sys.executable, entry, "--help"], check=True, capture_output=True)

    print(json.dumps({
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "checks": ["core_imports", "cuda_convolution_backward", "torchvision_cuda_nms",
                   "triplet_loss_backward", "clip_tokenizer", "train_help", "evaluate_help"],
        "status": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
