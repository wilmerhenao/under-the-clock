"""Peak memory of train_lora.py's chunked, checkpointed loss vs trained positions and chunk size (CPU, real vocab).

usage: loss_mem.py <n_pos> <chunk> ckpt|eager   (one configuration per process: ru_maxrss is a process-lifetime peak)
B25 (2026-10-07): ckpt 1024/256 -> +847 MiB, 3072/256 -> +853, 3072/1024 -> +3,927; eager 3072/256 -> +789.
"""
import resource, sys, json, time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
torch.manual_seed(0)
V, H, SOFTCAP = 262144, 512, 30.0
n_pos, c, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
W = torch.randn(V, H, dtype=torch.bfloat16) * 0.02  # tied embedding, frozen (bf16 directly: no fp32 temporary in the baseline)
hs = torch.randn(n_pos, H, dtype=torch.bfloat16).requires_grad_(True)
ts = torch.randint(0, V, (n_pos,))
mib = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # noqa: E731  (peak RSS so far, MiB)


def chunk_loss(h, t):  # as in train_lora.py
    logits = (h @ W.T).float()
    logits = SOFTCAP * torch.tanh(logits / SOFTCAP)
    return F.cross_entropy(logits, t, reduction='sum')


base = mib()
t0 = time.time()
if mode == 'ckpt':      # train_lora.py: sum of checkpointed chunks, then one backward
    total = sum(checkpoint(chunk_loss, hs[i:i + c], ts[i:i + c], use_reentrant=False) for i in range(0, n_pos, c))
    fwd = mib()
    (total / n_pos).backward()
elif mode == 'eager':   # backward chunk by chunk inside the forward (grad of h accumulated), nothing kept
    fwd = None
    for i in range(0, n_pos, c):
        (chunk_loss(hs[i:i + c], ts[i:i + c]) / n_pos).backward()
peak = mib()
print(json.dumps({'n_pos': n_pos, 'chunk': c, 'mode': mode, 'base_mib': round(base), 'fwd_inc_mib': fwd and round(fwd - base),
                  'peak_inc_mib': round(peak - base), 'per_pos_mib': round((peak - base) / n_pos, 3),
                  'fp32_logit_mib_per_pos': V * 4 / 2**20, 'grad_ok': hs.grad is not None and bool(torch.isfinite(hs.grad.float()).all()),
                  'seconds': round(time.time() - t0, 1)}))
