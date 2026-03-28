# Sparse k-WTA — Non-Record Research Submission

## Track
Non-record, unlimited compute. Training exceeds 10 minutes.

## Hypothesis
k-Winners-Take-All (k-WTA) on MLP hidden activations causes structural
specialization aligned with the U-Net encoder/decoder split without an
explicit router. A larger sparse model (14L) should converge to better
BPB than a smaller dense model (11L) at the same token budget, with the
size difference funded by structural sparsity.

## Changes from baseline
**Optimizations from leaderboard submissions (applied to both runs):**
- Sliding window eval stride=64 (~0.032 BPB free)
- FP16 tied embedding export (~0.007 BPB)
- LR: matrix=0.02, scalar=0.02, tied_embed=0.03
- Muon momentum 0.99 (warmup from 0.92 over 1500 steps)
- Muon weight decay 0.04
- int6 QAT (STE fake-quantization during training)
- zstd-22 compression
- EMA weight averaging (decay=0.997)
- Warmdown 3000 iterations

**Sparse-specific (Run B only):**
- k-WTA after relu^2 — encoder k=0.15, bottleneck k=0.05, decoder k=0.08
- Utilization EMA tracked per layer during training
- Weight pruning: linear 0->50% sparsity from 10% to 70% of training
- Per-layer utilization maps saved every 2000 steps for analysis

## Runs
| Run | Layers | Sparse | val_bpb | Steps | Time |
|-----|--------|--------|---------|-------|------|
| Dense baseline | 11L 3x | No | | 80k | |
| Sparse 14L | 14L 3x | Yes 50% | | 80k | |

## Specialization analysis
[Fill after run: enc_util vs dec_util ratio progression,
bottleneck behavior, dead neuron distribution]

## Result interpretation
If enc/dec ratio > 1.3 and growing -> structural specialization confirmed.
If sparse 14L BPB < dense 11L at same token count -> sparse scaling works.
