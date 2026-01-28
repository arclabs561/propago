# propago

Graph neural network primitives and training loops.

This crate is a small, reusable “learning layer” for graph ML:
- GNN layers (GCN/GAT/SAGE-like, and a hyperbolic variant)
- built on `candle` tensors

## Status / scope

- This is not a full end-to-end training framework; it’s “layers + small loops”.
- Keep interfaces small so higher-level graph stacks can integrate without tight coupling.
- The hyperbolic layer `HGCNConv` now uses a Tensor-native Poincaré ball implementation
  (`CandlePoincareBall`: `log0`/`exp0` + projection) so it can run on Candle backends (CPU/GPU).

## Backends

- **Candle (default)**: `--features backend-candle` (enabled by default).
- **Burn (opt-in)**: `--features backend-burn` exposes Burn-tensor Poincaré ops (spec-tested vs `hyp`).
- **MLX (opt-in)**: `--features backend-mlx` builds `mlx-rs` and requires `cmake` + Xcode MetalToolchain; tests force CPU for determinism.

