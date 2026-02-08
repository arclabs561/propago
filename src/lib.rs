//! propago: The Learning Layer.
//!
//! Provides a small set of graph learning primitives built on `candle` tensors.
//!
//! Current public surface:
//! - `GCNConv`: a simple graph convolution (linear + adjacency matmul)
//! - `HGCNConv`: hyperbolic graph convolution on the Poincaré ball (Tensor-native ops)

#![forbid(unsafe_code)]

pub mod hyperbolic;
pub mod nn;

pub use hyperbolic::HGCNConv;
pub use nn::GCNConv;

#[cfg(feature = "backend-burn")]
pub mod burn_hyperbolic;

#[cfg(feature = "backend-mlx")]
pub mod mlx_hyperbolic;
