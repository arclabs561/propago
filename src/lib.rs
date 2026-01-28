//! propago: The Learning Layer.
//!
//! Provides Graph Neural Network layers (GCN, GAT, SAGE) and training loops
//! built on `candle` tensors.

pub mod hyperbolic;
pub mod nn;

pub use hyperbolic::HGCNConv;
pub use nn::{GATConv, GCNConv};

#[cfg(feature = "backend-burn")]
pub mod burn_hyperbolic;

#[cfg(feature = "backend-mlx")]
pub mod mlx_hyperbolic;
