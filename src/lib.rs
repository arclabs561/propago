//! propago: The Learning Layer.
//! 
//! Provides Graph Neural Network layers (GCN, GAT, SAGE) and training loops
//! built on `candle` tensors and `nexus` graph structures.

pub mod nn;
pub mod hyperbolic;

pub use nn::{GCNConv, GATConv};
pub use hyperbolic::HGCNConv;
