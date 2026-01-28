//! Minimal HGCN smoke example (Candle backend).
//!
//! Run:
//!   cargo run -p propago --example hgcn_smoke

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use propago::HGCNConv;

fn main() -> Result<()> {
    let dev = &Device::Cpu;
    let dtype = DType::F32;

    let n = 6usize;
    let d = 4usize;

    let x = Tensor::randn(0f32, 0.1f32, (n, d), dev)?.to_dtype(dtype)?;
    let adj = Tensor::eye(n, dtype, dev)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, dev);
    let lin = candle_nn::linear(d, d, vb)?;
    let layer = HGCNConv::new(lin, 1.0);

    let y = layer.forward(&x, &adj)?;
    let (yn, yd) = y.dims2()?;
    println!("y shape: [{yn}, {yd}]");

    let p = Tensor::zeros((1, d), dtype, dev)?;
    let y2 = layer.forward_with_basepoint(&x, &adj, &p)?;
    let (yn2, yd2) = y2.dims2()?;
    println!("y(basepoint) shape: [{yn2}, {yd2}]");

    Ok(())
}
