//! Numerical integration tests for the public `GCNConv` forward paths.

use burn::module::{Module, Param, ParamId};
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use ricci::GCNConv;

type B = NdArray<f32>;

fn dev() -> <B as Backend>::Device {
    <B as Backend>::Device::default()
}

fn fixed_layer() -> GCNConv<B> {
    let weight = Tensor::from_data(
        TensorData::new(vec![2.0f32, -1.0, 0.5, 3.0], [2, 2]),
        &dev(),
    );
    let bias = Tensor::from_data(TensorData::new(vec![1.0f32, -2.0], [2]), &dev());
    GCNConv::new(Linear {
        weight: Param::initialized(ParamId::new(), weight),
        bias: Some(Param::initialized(ParamId::new(), bias)),
    })
}

fn inputs() -> (Tensor<B, 2>, Tensor<B, 2>) {
    let x = Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, -1.0, 4.0, 3.0, 0.5], [3, 2]),
        &dev(),
    );
    // Deliberately not row-stochastic: row sums are 3, 1.5, and 0.
    let adj = Tensor::from_data(
        TensorData::new(vec![2.0f32, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], [3, 3]),
        &dev(),
    );
    (x, adj)
}

fn assert_close(got: Tensor<B, 2>, expected: &[f32]) {
    let got = got.into_data().to_vec::<f32>().unwrap();
    assert_eq!(got.len(), expected.len());
    for (i, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() < 1e-6,
            "element {i}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn gcn_forward_matches_affine_gcn_equation() {
    let layer = fixed_layer();
    let (x, adj) = inputs();

    // XW = [[3,5], [0,13], [6.25,-1.5]], then adj @ XW + b.
    assert_close(layer.forward(x, adj), &[7.0, 21.0, 7.25, 3.0, 1.0, -2.0]);
}

#[test]
fn gcn_legacy_differs_by_adjacency_weighted_bias() {
    let layer = fixed_layer();
    let (x, adj) = inputs();

    assert_close(
        layer.forward_legacy(x.clone(), adj.clone()),
        &[9.0, 17.0, 7.75, 2.0, 0.0, 0.0],
    );

    let corrected = layer.forward(x, adj);
    // legacy - corrected = (row_sum(adj) - 1) * b
    assert_close(
        layer.forward_legacy(inputs().0, inputs().1) - corrected,
        &[2.0, -4.0, 0.5, -1.0, -1.0, 2.0],
    );
}

#[test]
fn gcn_record_round_trip_preserves_both_forward_paths() {
    let layer = fixed_layer();
    let record = layer.clone().into_record();
    let restored = fixed_layer().load_record(record);
    let (x, adj) = inputs();

    assert_close(
        restored.forward(x.clone(), adj.clone()),
        &[7.0, 21.0, 7.25, 3.0, 1.0, -2.0],
    );
    assert_close(
        restored.forward_legacy(x, adj),
        &[9.0, 17.0, 7.75, 2.0, 0.0, 0.0],
    );
}
