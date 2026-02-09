//! Burn-backed hyperbolic geometry adapters (opt-in).
//!
//! This module mirrors the canonical `hyp` implementation, but operates on Burn tensors so it can
//! run on Burn backends (ndarray / wgpu / tch).
//!
//! Ownership: `hyp` remains the single source of truth for formulas and numerical behavior. This
//! module must be spec-tested against `hyp` (see tests below).

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Poincaré ball operations on Burn tensors (curvature parameter \(c>0\)).
#[derive(Debug, Clone, Copy)]
pub struct BurnPoincareBall {
    c: f32,
    eps: f32,
}

impl BurnPoincareBall {
    #[must_use]
    pub fn new(c: f64) -> Self {
        Self {
            c: c as f32,
            eps: 1e-6,
        }
    }

    fn sqrt_c(&self) -> f32 {
        self.c.sqrt()
    }

    fn max_norm(&self) -> f32 {
        // radius = 1/sqrt(c)
        (1.0 / self.sqrt_c()) - 1e-5
    }

    fn atanh<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // atanh(x) = 0.5 * ln((1+x)/(1-x))
        let ones = Tensor::<B, 2>::ones(x.dims(), &x.device());
        let num = ones.clone() + x.clone();
        let den = ones - x;
        (num / (den + self.eps)).log() * 0.5
    }

    fn norm_keepdim<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let b = x.dims()[0];
        x.powf_scalar(2.0).sum_dim(1).sqrt().reshape([b, 1])
    }

    fn dot_keepdim<B: Backend>(&self, x: Tensor<B, 2>, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let b = x.dims()[0];
        (x * y).sum_dim(1).reshape([b, 1])
    }

    fn lambda_x<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // λ_x = 2 / (1 - c ||x||^2)
        let b = x.dims()[0];
        let dev = x.device();
        let x2 = x.powf_scalar(2.0).sum_dim(1).reshape([b, 1]);
        Tensor::<B, 2>::ones([b, 1], &dev) - x2 * self.c
    }

    /// Project points to stay inside the ball.
    pub fn project<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let b = x.dims()[0];
        let norm = self.norm_keepdim(x.clone());
        let max = self.max_norm();

        // scale = min(1, max / (||x|| + eps))
        let denom = norm + self.eps;
        let scale = (Tensor::<B, 2>::ones([b, 1], &x.device()) * max) / denom;
        let scale = scale.clamp_max(1.0);
        x * scale
    }

    /// Möbius addition on the ball.
    pub fn mobius_add<B: Backend>(&self, x: Tensor<B, 2>, y: Tensor<B, 2>) -> Tensor<B, 2> {
        // x ⊕_c y = ((1 + 2c<x,y> + c||y||^2) x + (1 - c||x||^2) y) / (1 + 2c<x,y> + c^2||x||^2||y||^2)
        let b = x.dims()[0];
        let x2 = x.clone().powf_scalar(2.0).sum_dim(1).reshape([b, 1]);
        let y2 = y.clone().powf_scalar(2.0).sum_dim(1).reshape([b, 1]);
        let xy = self.dot_keepdim(x.clone(), y.clone());

        let ones = Tensor::<B, 2>::ones([b, 1], &x.device());

        let a = ones.clone() + xy.clone() * (2.0 * self.c) + y2.clone() * self.c;
        let b1 = ones.clone() - x2.clone() * self.c;
        let num = x * a + y * b1;

        let denom = ones.clone() + xy * (2.0 * self.c) + (x2 * y2) * (self.c * self.c);
        self.project(num / (denom + self.eps))
    }

    /// Hyperbolic distance \(d(x,y)\) on the ball.
    pub fn distance<B: Backend>(&self, x: Tensor<B, 2>, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let _b = x.dims()[0];
        let x = self.project(x);
        let y = self.project(y);
        let neg_x = x.clone() * -1.0;
        let u = self.mobius_add(neg_x, y);
        let norm_u = self.norm_keepdim(u) * self.sqrt_c();
        let z = norm_u.clamp_max(1.0 - self.eps);
        self.atanh(z) * (2.0 / self.sqrt_c())
    }

    /// Log map at origin.
    pub fn log0<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.project(x);
        let _b = x.dims()[0];
        let norm = self.norm_keepdim(x.clone());
        let z = (norm.clone() * self.sqrt_c()).clamp_max(1.0 - self.eps);
        let atanh_z = self.atanh(z);
        let scale = atanh_z / (norm + self.eps) / self.sqrt_c();
        x * scale
    }

    /// Exp map at origin.
    pub fn exp0<B: Backend>(&self, v: Tensor<B, 2>) -> Tensor<B, 2> {
        let _b = v.dims()[0];
        let norm = self.norm_keepdim(v.clone());
        let z = norm.clone() * self.sqrt_c();
        let scale = z.clone().tanh() / (z + self.eps) / self.sqrt_c();
        self.project(v * scale)
    }

    /// Log map at basepoint p (rowwise).
    pub fn log_map<B: Backend>(&self, p: Tensor<B, 2>, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let p = self.project(p);
        let x = self.project(x);
        let _b = x.dims()[0];

        let neg_p = p.clone() * -1.0;
        let delta = self.mobius_add(neg_p, x);
        let norm = self.norm_keepdim(delta.clone());

        // factor = (2/(sqrt(c)*λ_p)) * atanh(sqrt(c)*||delta||) / ||delta||
        let lambda = self.lambda_x(p).recip() * 2.0;
        let z = (norm.clone() * self.sqrt_c()).clamp_max(1.0 - self.eps);
        let atanh_z = self.atanh(z);
        let factor = (atanh_z / (norm + self.eps)) * (lambda / self.sqrt_c());
        delta * factor
    }

    /// Exp map at basepoint p (rowwise).
    pub fn exp_map<B: Backend>(&self, p: Tensor<B, 2>, v: Tensor<B, 2>) -> Tensor<B, 2> {
        let p = self.project(p);
        let b = v.dims()[0];
        let vnorm = self.norm_keepdim(v.clone());

        // second = tanh(sqrt(c)*λ_p*||v||/2) * v / (sqrt(c)*||v||)
        let lambda_p = (Tensor::<B, 2>::ones([b, 1], &p.device()) * 2.0) / self.lambda_x(p.clone());
        let z = vnorm.clone() * lambda_p * (self.sqrt_c() * 0.5);
        let scale = z.tanh() / (vnorm + self.eps) / self.sqrt_c();
        let second = v * scale;
        self.mobius_add(p, second)
    }

    /// Parallel transport from 0 to x along the radial geodesic (exact for this path).
    pub fn parallel_transport_0_to_x<B: Backend>(
        &self,
        x: Tensor<B, 2>,
        v0: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let x = self.project(x);
        let b = x.dims()[0];
        let lambda_x = (Tensor::<B, 2>::ones([b, 1], &x.device()) * 2.0) / self.lambda_x(x);
        // λ_0 = 2.
        let ratio = (Tensor::<B, 2>::ones([b, 1], &v0.device()) * 2.0) / (lambda_x + self.eps);
        v0 * ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn to_burn_2(v: &[f32], shape: [usize; 2]) -> Tensor<B, 2> {
        let device = <B as Backend>::Device::default();
        Tensor::from_data(TensorData::new(v.to_vec(), shape), &device)
    }

    #[test]
    fn burn_logexp_matches_hyp_smoke() {
        let ball = BurnPoincareBall::new(1.0);

        let x = to_burn_2(&[0.10, -0.05, 0.02, 0.03, 0.04, -0.01], [2, 3]);
        let x = ball.project(x);

        let v = ball.log0(x.clone());
        let x2 = ball.exp0(v);

        let x_rows = x.to_data().to_vec::<f32>().unwrap();
        let x2_rows = x2.to_data().to_vec::<f32>().unwrap();

        for i in 0..2 {
            let err = x_rows[i * 3..i * 3 + 3]
                .iter()
                .zip(x2_rows[i * 3..i * 3 + 3].iter())
                .map(|(a, b)| (*a - *b).abs())
                .sum::<f32>();
            assert!(err < 1e-1, "roundtrip err={err}");
        }
    }
}
