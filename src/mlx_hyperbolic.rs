//! MLX-backed hyperbolic geometry adapters (opt-in).
//!
//! This is intentionally minimal: it provides the core Poincaré ball ops needed to spec-test
//! correctness against `hyp`, and to serve as a seed for MLX-native layers.
//!
//! Notes:
//! - MLX evaluation is lazy; `as_slice()` forces evaluation.
//! - We avoid the operator-overload API since it `unwrap()`s internally.

use mlx_rs::error::Result;
use mlx_rs::{ops, Array};

/// Poincaré ball operations on MLX arrays (curvature parameter \(c>0\)).
#[derive(Debug, Clone, Copy)]
pub struct MlxPoincareBall {
    c: f32,
    eps: f32,
}

impl MlxPoincareBall {
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
        (1.0 / self.sqrt_c()) - 1e-5
    }

    fn atanh(&self, x: &Array) -> Result<Array> {
        // atanh(x) = 0.5 * ln((1+x)/(1-x))
        let one = Array::from_f32(1.0);
        let num = ops::add(x, &one)?;
        let den = ops::subtract(&one, x)?;
        let frac = ops::divide(&num, &ops::add(&den, Array::from_f32(self.eps))?)?;
        ops::multiply(&ops::log(&frac)?, &Array::from_f32(0.5))
    }

    fn norm_keepdim(&self, x: &Array) -> Result<Array> {
        // [n,d] -> [n,1]
        let sq = x.square()?;
        let sum = sq.sum_axis(1, Some(true))?;
        sum.sqrt()
    }

    fn dot_keepdim(&self, x: &Array, y: &Array) -> Result<Array> {
        ops::multiply(x, y)?.sum_axis(1, Some(true))
    }

    fn denom_1_minus_cx2(&self, x: &Array) -> Result<Array> {
        // (1 - c||x||^2), shape [n,1]
        let x2 = x.square()?.sum_axis(1, Some(true))?;
        ops::subtract(
            &Array::from_f32(1.0),
            &ops::multiply(&x2, &Array::from_f32(self.c))?,
        ) // 1 - c x2
    }

    /// Project points to stay inside the ball.
    pub fn project(&self, x: &Array) -> Result<Array> {
        let norm = self.norm_keepdim(x)?;
        let denom = ops::add(&norm, Array::from_f32(self.eps))?;
        let scale = ops::divide(&Array::from_f32(self.max_norm()), &denom)?; // [n,1]
        let scale = ops::minimum(&scale, &Array::from_f32(1.0))?;
        ops::multiply(x, &scale)
    }

    /// Möbius addition on the ball.
    pub fn mobius_add(&self, x: &Array, y: &Array) -> Result<Array> {
        let x = self.project(x)?;
        let y = self.project(y)?;

        let x2 = x.square()?.sum_axis(1, Some(true))?;
        let y2 = y.square()?.sum_axis(1, Some(true))?;
        let xy = self.dot_keepdim(&x, &y)?;

        let one = Array::from_f32(1.0);
        let a = ops::add(
            &one,
            &ops::add(
                &ops::multiply(&xy, &Array::from_f32(2.0 * self.c))?,
                &ops::multiply(&y2, &Array::from_f32(self.c))?,
            )?,
        )?;
        let b = ops::subtract(&one, &ops::multiply(&x2, &Array::from_f32(self.c))?)?;

        let num = ops::add(&ops::multiply(&x, &a)?, &ops::multiply(&y, &b)?)?;
        let denom = ops::add(
            &one,
            &ops::add(
                &ops::multiply(&xy, &Array::from_f32(2.0 * self.c))?,
                &ops::multiply(&ops::multiply(&x2, &y2)?, &Array::from_f32(self.c * self.c))?,
            )?,
        )?;

        self.project(&ops::divide(
            &num,
            &ops::add(&denom, Array::from_f32(self.eps))?,
        )?)
    }

    /// Hyperbolic distance \(d(x,y)\), shape `[n,1]`.
    pub fn distance(&self, x: &Array, y: &Array) -> Result<Array> {
        let x = self.project(x)?;
        let y = self.project(y)?;
        let neg_x = ops::negative(&x)?;
        let u = self.mobius_add(&neg_x, &y)?;
        let norm_u = ops::multiply(&self.norm_keepdim(&u)?, &Array::from_f32(self.sqrt_c()))?;
        let z = ops::minimum(&norm_u, &Array::from_f32(1.0 - self.eps))?;
        ops::multiply(&self.atanh(&z)?, &Array::from_f32(2.0 / self.sqrt_c()))
    }

    /// Log map at origin.
    pub fn log0(&self, x: &Array) -> Result<Array> {
        let x = self.project(x)?;
        let norm = self.norm_keepdim(&x)?;
        let z = ops::minimum(
            &ops::multiply(&norm, &Array::from_f32(self.sqrt_c()))?,
            &Array::from_f32(1.0 - self.eps),
        )?;
        let atanh_z = self.atanh(&z)?;
        let scale = ops::divide(&atanh_z, &ops::add(&norm, Array::from_f32(self.eps))?)?;
        let scale = ops::divide(&scale, &Array::from_f32(self.sqrt_c()))?;
        ops::multiply(&x, &scale)
    }

    /// Exp map at origin.
    pub fn exp0(&self, v: &Array) -> Result<Array> {
        let norm = self.norm_keepdim(v)?;
        let z = ops::multiply(&norm, &Array::from_f32(self.sqrt_c()))?;
        let tanh_z = ops::tanh(&z)?;
        let scale = ops::divide(&tanh_z, &ops::add(&z, Array::from_f32(self.eps))?)?;
        let scale = ops::divide(&scale, &Array::from_f32(self.sqrt_c()))?;
        self.project(&ops::multiply(v, &scale)?)
    }

    /// Log map at basepoint `p` (rowwise).
    pub fn log_map(&self, p: &Array, x: &Array) -> Result<Array> {
        let p = self.project(p)?;
        let x = self.project(x)?;

        let delta = self.mobius_add(&ops::negative(&p)?, &x)?;
        let norm = self.norm_keepdim(&delta)?;

        // factor = (2/(sqrt(c)*λ_p)) * atanh(sqrt(c)*||delta||) / ||delta||
        let denom = self.denom_1_minus_cx2(&p)?; // (1 - c||p||^2)

        let z = ops::minimum(
            &ops::multiply(&norm, &Array::from_f32(self.sqrt_c()))?,
            &Array::from_f32(1.0 - self.eps),
        )?;
        let atanh_z = self.atanh(&z)?;
        let frac = ops::divide(&atanh_z, &ops::add(&norm, Array::from_f32(self.eps))?)?;
        // Since λ_p = 2 / denom, we have 2 / (sqrt(c) * λ_p) = denom / sqrt(c).
        let factor = ops::multiply(
            &frac,
            &ops::divide(&denom, &Array::from_f32(self.sqrt_c()))?,
        )?;
        ops::multiply(&delta, &factor)
    }

    /// Exp map at basepoint `p` (rowwise).
    pub fn exp_map(&self, p: &Array, v: &Array) -> Result<Array> {
        let p = self.project(p)?;
        let vnorm = self.norm_keepdim(v)?;

        let denom = self.denom_1_minus_cx2(&p)?;
        let lambda = ops::divide(&Array::from_f32(2.0), &denom)?; // λ_p

        let z = ops::multiply(
            &ops::multiply(&vnorm, &lambda)?,
            &Array::from_f32(self.sqrt_c() * 0.5),
        )?;
        let scale = ops::divide(
            &ops::tanh(&z)?,
            &ops::add(&vnorm, Array::from_f32(self.eps))?,
        )?;
        let scale = ops::divide(&scale, &Array::from_f32(self.sqrt_c()))?;
        let second = ops::multiply(v, &scale)?;
        self.mobius_add(&p, &second)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Device;
    use ndarray::Array1;
    use proptest::prelude::*;
    use skel::Manifold;

    fn to_hyp(v: &[f32]) -> Array1<f64> {
        Array1::from_vec(v.iter().map(|x| *x as f64).collect())
    }

    fn l1(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }

    fn force_cpu() {
        // MLX defaults to GPU when available. For correctness + determinism in tests, force CPU.
        Device::set_default(&Device::cpu());
    }

    fn arb_point_vec(dim: usize) -> impl Strategy<Value = Vec<f32>> {
        // Keep points well inside the ball to avoid extreme curvature effects in randomized tests.
        // We still project before using them.
        prop::collection::vec(-0.15f32..0.15f32, dim)
    }

    #[test]
    fn mlx_logexp_matches_hyp_smoke() {
        force_cpu();
        let ball = MlxPoincareBall::new(1.0);
        let hyp_ball = hyp::PoincareBall::<f64>::new(1.0);

        let x = Array::from_slice(&[0.10f32, -0.05, 0.02, 0.03, 0.04, -0.01], &[2, 3]);
        let x = ball.project(&x).unwrap();

        let v = ball.log0(&x).unwrap();
        let x2 = ball.exp0(&v).unwrap();

        let x_rows: &[f32] = x.as_slice();
        let x2_rows: &[f32] = x2.as_slice();

        for i in 0..2 {
            let xi = to_hyp(&x_rows[i * 3..i * 3 + 3]);
            let x2i = to_hyp(&x2_rows[i * 3..i * 3 + 3]);
            let err = (&xi - &x2i).mapv(|t| t.abs()).sum();
            assert!(err < 1e-1, "roundtrip err={err}");
        }

        for i in 0..2 {
            let xi = to_hyp(&x_rows[i * 3..i * 3 + 3]);
            let v_ref = hyp_ball.log_map_zero(&xi.view());
            let x_ref = hyp_ball.exp_map_zero(&v_ref.view());
            let x_ref = x_ref.to_vec();

            let x2i = &x2_rows[i * 3..i * 3 + 3];
            let err = x_ref
                .iter()
                .zip(x2i.iter())
                .map(|(a, b)| (*a as f32 - *b).abs())
                .sum::<f32>();
            assert!(err < 1e-1, "mlx vs hyp origin roundtrip l1={err}");
        }
    }

    #[test]
    fn mlx_distance_and_mobius_add_match_hyp_smoke() {
        force_cpu();
        let ball = MlxPoincareBall::new(1.0);
        let hyp_ball = hyp::PoincareBall::<f64>::new(1.0);

        // Two safe points (well inside the ball), shape [2,3].
        let x = Array::from_slice(&[0.10f32, -0.05, 0.02, 0.03, 0.04, -0.01], &[2, 3]);
        let y = Array::from_slice(&[0.02f32, 0.01, -0.03, -0.04, 0.01, 0.02], &[2, 3]);
        let x = ball.project(&x).unwrap();
        let y = ball.project(&y).unwrap();

        // Distance parity.
        let d_m = ball.distance(&x, &y).unwrap();
        let d_m: &[f32] = d_m.as_slice();

        let x_rows: &[f32] = x.as_slice();
        let y_rows: &[f32] = y.as_slice();
        for i in 0..2 {
            let xi = to_hyp(&x_rows[i * 3..i * 3 + 3]);
            let yi = to_hyp(&y_rows[i * 3..i * 3 + 3]);
            let d_h = hyp_ball.distance(&xi.view(), &yi.view()) as f32;
            assert!(
                (d_m[i] - d_h).abs() < 2e-2,
                "distance mismatch: mlx={} hyp={}",
                d_m[i],
                d_h
            );
        }

        // Möbius add parity.
        let z_m = ball.mobius_add(&x, &y).unwrap();
        let z_m: &[f32] = z_m.as_slice();
        for i in 0..2 {
            let xi = to_hyp(&x_rows[i * 3..i * 3 + 3]);
            let yi = to_hyp(&y_rows[i * 3..i * 3 + 3]);
            let z_h = hyp_ball.mobius_add(&xi.view(), &yi.view());
            let z_h: Vec<f32> = z_h.iter().map(|t| *t as f32).collect();
            let z_mi = &z_m[i * 3..i * 3 + 3];
            let err = l1(z_mi, &z_h);
            assert!(err < 2e-2, "mobius_add l1 mismatch: {err}");
        }
    }

    #[test]
    fn mlx_basepoint_logexp_match_hyp_smoke() {
        force_cpu();
        let ball = MlxPoincareBall::new(1.0);
        let hyp_ball = hyp::PoincareBall::<f64>::new(1.0);

        // Two rows; we use the same basepoint p for both rows.
        let p = Array::from_slice(&[0.02f32, 0.01, -0.03], &[1, 3]);
        let x = Array::from_slice(&[0.10f32, -0.05, 0.02, 0.03, 0.04, -0.01], &[2, 3]);
        let p = ball.project(&p).unwrap();
        let x = ball.project(&x).unwrap();

        let p2 = ops::broadcast_to(&p, &[2, 3]).unwrap();
        let v_m = ball.log_map(&p2, &x).unwrap();
        let x2_m = ball.exp_map(&p2, &v_m).unwrap();

        let x2_m: &[f32] = x2_m.as_slice();
        let x: &[f32] = x.as_slice();
        for i in 0..2 {
            let xi = to_hyp(&x[i * 3..i * 3 + 3]);
            let pi = to_hyp(p.as_slice());
            let v_h = hyp_ball.log_map(&pi.view(), &xi.view());
            let x2_h = hyp_ball.exp_map(&pi.view(), &v_h.view());
            let x2_h: Vec<f32> = x2_h.iter().map(|t| *t as f32).collect();

            let x2_mi = &x2_m[i * 3..i * 3 + 3];
            let err = l1(x2_mi, &x2_h);
            assert!(err < 5e-2, "basepoint exp(log(x)) mismatch l1={err}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            max_shrink_iters: 0,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_mlx_distance_matches_hyp(x in arb_point_vec(3), y in arb_point_vec(3)) {
            force_cpu();
            let ball = MlxPoincareBall::new(1.0);
            let hyp_ball = hyp::PoincareBall::<f64>::new(1.0);

            let x_m = ball.project(&Array::from_slice(&x, &[1, 3])).unwrap();
            let y_m = ball.project(&Array::from_slice(&y, &[1, 3])).unwrap();
            let d_m = ball.distance(&x_m, &y_m).unwrap();
            let d_m: &[f32] = d_m.as_slice();

            let x_h = to_hyp(x_m.as_slice());
            let y_h = to_hyp(y_m.as_slice());
            let d_h = hyp_ball.distance(&x_h.view(), &y_h.view()) as f32;

            prop_assert!((d_m[0] - d_h).abs() < 5e-2, "distance mismatch: mlx={} hyp={}", d_m[0], d_h);
        }

        #[test]
        fn prop_mlx_mobius_add_matches_hyp(x in arb_point_vec(3), y in arb_point_vec(3)) {
            force_cpu();
            let ball = MlxPoincareBall::new(1.0);
            let hyp_ball = hyp::PoincareBall::<f64>::new(1.0);

            let x_m = ball.project(&Array::from_slice(&x, &[1, 3])).unwrap();
            let y_m = ball.project(&Array::from_slice(&y, &[1, 3])).unwrap();
            let z_m = ball.mobius_add(&x_m, &y_m).unwrap();
            let z_m: &[f32] = z_m.as_slice();

            let x_h = to_hyp(x_m.as_slice());
            let y_h = to_hyp(y_m.as_slice());
            let z_h = hyp_ball.mobius_add(&x_h.view(), &y_h.view());
            let z_h: Vec<f32> = z_h.iter().map(|t| *t as f32).collect();

            let err = l1(z_m, &z_h);
            prop_assert!(err < 5e-2, "mobius_add l1 mismatch: {err}");
        }

        #[test]
        fn prop_mlx_basepoint_exp_log_roundtrip_matches_hyp(p in arb_point_vec(3), x in arb_point_vec(3)) {
            force_cpu();
            let ball = MlxPoincareBall::new(1.0);
            let hyp_ball = hyp::PoincareBall::<f64>::new(1.0);

            let p_m = ball.project(&Array::from_slice(&p, &[1, 3])).unwrap();
            let x_m = ball.project(&Array::from_slice(&x, &[1, 3])).unwrap();
            let v_m = ball.log_map(&p_m, &x_m).unwrap();
            let x2_m = ball.exp_map(&p_m, &v_m).unwrap();

            let x2_m: &[f32] = x2_m.as_slice();
            let p_h = to_hyp(p_m.as_slice());
            let x_h = to_hyp(x_m.as_slice());
            let v_h = hyp_ball.log_map(&p_h.view(), &x_h.view());
            let x2_h = hyp_ball.exp_map(&p_h.view(), &v_h.view());
            let x2_h: Vec<f32> = x2_h.iter().map(|t| *t as f32).collect();

            let err = l1(x2_m, &x2_h);
            prop_assert!(err < 1e-1, "basepoint exp(log(x)) mismatch l1={err}");
        }
    }
}
