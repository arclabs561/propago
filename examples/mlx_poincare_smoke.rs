//! Minimal MLX Poincaré smoke example.
//!
//! Run:
//!   cargo run -p propago --example mlx_poincare_smoke --features backend-mlx

fn main() {
    #[cfg(not(feature = "backend-mlx"))]
    {
        eprintln!("This example requires `--features backend-mlx`.");
    }

    #[cfg(feature = "backend-mlx")]
    {
        use mlx_rs::{Array, Device};
        use propago::mlx_hyperbolic::MlxPoincareBall;

        // For a simple, deterministic smoke example, force CPU.
        Device::set_default(&Device::cpu());

        let ball = MlxPoincareBall::new(1.0);
        let x = Array::from_slice(&[0.10f32, -0.05, 0.02, 0.03, 0.04, -0.01], &[2, 3]);

        let x = ball.project(&x).unwrap();
        let v = ball.log0(&x).unwrap();
        let x2 = ball.exp0(&v).unwrap();

        let x2v: &[f32] = x2.as_slice();
        println!("x2 (first row): {:?}", &x2v[0..3]);
    }
}
