use crate::matrix::Mat;

#[derive(Clone, Copy)]
pub enum WeightInit {
    Value(f32),
    Sqrt,
    Range(
        f32, // min, 
        f32 // max
    ),
    RangeNorm(
        f32, // min
        f32, // max
        f32, // normalize
    )
}

impl WeightInit {
    /// Creates weights given nodes going `n_in` and `n_out`
    pub fn init(&self, n_in: usize, n_out: usize) -> Mat {
        match self {
            WeightInit::Value(n) => Mat::filled((n_out, n_in), *n),
            WeightInit::Sqrt => {
                let bounds = 1.0 / (n_in as f32).sqrt();
                Mat::random((n_out, n_in), -bounds, bounds).map(|n| n.sqrt())
            }
            WeightInit::Range(min, max) => Mat::random((n_out, n_in), *min, *max),
            WeightInit::RangeNorm(min, max, norm) => Mat::random((n_out, n_in), *min, *max).scale(1.0 / norm),
        }
    }
}