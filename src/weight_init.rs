use serde::{Serialize, Deserialize};

use crate::network::Mat;

#[derive(Serialize, Deserialize, Clone, Copy)]
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
            WeightInit::Value(n) => {
                Mat::from_element(n_out, n_in, *n)
            }
            WeightInit::Sqrt => {
                let bounds = 1.0 / (n_in as f32).sqrt();
                Mat::new_random(n_out, n_in).map(|n| bounds * n)
            }
            WeightInit::Range(min, max) => {
                Mat::new_random(n_out, n_in).map(|n| (max - min) * n + min)
            }
            WeightInit::RangeNorm(min, max, norm) => {
                Mat::new_random(n_out, n_in).map(|n| ((max - min) * n + min) / norm)
            }
        }
    }
}