use std::f64::consts::E;

#[derive(Clone, Copy)]
pub enum Act {
    Tanh,
    Sig,
    Lin
}

fn sigmoid(n: f64) -> f64 {
    1.0 / (1.0 + E.powf(-n))
}

impl Act {
    /// Applies non-linearity function to `n`
    pub fn value(&self, n: f64) -> f64 {
        match self {
            Act::Tanh => n.tanh(),
            Act::Sig  => sigmoid(n),
            Act::Lin  => n
        }
    }

    /// Applies non-linearity derivative to `n`
    pub fn deriv(&self, n: f64) -> f64 {
        match self {
            Act::Tanh => 1.0  - n.tanh().powi(2),
            Act::Sig  => {
                let sig = sigmoid(n);
                sig * (1.0 - sig)
            }
            Act::Lin  => 1.0,
        }
    }
}