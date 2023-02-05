
#[derive(Clone, Copy)]
pub enum Cost {
    MSE
}

impl Cost {
    /// Applies cost function to `err`
    pub fn value(&self, err: f32) -> f32 {
        match self {
            Cost::MSE => err.powi(2)
        }
    }

    /// Applies cost derivative to `err`
    pub fn deriv(&self, err: f32) -> f32 {
        match self {
            Cost::MSE => 2.0 * err
        }
    }
}