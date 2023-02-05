use crate::{activation::Act, cost::Cost, network::Net, weight_init::WeightInit};

/// Default learn rate
const LEARN_RATE: f32 = 0.01;
/// Default batch size
const BATCH_SIZE: usize = 32;
/// Default train epochs count
const EPOCHS: usize = 5;
/// Default activation function
const ACTIVATION: Act = Act::Tanh;
/// Default cost function
const COST: Cost = Cost::MSE;
/// Default weight initialization method
const WEIGHT_INIT: WeightInit = WeightInit::RangeNorm(
    -1.0, // min
     1.0, // max
     5.0  // normalizer
);

#[derive(Clone)]
pub struct Params<const L: usize> {
    pub form: [usize; L],
    pub learn_rate:  f32,
    pub batch_size:  usize,
    pub epochs:      usize,
    pub weight_init: WeightInit,
    pub act:         Act,
    pub cost:        Cost,
    pub shuffle:     bool,
    pub verbose:     bool,
}

impl<const L: usize> From<[usize; L]> for Params<L> {
    fn from(form: [usize; L]) -> Self {
        Self {
            form,
            learn_rate: LEARN_RATE,
            batch_size: BATCH_SIZE,
            epochs: EPOCHS,
            weight_init: WEIGHT_INIT,
            act: ACTIVATION,
            cost: COST,
            shuffle: true,
            verbose: true
        }    
    }
}

impl<const L: usize> Params<L> {
    pub fn new(form: [usize; L]) -> Self {
        Self::from(form)
    }

    /// Build `Net` 
    pub fn build(&self) -> Net<L> {
        Net::from(self.clone())
    }

    /// Set model `learn_rate`
    pub fn learn_rate(&mut self, learn_rate: f32) -> &mut Self {
        self.learn_rate = learn_rate;
        self
    } 

    /// Set model `batch_size`
    pub fn batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = batch_size;
        self
    }

    /// Set model train `epochs`
    pub fn epochs(&mut self, epochs: usize) -> &mut Self {
        self.epochs = epochs;
        self
    }

    /// Set model `weight_init` method
    pub fn weight_init(&mut self, weight_init: WeightInit) -> &mut Self {
        self.weight_init = weight_init;
        self
    }

    /// Set model `activation`
    pub fn activation(&mut self, act: Act) -> &mut Self {
        self.act = act;
        self
    }

    /// Set model `cost`
    pub fn cost(&mut self, cost: Cost) -> &mut Self {
        self.cost = cost;
        self
    }

    /// Set whether model shuffles its data per epoch
    pub fn shuffle(&mut self, shuffle: bool) -> &mut Self {
        self.shuffle = shuffle;
        self
    }

    /// Set whether model prints stats per epoch
    pub fn verbose(&mut self, verbose: bool) -> &mut Self {
        self.verbose = verbose;
        self
    }
}