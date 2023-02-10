use serde::{Deserialize, Serialize};
use crate::{activation::Act, cost::Cost, network::FeedForward, weight_init::Weight};

/// Default learn rate
const LEARN_RATE: f32 = 0.01;
/// Default momentum rate
const MOMENTUM: f32 = 0.80;
/// Default batch size
const BATCH_SIZE: usize = 32;
/// Default train epochs count
const EPOCHS: usize = 5;
/// Default activation function
const ACTIVATION: Act = Act::Tanh;
/// Default cost function
const COST: Cost = Cost::MSE;
/// Default weight initialization method
const WEIGHT: Weight = Weight::Range(-0.2, 0.2);
/// Default model save path
const SAVE_PATH: &str = "src/models/model";

#[derive(Serialize, Deserialize, Clone)]
pub struct Params<const L: usize> {
    pub form: Vec<usize>,
    pub learn_rate: f32,
    pub momentum: f32,
    pub batch_size: usize,
    pub epochs:  usize,
    pub weight: Weight,
    pub act:       Act,
    pub cost:      Cost,
    pub shuffle:   bool,
    pub verbose:   bool,
    pub save_path: String
}

impl<const L: usize> From<[usize; L]> for Params<L> {
    fn from(form: [usize; L]) -> Self {
        Self {
            form: form.to_vec(),
            learn_rate: LEARN_RATE,
            momentum: MOMENTUM,
            batch_size: BATCH_SIZE,
            epochs: EPOCHS,
            weight: WEIGHT,
            act: ACTIVATION,
            cost: COST,
            shuffle: true,
            verbose: true,
            save_path: SAVE_PATH.to_string()
        }    
    }
}

impl<const L: usize> Params<L> {
    pub fn new(form: [usize; L]) -> Self {
        Self::from(form)
    }

    /// Build `Net` 
    pub fn build(&self) -> FeedForward<L> {
        FeedForward::from(self.clone())
    }

    /// Set model `learn_rate`
    pub fn learn_rate(&mut self, learn_rate: f32) -> &mut Self {
        self.learn_rate = learn_rate;
        self
    } 

    /// Set model `momentum` rate
    pub fn momentum(&mut self, momentum: f32) -> &mut Self {
        self.momentum = momentum;
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
    pub fn weight(&mut self, weight_init: Weight) -> &mut Self {
        self.weight = weight_init;
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

    /// Set model `save_path`
    pub fn save_path(&mut self, save_path: &str) -> &mut Self {
        self.save_path = save_path.to_string();
        self
    }
}