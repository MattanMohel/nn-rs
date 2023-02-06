use rand::seq::SliceRandom;
use serde::{Serialize, Deserialize};
use std::{time::Instant, ops::{AddAssign}, fs::File};
use nalgebra::DMatrix;
use crate::{
    activation::Act, 
    parameters::Params, 
    back_index::Side::Rev,
    cost::Cost,
};

pub type Mat = DMatrix<f32>;

/// Represents a neuron layer
#[derive(Serialize, Deserialize)]
struct Layer {
    weights:    Mat,
    w_grad:     Mat,
    w_grad_acc: Mat,
    biases:     Mat,
    grad:       Mat,
    grad_acc:   Mat,
    sums:       Mat,
    act:        Act
}

impl Layer {
    /// Creates `Layer` given nodes going `n_in` and `n_out`
    pub fn new<const L: usize>(params: &Params<L>, n_in: usize, n_out: usize) -> Self {
        Self {
            weights:    params.weight_init.init(n_in, n_out),
            w_grad:     Mat::zeros(n_out, n_in),
            w_grad_acc: Mat::zeros(n_out, n_in),
            biases:     Mat::zeros(n_out, 1),
            grad:       Mat::zeros(n_out, 1),
            grad_acc:   Mat::zeros(n_out, 1),
            sums:       Mat::zeros(n_out, 1),
            act: params.act
        }
    }

    /// Computes forward pass from layer `l → l1`
    /// returning the activation of the next layer
    /// 
    /// ## Equations
    /// - `Z ₗ = W ₗ x A ₗ + B ₗ`
    /// - `A ₗ₊₁ = σ( Zₗ )`
    #[inline]
    fn forward_pass(&mut self, a: &Mat) -> Mat {
        self.weights.mul_to(a, &mut self.sums);
        self.sums.add_assign(&self.biases);
        self.sums.map(|n| self.act.value(n))
    }

    /// Computes a backward pass from `l ← l1`
    /// setting the error of the previous layer
    /// 
    /// ## Equations
    /// - `ϵ ₗ₋₁ = W ₗᵀ x ϵ ₗ . σ'( Z ₗ₋₁ )`
    #[inline]
    fn backward_pass(&self, l_prev: &mut Layer) {
        self.weights.transpose().mul_to(&self.grad, &mut l_prev.grad);
        l_prev.grad.component_mul_assign(&l_prev.sums.map(|n| l_prev.act.deriv(n)));
    }

    /// Computes weight error on layer `l`
    /// 
    /// ## Equations
    /// - `ΔW ₗ₊₁ = ϵ ₗ₊₁ x A ₗ₋₁ᵀ`
    #[inline]
    fn weight_error(&mut self, a_prev: &Mat) {
        self.grad.mul_to(&a_prev.transpose(), &mut self.w_grad);
    }

    /// Computes error on output layer `L`
    /// 
    /// ## Equations
    /// - `ϵ ₗ = cost'( y - A ₗ ) . σ'( Z ₗ )`
    #[inline]
    fn output_err(&mut self, cost: Cost, y: &Mat, a_out: &Mat) {
        y.sub_to(a_out, &mut self.grad);
        self.grad.apply(|n| *n = cost.deriv(*n));
        self.grad.component_mul_assign(&self.sums.map(|n| self.act.deriv(n)));
    }

    /// Accumulate error
    #[inline]
    fn accum_err(&mut self) {
        self.grad_acc.add_assign(&self.grad);
        self.w_grad_acc.add_assign(&self.w_grad);
    }

    /// Apply accumulated error
    #[inline]
    fn apply_err(&mut self, eta: f32) {
        self.weights.add_assign(&self.w_grad_acc.scale(eta));
        self.biases.add_assign(&self.grad_acc.scale(eta));
    }

    /// Clears propagation data
    #[inline]
    fn clear_prop(&mut self) {
        self.sums.fill(0.0);
        self.grad.fill(0.0);
        self.w_grad.fill(0.0);
    }

    /// Clears accumulation data
    #[inline]
    fn clear_accum(&mut self) {
        self.w_grad_acc.fill(0.0);
        self.grad_acc.fill(0.0);
    }
}

/// Neural Network
#[derive(Serialize, Deserialize)]
pub struct Net<const L: usize> {
    params: Params<L>,
    layers: Vec<Layer>,
    acts:   Vec<Mat>
}

impl<const L: usize> From<Params<L>> for Net<L> {
    fn from(params: Params<L>) -> Self {
        let acts = params.form
            .iter()
            .map(|l| Mat::zeros(*l, 1))
            .collect();

        let layers = params.form
            .windows(2)
            .map(|l| Layer::new(&params, l[0], l[1]))
            .collect();

        Self {
            acts,
            layers,
            params
        }
    }
}

impl<const L: usize> Net<L> {
    /// Creates a new `Net` builder
    pub fn new(form: [usize; L]) -> Params<L> {
        Params::from(form)
    }

    pub fn save(&self) {
        let net = serde_json::to_string(&self)
            .expect("could not serialize model!");

        let work_dir = std::env::current_dir().unwrap();

        // absolute path
        let abs_path = work_dir.join(&self.params.save_path);

        File::create(&abs_path)
            .expect("couldn't create model save file!");

        std::fs::write(&abs_path, net)
            .expect("couldn't write model to file!");
    }

    pub fn from_file(path: &str) -> Self {
        let work_dir = std::env::current_dir().unwrap();

        // absolute path
        let abs_path = work_dir.join(&path);

        let src = std::fs::read_to_string(&abs_path)
            .expect("couldn't read model file!");
        
        serde_json::from_str(&src)
            .expect("couldn't deserialize model!")
    }


    /// Forward propagates and returns a model prediction
    pub fn predict(&mut self, x: &Mat) -> &Mat {
        self.forward_pass(x);
        &self.acts[Rev(0)]
    }

    /// Trains the model on inputs `xs` and labels `ys`
    pub fn train(&mut self, xs: &[Mat], ys: &[Mat]) {
        assert_eq!(xs.len(), ys.len());

        // index map for shuffling the immutable data
        let mut indices: Vec<_> = (0..xs.len()).collect();

        for epoch in 0..self.params.epochs {
            let timer = Instant::now();

            if self.params.verbose {
                println!("epoch {} of {}", epoch+1, self.params.epochs);
            }
            if self.params.shuffle {
                indices.shuffle(&mut rand::thread_rng());
            }

            // clear accumulated gradients
            for layer in self.layers.iter_mut() {
                layer.clear_accum();
            }

            let mut samples = 0;

            for i in 0..xs.len() {
                // evaluate gradients
                self.backward_pass(&xs[i], &ys[i]);
                
                // accumulate evaluated gradients
                for layer in self.layers.iter_mut() {
                    layer.accum_err();
                }

                samples += 1;

                if samples == self.params.batch_size || i + 1 == xs.len() {
                    // learn rate
                    let eta = self.params.learn_rate / samples as f32;
                    
                    // apply accumulated gradients
                    for layer in self.layers.iter_mut() {
                        layer.apply_err(eta);
                        layer.clear_accum();
                    }

                    samples = 0;
                }
            }

            if self.params.verbose {
                let elapsed = timer.elapsed();
                println!("finished in {}s", elapsed.as_secs());
                let accuracy = self.accuracy(xs, ys);
                println!("accuracy: {}", accuracy);
            }
        }
    }

    /// Forward propagates input `x`
    /// 
    /// ## Note
    /// `Self` caches the propagated activations
    pub fn forward_pass(&mut self, x: &Mat) {
        self.acts[0] = x.clone();

        for l in 0..L-1 {
            // forward propagate layer activations
            self.acts[l+1] = self.layers[l].forward_pass(&self.acts[l]);
        }
    }

    /// Backward propagates input `x` against label `y`
    /// 
    /// ## Note
    /// `Self` caches the propagated error 
    pub fn backward_pass(&mut self, x: &Mat, y: &Mat) {
        for layer in self.layers.iter_mut() {
            layer.clear_prop();
        }

        // propagate input
        self.forward_pass(x);
        // evaluate output error
        self.layers[Rev(0)].output_err(self.params.cost, y, &self.acts[Rev(0)]);

        for l in 0..L-1 {
            // evaluate layer weight error
            self.layers[Rev(l)].weight_error(&self.acts[Rev(l+1)]);

            // backward propagate layer gradients
            let split = Rev(l).to_index(self.layers.len());
            if let ([.., l], [l1, ..]) = self.layers.split_at_mut(split) {
                l1.backward_pass(l);
            } 
        }
    }

    /// Measures `accuracte_predictions / samples`
    /// 
    /// ## TODO
    /// Abstract this method into trait
    /// - enforce size between `xs` and `ys`  
    /// - add variable accuracy function 
    pub fn accuracy(&mut self, xs: &[Mat], ys: &[Mat]) -> f32 {
        fn max(mat: &Mat) -> usize {
            mat
                .iter()
                .enumerate()
                .reduce(|max, n| if max.1 >= n.1 { max } else { n })
                .unwrap()
                .0
        }

        let mut accurate = 0;

        for (x, y) in xs.iter().zip(ys.iter()) {
            let out = max(&self.predict(x));

            if out == max(y) {
                accurate += 1;
            }
        }

        accurate as f32 / xs.len() as f32
    }
}