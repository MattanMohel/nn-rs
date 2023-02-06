use std::ops::Mul;

use activation::Act::*;
use nalgebra::{DMatrix, Matrix2x3, Matrix3x1};
// use matrix::MatBase;
use network::{Net, Mat};
use crate::{data::mnist::{Reader}, weight_init::WeightInit::*};

pub mod activation;
pub mod cost;
pub mod parameters;
pub mod network;
pub mod data;
// pub mod matrix;
pub mod back_index;
pub mod weight_init;

// TODO: create a "DataReader" trait that feeds into "Net"

fn main() {
    let data = Reader::new();

    let mut net = Net::new([784, 450, 250, 10])
        .epochs(10)
        .shuffle(true)
        .activation(Tanh)
        .learn_rate(0.01)
        .batch_size(32)
        .weight_init(RangeNorm(-1., 1., 5.))
        .build();

    net.train(&data.train_images()[0..500], &data.train_labels()[0..500]);

    // for i in 0..10 {
    //     data.print_image(data::mnist::DataType::Train, i);

    //     let image = &data.train_images()[i];
    //     let out = net.predict(&image);

    //     println!("{}", out);
    // }
}
