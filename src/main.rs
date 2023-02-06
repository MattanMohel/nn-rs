use std::ops::Mul;

use activation::Act::*;
use draw::run_sketch;
use nalgebra::{DMatrix, Matrix2x3, Matrix3x1};
// use matrix::MatBase;
use network::{Net, Mat};
use crate::{data::mnist::{Reader, DataType::Test}, weight_init::WeightInit::*};

pub mod activation;
pub mod cost;
pub mod parameters;
pub mod network;
pub mod data;
// pub mod matrix;
pub mod back_index;
pub mod weight_init;
pub mod draw;

// TODO: create a "DataReader" trait that feeds into "Net"

fn main() {
    let data = Reader::new();

    // let mut net = Net::new([784, 450, 250, 10]).build();

    // net.train(&data.train_images(), &data.train_labels());
    // net.save();

    run_sketch();
}
