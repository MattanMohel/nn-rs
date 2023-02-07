use activation::Act;
use data::{mnist::Reader, dataset::Dataset};
use matrix::Mat;
use network::Net;
use weight_init::Weight;

use crate::matrix::MatBase;

pub mod parameters;
pub mod network;
pub mod data;
pub mod matrix;
pub mod back_index;
pub mod weight_init;
pub mod draw;
pub mod activation;
pub mod cost;

// TODO: add interface for fixing model save path if ot fails
// so you don't lose data

fn main() {
    draw::run_sketch();
}

#[test]
fn train_mnist() {
    let data = Reader::load_dataset();

    let mut net = Net::new([784, 450, 250, 10])
        .save_path("src/models/test")
        .build();

    net.train(data.train_set());
    net.save().unwrap();

    println!("acc: {}", net.accuracy(data.test_set()));
}

#[test]
fn linear_regression() {
    let xs = [0.0,  1.0].map(|n| Mat::from_elem(n));
    let ys = [10.0, 5.0].map(|n| Mat::from_elem(n));

    let mut nn = Net::new([1, 1])
        .activation(Act::Lin)
        .batch_size(1)
        .learn_rate(0.5)
        .build();

    nn.train((&xs, &ys));

    for i in 0..10 {
        let out = nn.predict(&Mat::from_elem(i as f32));
        println!("({}, {:?})", i, out.data());
    }
}
