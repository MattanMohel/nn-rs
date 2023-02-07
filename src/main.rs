use data::dataset::Dataset;
// use matrix::MatBase;
use network::Net;
use crate::data::mnist::Reader;

pub mod activation;
pub mod cost;
pub mod parameters;
pub mod network;
pub mod data;
// pub mod matrix;
pub mod back_index;
pub mod weight_init;
pub mod draw;
pub mod matrix;

// TODO: add interface for fixing model save path if ot fails
// so you don't lose data

fn main() {
    let data = Reader::load_dataset();

    let mut net = Net::new([784, 450, 250, 10])
        .save_path("models/test")
        .build();

    net.train(data.train_set());
    net.save();

    net.accuracy(data.test_set());

    // let mut net = Net::<4>::from_file("models/model1");


    // run_sketch();
}
