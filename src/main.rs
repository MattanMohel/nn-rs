use data::{mnist::Reader, dataset::Dataset};
use network::Net;

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
    let data = Reader::load_dataset();

    let mut net = Net::new([784, 450, 250, 10])
        .save_path("src/models/test1")
        .build();

    net.train(data.train_set());
    net.save();

    println!("acc: {}", net.accuracy(data.test_set()));

    draw::run_sketch();
}
