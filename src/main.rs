use activation::Act;
use data::{mnist::Reader, dataset::Dataset};
use matrix::Mat;
use network::FeedForward;
use crate::{matrix::MatBase, data::mnist::one_hot};

pub mod parameters;
pub mod network;
pub mod data;
pub mod matrix;
pub mod back_index;
pub mod weight_init;
pub mod draw;
pub mod activation;
pub mod cost;

fn main() {
    // train_mnist_to_digit();
    draw::run_sketch();
}

fn train_mnist_to_digit() {
    let data = Reader::load_dataset();

    let mut net = FeedForward::new([784, 450, 300, 80, 10])
        .save_path("src/models/test")
        .build();

    net.train(data.train_set());
    net.save_model().unwrap();

    println!("acc: {}", net.accuracy(data.test_set()));
}

fn train_mnist_to_image() {
    let data = Reader::load_dataset();

    let mut net = FeedForward::new([10, 80, 200, 784])
        .save_path("src/models/test_rev")
        .build();

    net.train((data.train_labels(), data.train_data()));
    net.save_model().unwrap();

    for i in 0..10 {
        let input = one_hot(i);
        let out = net.predict(&input);

        for (i, byte) in out.data().iter().enumerate() {
            if *byte > 0.8 {
                print!("■ ")
            } else if *byte > 0.4 {
                print!("▧ ")
            } else if *byte > 0.0 {
                print!("□ ")
            } else {
                print!("- ")
            };

            if i % 28 == 0 {
                println!()
            } 
        }
    }

    println!("acc: {}", net.accuracy(data.test_set()));
}

#[test]
fn linear_regression() {
    let xs = [0.0,  1.0].map(|n| Mat::from_elem(n));
    let ys = [10.0, 5.0].map(|n| Mat::from_elem(n));

    let mut nn = FeedForward::new([1, 1])
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
