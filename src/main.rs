use activation::Act::*;
use matrix::MatBase;
use network::Net;
use crate::{data::mnist::{Reader, DataType::Train}, matrix::Mat, weight_init::WeightInit::*};

pub mod activation;
pub mod cost;
pub mod parameters;
pub mod network;
pub mod data;
pub mod matrix;
pub mod back_index;
pub mod weight_init;

// TODO: create a "DataReader" trait that feeds into "Net"

fn main() {
    // let mut nn = Net::new([3, 100, 200, 3])
    //     .epochs(10)
    //     .activation(Tanh)
    //     .learn_rate(0.1)
    //     .batch_size(1)
    //     .weight_init(Value(0.02023))
    //     .verbose(false)
    //     .build();

    // let x = Mat::from_vec((3, 1), vec![0.123, 0.654, -0.38]);
    // let y = Mat::from_vec((3, 1), vec![0.323, -0.723, 0.67]);

    // nn.train(&[x.clone()], &[y.clone()]);
    // let out = nn.predict(&x);
    // println!("{:?}", out.buf());

    let data = Reader::new();

    let mut net = Net::new([784, 450, 250, 10])
        .epochs(5)
        .shuffle(true)
        .activation(Tanh)
        .learn_rate(0.01)
        .batch_size(32)
        .weight_init(RangeNorm(-1.0, 1.0, 10.0))
        .build();

    net.train(&data.train_images()[0..500], &data.train_labels()[0..500]);

    // for i in 0..10 {
    //     data.print_image(data::mnist::DataType::Test, i);
        
    //     let image = &data.test_images()[i];
    //     let out = net.predict(&image);

    //     println!("{:?}", out.buf());
    // }
}
