# nn-rs

A project for exploring perceptron neural network in Rust

### What is This Exactly?

The network aims to identify 'handwritten' digits from ```0-9``` (hopefully accurately) using the MNIST dataset

### Drawing 

The library includes a drawing app designed to predict hand-written digits using the MNIST dataset. The current model achieves a 95% accuracy on both the training and testing MNIST dataset

### Goals

- [x] get matrix library up and running
- [x] MNIST data loader
- [x] get basic network structure
- [x] drawing app
- [x] model serialization with Serde
- [x] momentum optimizer
- [x] improve model accuracy / generalization
- [x] rework matrix library trait abstractions
- [x] create generic Data trait for feeding network data

### Note

This project is meant as a hobby. All the logic, including the matrix library, is implemented from scratch. 

