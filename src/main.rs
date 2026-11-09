#![allow(unused)]

mod matrix;
use matrix::*;

mod adamw;
use adamw::*;

mod layer;
use layer::*;

mod graph;
use graph::*;

mod dense;
use dense::*;

mod optimizer;
use optimizer::*;

fn main() {
    let mut mlp = Graph::new();

    let layers = [
        mlp.push_layer(Dense::new(1, 32)),
        mlp.push_layer(Dense::new(32, 16)),
        mlp.push_layer(Dense::new(16, 1)),
    ];

    mlp.connect_layer_list(&layers);
    mlp.set_start_end(*layers.first().unwrap(), *layers.last().unwrap());
    mlp.build_exec();

    let mut input = Matrix::new(1, 1);
    input[0] = 0.5;

    mlp.forward(&mut input);

    println!("Output: {:?}", input);
}