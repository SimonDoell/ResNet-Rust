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

mod activation;
use activation::*;

mod layernorm;
use layernorm::*;

mod dropout;
use dropout::*;

mod positional_encoding;
use positional_encoding::*;

fn learn(x: f32) -> f32 {
    return (x * 2.0 * std::f32::consts::PI).sin() * 0.8;
}

fn get_mlp_error(test: u32, mlp: &mut Graph) -> f32 {
    let mut error: f32 = 0.0;

    let mut input = Matrix::new(1, 1);

    for t in 0..test {
        let value = rand::random_range(-1.0..1.0);
        
        input[0] = value;
        mlp.forward(&mut input);
        
        error += (input[0] - learn(value)).abs();
    }

    error = error / test as f32;

    return error;
}

fn run_mlp(iter: u32, test: u32, freq: u32) -> f32 {
    let mut mlp = Graph::new();
    mlp.lr_start       = 0.001;
    mlp.lr_end         = 0.000001;
    mlp.total_steps    = iter;
    mlp.betta_1        = 0.9;
    mlp.betta_2        = 0.999;
    mlp.lambda         = 0.0001;
    mlp.mini_batch     = 64;
    mlp.clip_threshold = 8.0;

    let layers = [
        // mlp.push_layer(PositionalEncoding::new(16)),
        mlp.push_layer(Dense::new(1, 32)),
        mlp.push_layer(Activation::new_leaky_relu()),
        mlp.push_layer(LayerNorm::new(32)),
        mlp.push_layer(Dense::new(32, 32)),
        mlp.push_layer(Activation::new_leaky_relu()),
        mlp.push_layer(LayerNorm::new(32)),
        mlp.push_layer(Dense::new(32, 32)),
        mlp.push_layer(Activation::new_leaky_relu()),
        mlp.push_layer(LayerNorm::new(32)),
        mlp.push_layer(Dense::new(32, 32)),
        mlp.push_layer(Activation::new_leaky_relu()),
        mlp.push_layer(LayerNorm::new(32)),
        mlp.push_layer(Dense::new(32, 32)),
        mlp.push_layer(Activation::new_leaky_relu()),
        mlp.push_layer(LayerNorm::new(32)),
        mlp.push_layer(Dense::new(32, 32)),
        mlp.push_layer(Activation::new_leaky_relu()),
        mlp.push_layer(LayerNorm::new(32)),
        mlp.push_layer(Dense::new(32, 32)),
        mlp.push_layer(Activation::new_leaky_relu()),
        mlp.push_layer(LayerNorm::new(32)),
        mlp.push_layer(Dense::new(32, 32)),
        mlp.push_layer(Activation::new_leaky_relu()),
        mlp.push_layer(LayerNorm::new(32)),
        mlp.push_layer(Dense::new(32, 32)),
        mlp.push_layer(Activation::new_leaky_relu()),
        mlp.push_layer(LayerNorm::new(32)),
        mlp.push_layer(Dense::new(32, 32)),
        mlp.push_layer(Activation::new_leaky_relu()),
        mlp.push_layer(LayerNorm::new(32)),
        mlp.push_layer(Dense::new(32, 32)),
        mlp.push_layer(Activation::new_leaky_relu()),
        mlp.push_layer(Dense::new(32, 1)),
    ];

    for i in (0..(layers.len() - 4) as u32).step_by(4) {
        mlp.connect_layers(i, i + 4);
    }

    mlp.connect_layer_list(&layers);
    
    mlp.set_start_end(*layers.first().unwrap(), *layers.last().unwrap());
    mlp.build_exec();

    let mut input  = Matrix::empty();
    let mut output = Matrix::empty();

    for i in 0..iter {
        input .ensure_shape((1, 1));
        output.ensure_shape((1, 1));

        input [0] = rand::random_range(-1.0..1.0);
        output[0] = learn(input[0]);

        mlp.train(&input, &output);

        if (i % freq) == 0 {
            println!("Test error: {}", get_mlp_error(test, &mut mlp));
        }
    }

    return get_mlp_error(test, &mut mlp);
}

fn main() {
    let iter = 1;

    let mut error: Vec<f32> = Vec::new();

    for i in 0..iter {
        error.push(run_mlp(2_000_000, 10_000, 10_000));
        println!("Error {i}: {}", error.last().unwrap());
    }

    let mean = error.iter().sum::<f32>() / error.len() as f32;
    let std_sqr  = error.iter().map(|x| { return (x - mean).abs(); } ).sum::<f32>() / error.len() as f32;

    println!("Final:");
    println!("  Mean:     {}", mean);
    println!("  Std Dev.: {}", std_sqr);
}