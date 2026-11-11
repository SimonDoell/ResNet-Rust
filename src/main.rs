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
<<<<<<< HEAD

use std::fs::File;
use std::io::Write;

static WIDTH:  i32 = 200;
static HEIGHT: i32 = 200;

static EPOCHS: i32 = 1_000_000;
static SAVE_STOPS: i32 = 10_000;

fn hash(x: i32, y: i32) -> f32 {
    let mut n = (x as u32)
        .wrapping_mul(374761393)
        .wrapping_add((y as u32).wrapping_mul(668265263));

    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    n ^= n >> 16;

    (n as f32) / (u32::MAX as f32)
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn noise(x: f32, y: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;

    let fx = smooth(x - ix as f32);
    let fy = smooth(y - iy as f32);

    let a = hash(ix, iy);
    let b = hash(ix + 1, iy);
    let c = hash(ix, iy + 1);
    let d = hash(ix + 1, iy + 1);

    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;

    top + (bottom - top) * fy
}

fn fbm(x: f32, y: f32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 0.5;
    let mut frequency = 1.0;
    let mut total = 0.0;

    for _ in 0..7 {
        value += noise(x * frequency, y * frequency) * amplitude;
        total += amplitude;

        frequency *= 2.0;
        amplitude *= 0.5;
    }

    value / total
}

fn clamp(v: f32) -> f32 {
    v.max(0.0).min(1.0)
}

fn target_function(i: i32, j: i32) -> (u8, u8, u8) {
    let x = i as f32 / WIDTH as f32;
    let y = j as f32 / HEIGHT as f32;

    // Center coordinates and correct aspect ratio.
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let px = (x - 0.5) * aspect;
    let py = y - 0.5;

    // Large-scale organic structures.
    let warp_x = fbm(px * 3.0 + 5.0, py * 3.0 + 2.0);
    let warp_y = fbm(px * 3.0 + 8.0, py * 3.0 + 7.0);

    let wx = px + (warp_x - 0.5) * 0.35;
    let wy = py + (warp_y - 0.5) * 0.35;

    // Multiple noise frequencies.
    let large = fbm(wx * 5.0, wy * 5.0);
    let medium = fbm(wx * 14.0, wy * 14.0);
    let fine = fbm(wx * 35.0, wy * 35.0);
    let detail = fbm(wx * 90.0, wy * 90.0);

    let terrain =
        large * 0.45 +
        medium * 0.27 +
        fine * 0.18 +
        detail * 0.10;

    // Create sharp ridges and valleys.
    let ridges = 1.0 - (terrain * 2.0 - 1.0).abs();
    let ridges = clamp(ridges * 1.8);

    // Additional fine surface detail.
    let surface = clamp(
        terrain * 0.75 +
        ridges * 0.15 +
        (detail - 0.5) * 0.25
    );

    // Height-based color transitions.
    let t = surface;

    // Dark blue, purple, turquoise, gold and bright highlights.
    let stops: [(f32, f32, f32); 6] = [
        (5.0,   8.0,  30.0),
        (22.0,  25.0, 100.0),
        (85.0,  35.0, 135.0),
        (20.0, 135.0, 145.0),
        (220.0, 145.0,  65.0),
        (255.0, 230.0, 175.0),
    ];

    let pos = t * 5.0;
    let idx = (pos.floor() as usize).min(4);
    let blend = smooth(pos - idx as f32);

    let a = stops[idx];
    let b = stops[idx + 1];

    let mut r = a.0 + (b.0 - a.0) * blend;
    let mut g = a.1 + (b.1 - a.1) * blend;
    let mut b = a.2 + (b.2 - a.2) * blend;

    // Simulated directional lighting.
    let light_noise = fbm(wx * 25.0 + 10.0, wy * 25.0);
    let lighting = 0.65 + light_noise * 0.55;

    // Highlight ridges.
    let highlight = ridges.powf(3.0) * 35.0;

    r = r * lighting + highlight;
    g = g * lighting + highlight * 0.8;
    b = b * lighting + highlight * 0.5;

    // Subtle vignette.
    let distance = (px * px + py * py).sqrt();
    let vignette = clamp(1.0 - distance * 0.75);

    r *= vignette;
    g *= vignette;
    b *= vignette;

    (
        clamp(r / 255.0).mul_add(255.0, 0.0) as u8,
        clamp(g / 255.0).mul_add(255.0, 0.0) as u8,
        clamp(b / 255.0).mul_add(255.0, 0.0) as u8,
    )
}

fn mlp_snapshot(mlp: &mut MLP, file_name: String) -> Result<(), std::io::Error> {
    let mut file = File::create(file_name)?;

    file.write(b"P6\n")?;
    file.write(format!("{} {}\n", WIDTH, HEIGHT).as_bytes())?;
    file.write(b"255\n")?;

    let mut input = Matrix::new(2, 1);

    for i in 0..WIDTH {
        for j in 0..HEIGHT {
            input.ensure_shape(2, 1);
            input.set_values(&[
                i as f32 / WIDTH as f32,
                j as f32 / HEIGHT as f32,
            ]);

            mlp.forward(&mut input);

            let pixel = [
                mapclamp(input[0], -0.8, 0.8, 0.0, 255.0) as u8,
                mapclamp(input[1], -0.8, 0.8, 0.0, 255.0) as u8,
                mapclamp(input[2], -0.8, 0.8, 0.0, 255.0) as u8,
            ];

            file.write_all(&pixel)?;
        }
    }

    Ok(())
}

fn lerp(p: f32, a: f32, b: f32) -> f32 {
    a + (b - a) * p
}

fn invlerp(x: f32, a: f32, b: f32) -> f32 {
    (x - a) / (b - a)
}

fn map(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    let p = invlerp(x, a, b);
    lerp(p, c, d)
}

fn mapclamp(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    let p = invlerp(x, a, b).clamp(0.0, 1.0);
    lerp(p, c, d)
}

fn learn(x: f32) -> f32 {
    (x * 2.0 * std::f32::consts::PI).sin() * 0.8
}

fn get_mlp_error(test: u32, mlp: &mut Graph) -> f32 {
    let mut error: f32 = 0.0;
    let mut input = Matrix::new(1, 1);

    for _ in 0..test {
        let value = rand::random_range(-1.0..1.0);
        input[0] = value;
        mlp.forward(&mut input);
        error += (input[0] - learn(value)).abs();
    }

    error / test as f32
}

fn run_mlp(iter: u32, test: u32, freq: u32) -> f32 {
    let mut mlp = Graph::new();
    mlp.lr_start = 0.001;
    mlp.lr_end = 0.000001;
    mlp.total_steps = iter;
    mlp.betta_1 = 0.9;
    mlp.betta_2 = 0.999;
    mlp.lambda = 0.0001;
    mlp.mini_batch = 64;
    mlp.clip_threshold = 8.0;

    let layers = [
        mlp.push_layer(PositionalEncoding::new(16)),
        mlp.push_layer(Dense::new(33, 32)), // 2 * 16 + 1 (identity)
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
        mlp.push_layer(Dense::new(32, 1)),
    ];

    for i in (1..(layers.len() - 4) as u32).step_by(4) {
        mlp.connect_layers(i, i + 4);
    }

    mlp.connect_layer_list(&layers);
    mlp.set_start_end(*layers.first().unwrap(), *layers.last().unwrap());
    mlp.build_exec();

    get_mlp_error(test, &mut mlp)
}

fn main() {
    let iter = 1;
    let mut error: Vec<f32> = Vec::new();

    for i in 0..iter {
        error.push(run_mlp(2_000_000, 10_000, 10_000));
        println!("Error {i}: {}", error.last().unwrap());
    }

    let mean = error.iter().sum::<f32>() / error.len() as f32;
    let std_sqr = error.iter().map(|x| (x - mean).abs()).sum::<f32>() / error.len() as f32;

    println!("Final:");
    println!("  Mean: {}", mean);
    println!("  Std Dev.: {}", std_sqr);
}
=======

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
        mlp.push_layer(PositionalEncoding::new(16)),
        mlp.push_layer(Dense::new(33, 32)), // 2 * 16 + 1 (identity)
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

    for i in (1..(layers.len() - 4) as u32).step_by(4) {
        mlp.connect_layers(i, i + 4);
    }

    mlp.connect_layer_list(&layers);
    
    // let mut mlp = Graph::new();
    
    // mlp.lr_start       = 0.001;
    // mlp.lr_end         = 0.000001;
    // mlp.total_steps    = iter;
    // mlp.betta_1        = 0.9;
    // mlp.betta_2        = 0.999;
    // mlp.lambda         = 0.0001;
    // mlp.mini_batch     = 64;
    // mlp.clip_threshold = 1.0;
    
    // let mut layers = Vec::new();
    
    // let input = mlp.push_layer(PositionalEncoding::new(16));
    // layers.push(input);
    
    // let projection = mlp.push_layer(Dense::new(33, 64));
    // layers.push(projection);
    
    // let mut previous = projection;
    
    // for _ in 0..6 {
    //     let norm = mlp.push_layer(LayerNorm::new(64));
    //     let dense1 = mlp.push_layer(Dense::new(64, 64));
    //     let act = mlp.push_layer(Activation::new_leaky_relu());
    //     let dense2 = mlp.push_layer(Dense::new(64, 64));
    
    //     mlp.connect_layers(previous, norm);
    //     mlp.connect_layers(norm, dense1);
    //     mlp.connect_layers(dense1, act);
    //     mlp.connect_layers(act, dense2);
    
    //     // Residual addition:
    //     mlp.connect_layers(previous, dense2);
    
    //     previous = dense2;
    //     layers.push(norm);
    //     layers.push(dense1);
    //     layers.push(act);
    //     layers.push(dense2);
    // }
    
    // let output = mlp.push_layer(Dense::new(64, 1));
    // mlp.connect_layers(previous, output);
    
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
>>>>>>> rewrite
