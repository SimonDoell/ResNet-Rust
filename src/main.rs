#![allow(unused)]

// agy --conversation=6ba0695a-e844-44e0-ba45-d55f26b295c9

use std::fs::*;
use std::io::prelude::*;

mod matrix;
use matrix::*;

mod mlp;
use mlp::*;

mod dense;
use dense::*;

mod actv;
use actv::*;

mod layer_norm;
use layer_norm::*;

mod positional_encoding;
use positional_encoding::*;

mod dropout;
use dropout::*;

use sfml::graphics::*;
use sfml::system::Vector2f;
use sfml::window::*;

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
        clamp(r / 255.0) .mul_add(255.0, 0.0) as u8,
        clamp(g / 255.0) .mul_add(255.0, 0.0) as u8,
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
                i as f32 / WIDTH  as f32,
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

    return Ok(());
}

fn lerp(p: f32, a: f32, b: f32) -> f32 {
    return a + (b - a) * p;
}

fn invlerp(x: f32, a: f32, b: f32) -> f32 {
    return (x - a) / (b - a);
}

fn map(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    let p = invlerp(x, a, b);
    return lerp(p, c, d);
}

fn mapclamp(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    let p = invlerp(x, a, b).clamp(0.0, 1.0);
    return lerp(p, c, d);
}

fn learn(x: f32) -> f32 {
    return (3.0 * x * 2.0 * std::f32::consts::PI).sin() * 0.8;
}

fn main() {
    let mut mlp = MLP::new();
    mlp.learning_rate = 0.001;
    mlp.betta_1       = 0.9;
    mlp.betta_2       = 0.999;
    mlp.lambda        = 0.00001;
    mlp.mini_batch    = 32;

    mlp.push_layer(PositionalEncoding::new(16));

    mlp.push_layer(Dense::new(32, 128));
    mlp.push_layer(Activation::tanh());
    
    mlp.push_layer(Dense::new(128, 128));
    mlp.push_layer(Activation::tanh());
    
    mlp.push_layer(Dense::new(128, 64));
    mlp.push_layer(Activation::tanh());
    
    mlp.push_layer(Dense::new(64, 1));

    let iter = 1_000_000;
    let test = 1_000;

    let mut input  = Matrix::new(1, 0);
    let mut output = Matrix::new(1, 0);

    for i in 0..iter {
        input .ensure_shape(1, 1);
        output.ensure_shape(1, 1);
        
        input [0] = rand::random_range(0.0..1.0);
        output[0] = learn(input[0]);

        mlp.train(&input, &output);

        if i % test == 0 {
            let mut error = 0.0;

            for k in 0..test {
                input [0] = rand::random_range(0.0..1.0);
                output[0] = learn(input[0]);

                mlp.forward(&mut input);

                error += (input[0] - output[0]).abs();
            }

            error /= test as f32;

            println!("Error: {error}");
        }
    }
}

// fn main() -> Result<(), std::io::Error> {
//     let mut file = File::create("./target.ppm")?;

//     file.write(b"P6\n")?;
//     file.write(format!("{} {}\n", WIDTH, HEIGHT).as_bytes())?;
//     file.write(b"255\n")?;

//     let mut file_pixels = Vec::<u8>::new();

//     for i in 0..WIDTH {
//         for j in 0..HEIGHT {
//             let pixels = target_function(i, j);
            
//             file.write_all(&[pixels.0, pixels.1, pixels.2])?;

//             file_pixels.extend_from_slice(&[
//                 pixels.0,
//                 pixels.1,
//                 pixels.2,
//             ]);
//         }
//     }

//     let mut mlp = MLP::new();
//     mlp.learning_rate = 0.001;
//     mlp.mini_batch    = 10;

//     mlp.push_layer(PositionalEncoding::new(8));
    
//     mlp.push_layer(Dense::new(32, 128));
//     mlp.push_layer(Activation::leaky_relu());

//     mlp.push_layer(Dense::new(128, 64));
//     mlp.push_layer(Activation::leaky_relu());

//     mlp.push_layer(Dense::new(64, 32));
//     mlp.push_layer(Activation::leaky_relu());

//     mlp.push_layer(Dense::new(32, 3));
//     mlp.push_layer(Activation::tanh());

//     let mut input  = Matrix::new(2, 1);
//     let mut output = Matrix::new(3, 1);

//     let mut counter = 0;
//     let mut image_counter = 0;
    
//     for e in 0..EPOCHS {
//         let pixel = (rand::random_range(0..WIDTH) as i32, rand::random_range(0..HEIGHT) as i32);

//         let pixel_value = (
//             file_pixels[((pixel.0 + pixel.1 * WIDTH) * 3 + 0) as usize],
//             file_pixels[((pixel.0 + pixel.1 * WIDTH) * 3 + 1) as usize],
//             file_pixels[((pixel.0 + pixel.1 * WIDTH) * 3 + 2) as usize],
//         );

//         input.set_values(&[pixel.0 as f32 / WIDTH as f32, pixel.1 as f32 / HEIGHT as f32]);

//         output.set_values(&[
//             map(pixel_value.0 as f32 / 255.0, 0.0, 1.0, -0.8, 0.8),
//             map(pixel_value.1 as f32 / 255.0, 0.0, 1.0, -0.8, 0.8),
//             map(pixel_value.2 as f32 / 255.0, 0.0, 1.0, -0.8, 0.8),
//         ]);

//         mlp.train(&input, &output);

//         counter += 1;

//         if counter % SAVE_STOPS == 0 {
//             let mut file_name = format!("./ai_saves/img_{}.ppm", image_counter);
//             println!("Generating: {}", file_name);
//             mlp_snapshot(&mut mlp, file_name)?;
//             image_counter += 1;
//         }
//     }
    

//     return Ok(());
// }

// fn main() {
//     let mut window = RenderWindow::new(VideoMode::new(800, 600, 32), "Title", Style::CLOSE | Style::RESIZE, &Default::default()).expect("Failed to create window");

//     let mut rect = RectangleShape::new();
//     rect.set_fill_color(Color::GREEN);
//     rect.set_position(Vector2f::new(0.0, 0.0));
//     rect.set_size(Vector2f::new(100.0, 100.0));
    
//     while window.is_open() {
//         while let Some(event) = window.poll_event() {
//             match event {
//                 Event::Closed => window.close(),
//                 _ => {}
//             }
//         }
//         if Key::is_pressed(Key::Escape) {window.close();}
        
//         window.clear(Color::rgb(33, 33, 33));

//         window.draw(&rect);

//         window.display();
//     }
// }