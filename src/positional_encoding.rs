use crate::matrix::*;
use crate::mlp::*;

pub struct PositionalEncoding {
    frequencys: usize,
    cached_activation: Matrix,
}

impl PositionalEncoding {
    pub fn new(frequencys: usize) -> Self {
        return Self { frequencys: frequencys, cached_activation: Matrix::new(0, 0) };
    }
}

impl Layer for PositionalEncoding {
    fn forward(&mut self, input: &mut Matrix, train: bool) {
        let n = input.get_rows() * input.get_cols();

        let mut output = Matrix::new(2 * n * self.frequencys, 1);

        for k in 0..self.frequencys {
            let freq = 2f32.powi(k as i32);
            for i in 0..n {
                let index = 2 * (k * n + i);
                let phase = input[i] * 2.0 * std::f32::consts::PI * freq;
                output[index + 0] = phase.sin();
                output[index + 1] = phase.cos();
            }
        }

        *input = output;
    }

    fn backward(&mut self, gradient: &mut Matrix) {
        // Does nothin!
    }

    fn get_params(&mut self) -> Vec<(&mut Matrix, &mut Matrix)> {
        return vec![];
    }

    fn get_params_shape(&self) -> Vec<(usize, usize)> {
        return vec![];
    }
}